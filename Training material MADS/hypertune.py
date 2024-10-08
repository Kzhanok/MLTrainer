from pathlib import Path
from typing import Dict

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics, rnn_models
from mltrainer.preprocessors import PaddedPreprocessor
from ray import tune
from ray.tune import CLIReporter
from ray.air.integrations.mlflow import MLflowLoggerCallback  # Import MLflowLoggerCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float
NUM_SAMPLES = 2
MAX_EPOCHS = 50
import torch.nn as nn
import torch.hub
class CustomResNextModel(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.resnext = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        for param in self.resnext.parameters():
            param.requires_grad = False  # Freeze the ResNeXt layers
        num_ftrs = self.resnext.fc.in_features
        # Replace the final fully connected layer
        self.resnext.fc = nn.Identity()  # Remove the original fully connected layer
        self.fc1 = nn.Linear(num_ftrs, config["output_size"])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnext(x)
        x = self.fc1(x)
        return x

def train(config: Dict):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """
    from mads_datasets import DatasetFactoryProvider, DatasetType

    data_dir = config["data_dir"]
    gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    preprocessor = PaddedPreprocessor()

    with FileLock(data_dir / ".lock"):
        # We lock the datadir to avoid parallel instances trying to access it
        streamers = gesturesdatasetfactory.create_datastreamer(
            batchsize=32, preprocessor=preprocessor
        )
        train_streamer = streamers["train"]
        valid_streamer = streamers["valid"]

    # Set up the metric and create the model with the config
    accuracy = metrics.Accuracy()
    
    
    import torch.nn as nn
    import torch.hub

    

    model = CustomResNextModel(config)
    
    
    #model = rnn_models.GRUmodel(config)

    trainersettings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=[accuracy],
        logdir=Path("."),
        train_steps=len(train_streamer),  # type: ignore
        valid_steps=len(valid_streamer),  # type: ignore
        reporttypes=[ReportTypes.RAY],  # Reporting to Ray Tune
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None,
    )

    # Determine the device to use
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = "cpu"  # type: ignore
    logger.info(f"Using {device}")
    if device != "cpu":
        logger.warning(
            f"Using acceleration with {device}. Check if it actually speeds up!"
        )

    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,  # type: ignore
        traindataloader=train_streamer.stream(),
        validdataloader=valid_streamer.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        device=device,
    )

    trainer.loop()  # Start the training loop


if __name__ == "__main__":
    ray.init()

    data_dir = Path("data/raw/gestures/gestures-dataset").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("models/ray").resolve()
    search = HyperOptSearch()
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=1,
        reduction_factor=3,
        max_t=MAX_EPOCHS,
    )

    config = {
        "input_size": 3,
        "output_size": 5,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "test_number": tune.randint(1, 100),
        
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(config["tune_dir"]),
        num_samples=NUM_SAMPLES,
        search_alg=search,
        scheduler=scheduler,
        verbose=1,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="mlruns",  # Set your MLflow tracking URI
                experiment_name="my_experiment",  # Set your experiment name
            )
        ],
    )

    ray.shutdown()
