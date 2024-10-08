from datetime import datetime
from pathlib import Path
from typing import Iterator
import os
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor


def get_fashion_streamers(batchsize: int) -> tuple[Iterator, Iterator]:
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    preprocessor = BasePreprocessor()
    streamers = fashionfactory.create_datastreamer(
        batchsize=batchsize, preprocessor=preprocessor
    )
    train = streamers["train"]
    valid = streamers["valid"]
    len_train = len(train)
    len_valid = len(valid)
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    return trainstreamer, validstreamer, len_train, len_valid


def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        logger.info("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        logger.info("Using cuda")
    else:
        device = "cpu"
        logger.info("Using cpu")
    return device

import torchvision.models as models
# Define model, #TODO change to resnext
class CNN(nn.Module):
    def __init__(self, filters: int, units1: int, units2: int, input_size= (16,3,224,224)):
        super().__init__()
        self.in_channels = input_size[1]
        logger.info(self.in_channels)
        self.input_size = input_size
        logger.info(self.input_size)

        self.convolutions = nn.Sequential(
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size halved
            nn.Conv2d(filters, filters*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size halved again
            nn.Conv2d(filters*2, filters*3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters*3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2),
            # Output size halved once more
        )

        # Calculate the flattened size based on actual output shape after convolutions
        flattened_size = self._get_flattened_size(input_size)
        logger.info(f"Flattened size for the first Linear layer: {flattened_size}")

        # Remove AdaptiveAvgPool2d, as the tensor is already reduced
        self.dense = nn.Sequential(
            nn.Linear(flattened_size, units1),  # First fully connected layer
            nn.BatchNorm1d(units1),            # Batch normalization
            nn.ReLU(),                         # Activation function
            
            nn.Linear(units1, units2),          # Second fully connected layer
            nn.BatchNorm1d(units2),            # Batch normalization
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.BatchNorm1d(units2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(units2, units2),
            nn.BatchNorm1d(units2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(units2, 5)  # Output 10 classes
        )

    # This function calculates the flattened size after convolutions
    def _get_flattened_size(self, input_size):
        x = torch.ones(1, *input_size[1:], dtype=torch.float32)  # Add batch dimension
        x = self.convolutions(x)
        logger.info(f"Output shape after convolutions: {x.shape}")
        return x.numel()  # Return the total number of elements (flattened size)

    def forward(self, x):
        # Extract features using the ResNet convolutional layers
        x = self.resnext(x)
        
        # Pass the extracted features through the custom dense layers
        x = self.dense(x)
        return x
    
def setup_mlflow(experiment_path: str) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_path)
    


def objective(params: dict = None, filters: int = None, units1: int = None, units2: int = None):
    modeldir = Path("models").resolve()
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
        logger.info(f"Created {modeldir}")
    batchsize = 32
    trainstreamer, validstreamer, len_train, len_valid = get_fashion_streamers(batchsize)
    accuracy = metrics.Accuracy()
    settings = TrainerSettings(
        epochs=20,
        metrics=[accuracy],
        logdir=Path("modellog"),
        train_steps=len_train,
        valid_steps=len_valid,
        reporttypes=[ReportTypes.MLFLOW],
    )
    # Start a new MLflow run for tracking the experiment
    device = get_device()
    with mlflow.start_run():
        # Set MLflow tags to record metadata about the model and developer
        mlflow.set_tag("model", "convnet")
        mlflow.set_tag("Leon", "Smit")
        mlflow.set_tag('mlflow.runName', f'{datetime.now().strftime("%Y%m%d-%H%M")}')
        # Log hyperparameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("batchsize", f"{batchsize}")

        # Initialize the optimizer, loss function, and accuracy metric
        optimizer = optim.AdamW
        loss_fn = torch.nn.CrossEntropyLoss()

        # Instantiate the CNN model with the given hyperparameters
        model = CNN(filters, units1, units2)
        model.to(device)
        # Train the model using a custom train loop
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer,  # type: ignore
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )
        trainer.loop()

        # Save the trained model with a timestamp
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / (tag + "model.pt")
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)

        # Log the saved model as an artifact in MLflow
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")
        return {"loss": trainer.test_loss, "status": STATUS_OK}
    

from torchsummary import summary
def main():
    setup_mlflow(f"model run {datetime.now().strftime('%Y%m%d-%H%M')}")
    
    search_space = {
        "filters": scope.int(hp.quniform("filters", 64, 256, 8)),
        "units1": scope.int(hp.quniform("units1",256 , 512, 8)),
        "units2": scope.int(hp.quniform("units2", 128, 256, 8)),
    }
    objective(filters=64, units1=256, units2=128)
    best_result = fmin(
        fn=objective, space=search_space, algo=tpe.suggest, max_evals=5, trials=Trials()
    )

    logger.info(f"Best result: {best_result}")
    


if __name__ == "__main__":
    main()
