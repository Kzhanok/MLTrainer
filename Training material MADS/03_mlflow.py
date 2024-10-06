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
# Define model
class CNNWithResNet(nn.Module):
    def __init__(self, units1: int, units2: int, filters=64):
        super().__init__()
        
        # Load a pre-trained ResNet model
        self.resnext = models.resnext50_32x4d(pretrained=True)
        
        # Optionally freeze the ResNet layers if you don't want to fine-tune them
        for param in self.resnext.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer of ResNet with an identity layer,
        # So that we can add our custom dense block after the feature extraction
        num_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Identity()  # We will handle the fully connected part later
        
        # Now, define your dense layers (similar to the dense block in your CNN)
        self.dense = nn.Sequential(
            nn.Linear(num_features, units1),  # First fully connected layer
            nn.BatchNorm1d(units1),            # Batch normalization
            nn.ReLU(),                         # Activation function
            
            nn.Linear(units1, units2),          # Second fully connected layer
            nn.BatchNorm1d(units2),            # Batch normalization
            nn.ReLU(),
            nn.Dropout(p=0.4),                 # Dropout for regularization
            
            nn.Linear(units2, 64),             # Optional layer to further reduce dimensions
            nn.ReLU(),
            nn.Dropout(p=0.4),                 # More dropout if overfitting is an issue
            
            nn.Linear(64, 5)                   # Final layer with 5 output classes
        )


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
    batchsize = 16
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
    if params is not None:
        with mlflow.start_run():
            # Set MLflow tags to record metadata about the model and developer
            mlflow.set_tag("model", "convnet")
            mlflow.set_tag("dev", "raoul")
            mlflow.set_tag('mlflow.runName', f'{datetime.now().strftime("%Y%m%d-%H%M")}')
            # Log hyperparameters to MLflow
            mlflow.log_params(params)
            mlflow.log_param("batchsize", f"{batchsize}")

            # Initialize the optimizer, loss function, and accuracy metric
            optimizer = optim.AdamW
            loss_fn = torch.nn.CrossEntropyLoss()

            # Instantiate the CNN model with the given hyperparameters
            model = CNNWithResNet(filters, units1, units2)
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
    else:
        with mlflow.start_run():
            # Set MLflow tags to record metadata about the model and developer
            mlflow.set_tag("model", "convnet")
            mlflow.set_tag("dev", "raoul")
            mlflow.set_tag('mlflow.runName', f'{datetime.now().strftime("%Y%m%d-%H%M")}')
            # Log hyperparameters to MLflow
            #mlflow.log_params(params)
            mlflow.log_param("batchsize", f"{batchsize}")

            # Initialize the optimizer, loss function, and accuracy metric
            optimizer = optim.AdamW
            loss_fn = torch.nn.CrossEntropyLoss()

            # Instantiate the CNN model with the given hyperparameters
            model = CNNWithResNet(filters, units1, units2)
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
        "filters": scope.int(hp.quniform("filters", 64, 128, 8)),
        "units1": scope.int(hp.quniform("units1", 256, 512, 8)),
        "units2": scope.int(hp.quniform("units2", 32, 128, 8)),
    }
    objective(filters=64, units1=256, units2=128)
    #best_result = fmin(
    #    fn=objective, space=search_space, algo=tpe.suggest, max_evals=5, trials=Trials()
    #)

    #logger.info(f"Best result: {best_result}")
    


if __name__ == "__main__":
    main()
