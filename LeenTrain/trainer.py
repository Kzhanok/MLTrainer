import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from model import Model
from torchvision import datasets, transforms
from loguru import logger
from settings import TrainerSettings
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from datetime import datetime
import metrics


def dir_add_timestamp(log_dir= None) -> Path:
    if log_dir is None:
        log_dir = Path(".")
    log_dir = Path(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = log_dir / timestamp
    logger.info(f"Logging to {log_dir}")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    return log_dir

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_fn ,train_loader, valid_loader,settings: TrainerSettings):
        self.config = config
        self.device = config['trainer']['device']
        
        self.model = model
        self.settings = settings
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        if self.device:
            self.model.to(self.device)
        self.last_epoch = 0
        self.optimizer = optimizer(  # type: ignore
            model.parameters(), **settings.optimizer_kwargs
        )
        # Pass the optimizer instance to the scheduler, not model parameters
        self.scheduler = scheduler(self.optimizer, **settings.scheduler_kwargs)
        if self.settings.save_model:
            self.log_dir = dir_add_timestamp(settings.logdir)
        else:
            self.log_dir = settings.logdir
        if settings.earlystop_kwargs is not None:
            logger.info(
                "Found earlystop_kwargs in settings."
                "Set to None if you don't want early stopping."
            )
            self.early_stopping = EarlyStopping(
                self.log_dir, **settings.earlystop_kwargs
            )
        else:
            self.early_stopping = None
    def loop(self):
        """
        amount of epochs
        #TODO add early stopping
        """
        
        for epoch in tqdm(range(self.settings.epochs)):
            train_loss = self.train_batch()
            metric_dict, test_loss = self.evaluate_batch()
            self.report(epoch, train_loss, test_loss, metric_dict)

            if self.early_stopping:
                self.early_stopping(test_loss, self.model)  # type: ignore

            if self.early_stopping is not None and self.early_stopping.early_stop:
                logger.info("Interrupting loop due to early stopping patience.")
                self.last_epoch = epoch
                if self.early_stopping.save:
                    logger.info("retrieving best model.")
                    self.model = self.early_stopping.get_best()  # type: ignore
                else:
                    logger.info(
                        "early_stopping_save was false, using latest model."
                        "Set to true to retrieve best model."
                    )
                break
        self.last_epoch = epoch

    def train_batch(self):
        """
        train the model
        """
        self.model.train()
        train_loss: float = 0.0
        train_steps = self.settings.train_steps
        for i in tqdm(range(train_steps)):
            x, y = next(iter(self.train_loader)) 
            if self.device:
                x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.cpu().detach().numpy()
        train_loss /= train_steps
        
        return train_loss
            
            
    def evaluate_batch(self):
        """
        Evaluate the model at the end of an train cycle
        """
        self.model.eval()
        valid_steps = self.settings.valid_steps
        test_loss: float = 0.0
        metric_dict = {}
        for i in range(valid_steps):
            x, y = next(iter(self.valid_loader))
            if self.device:
                x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_loss += self.loss_fn(output, y).cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            for m in self.settings.metrics:
                metric_dict[str(m)] = metric_dict.get(str(m), 0.0) + m(y, output)
        test_loss /= valid_steps
        
        if self.scheduler:
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(test_loss)
            else:
                self.scheduler.step()
        for key in metric_dict:
            metric_dict[str(key)] = metric_dict[str(key)] / valid_steps
        return metric_dict, test_loss

    def report(
        self, epoch: int, train_loss: float, test_loss: float, metric_dict: Dict
    ) -> None:
        epoch = epoch + self.last_epoch
        
        self.test_loss = test_loss
        metric_scores = [f"{v:.4f}" for v in metric_dict.values()]
        logger.info(
            f"Epoch {epoch} train {train_loss:.4f} test {test_loss:.4f} metric {metric_scores}"  # noqa E501
        )

class EarlyStopping:
    def __init__(
        self,
        log_dir: Path,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        save: bool = False,
        early_stop: bool = False,
    ) -> None:
        """
        Args:
            log_dir (Path): location to save checkpoint to.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss
            improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as
            an improvement. Default: 0.0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = early_stop
        self.delta = delta
        self.path = Path(log_dir) / "checkpoint.pt"
        self.save = save
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        
        # first epoch best_loss is still None
        if self.best_loss is None:
            self.best_loss = val_loss  # type: ignore
            if self.save:
                self.save_checkpoint(val_loss, model)
                logger.info(f"Saving model to {self.path}")
        elif val_loss >= self.best_loss + self.delta:  # type: ignore
            # we minimize loss. If current loss did not improve
            # the previous best (with a delta) it is considered not to improve.
            self.counter += 1
            logger.info(
                f"best loss: {self.best_loss:.4f}, current loss {val_loss:.4f}."
                f"Counter {self.counter}/{self.patience}."
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # if not the first run, and val_loss is smaller, we improved.
            self.best_loss = val_loss
            if self.save:
                self.save_checkpoint(val_loss, model)
                logger.info(f"Saving model to {self.path}")
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            logger.info(
                f"Validation loss ({self.best_loss:.4f} --> {val_loss:.4f})."
                f"Saving {self.path} ..."
            )
        logger.info(f"Saving model to {self.path}")
        torch.save(model, self.path)
        self.val_loss_min = val_loss

    def get_best(self) -> torch.nn.Module:
        return torch.load(self.path)
        
def main():
    config = load_config('LeenTrain/config.yaml')
    device = torch.device(config['trainer']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info('building model')
    model = Model(config, device=device)
    logger.info('model built')
    model.output_model_summary()
    

    #TODO build datastreamer
    from mads_datasets import DatasetFactoryProvider, DatasetType
    from mltrainer.preprocessors import BasePreprocessor
    preprocessor = BasePreprocessor()
    factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    streamers = factory.create_datastreamer(batchsize=32, preprocessor=preprocessor)
    train = streamers["train"]
    valid = streamers["valid"]
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    
    # Instantiating optimizer with model parameters
    optimizer = optim.Adam
    
    # Instantiating scheduler with optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau
    
    # Loss function and accuracy metrics
    loss_fn = nn.CrossEntropyLoss()
    accuracy = metrics.Accuracy()
    
    # Trainer settings
    settings = TrainerSettings(
        epochs=10,
        metrics=[accuracy],
        logdir=Path("."),
        train_steps=len(train),
        valid_steps=len(valid),
        save_model=False,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-5},
        scheduler_kwargs={'factor': 0.1, 'patience': 10},
        earlystop_kwargs={"save": False, "verbose": True, "patience": 10},
    )
    
    # Instantiate Trainer with optimizer and scheduler objects
    trainer = Trainer(config=config, model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, settings=settings, train_loader=trainstreamer, valid_loader=validstreamer)
    trainer.loop()
    
if __name__ == "__main__":
    main()
