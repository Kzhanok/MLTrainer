from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple
import gin
import mlflow
# needed to make summarywriter load without error
from loguru import logger
import ray
import torch
from tqdm import tqdm

#TODO add logging and settings and finish loop
class Trainer(
    TrainingSettings = None,
    model = None,
    
    ):
    
    def __init__():
        pass
    
    def train(model, trainstreamer, lossfn, optimizer, steps):
        model.train()
        train_loss: float = 0.0
        for _ in range(steps):
            x, y = next(trainstreamer)
            optimizer.zero_grad()
            yhat = model(x)
            loss = lossfn(yhat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss

    def validate(model, validstreamer, lossfn, metric, steps):
        model.eval()
        valid_loss: float = 0.0
        acc: float = 0.0
        with torch.no_grad():
            for _ in range(steps):
                x, y = next(validstreamer)
                yhat = model(x)
                loss = lossfn(yhat, y)
                valid_loss += loss.item()
                acc += metric(y, yhat)
        acc /= steps
        return valid_loss, acc.item()

    def loop():
        pass
