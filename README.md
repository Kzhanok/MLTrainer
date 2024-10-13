# MLTrainer


This module is a NN trainer using mlflow as a method of tracking the training process.


module is designed to take data of any size or structure.


Building Guide:

1) build objective function -> takes in model, data, and hyperparameters and returns a loss value
2) Build statespace tuner (Ray or Pytorch Lightning)
3) Integration with MLFlow
4) allow control of different model settings (optimizer, loss function, etc)

**Note: use loguru for logging**

Current State:

Model is defined!
Trainer is built!
See MLtrainer -> Readme.md for full context