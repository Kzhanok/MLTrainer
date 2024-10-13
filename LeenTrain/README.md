Each script has an example of how to call it below.

#TODO build datastreamer
#TODO add more metrics
#TODO build tuner
#TODO build mlflow implementation
#TODO build visualizer

Model.py:

Defines different blocks using functions that can be combined sequentially using the config file #TODO rewrite config.yaml to gin maybe?
Block definitions passes to .py file, can it be better?


Trainer.py

#example script
def main():
    config = load_config('LeenTrain/config.yaml')
    device = torch.device(config['trainer']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info('building model')
    model = Model(config, device=device)
    logger.info('model built')
    model.output_model_summary()
    
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