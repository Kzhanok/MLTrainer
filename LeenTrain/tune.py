

def tune_model(config: dict, model_dict: dict):
    from metrics import Accuracy

    # load data
    train, valid = 
    trainsteps = 
    validsteps = 
    trainstreamer = 
    validstreamer = 

    # create model with config
    model = 
    optimizer = 
    loss_fn = 
    metric = 

    for _ in range(config["epochs"]):
        # train and validate
        train_loss = train_fn(model, trainstreamer, loss_fn, optimizer, trainsteps)
        valid_loss, accuracy = validate(model, validstreamer, loss_fn, metric, validsteps)

        # report to ray
        ray.train.report({
            "valid_loss": valid_loss / validsteps,
            "train_loss": train_loss / trainsteps,
            "accuracy" : accuracy,
            })