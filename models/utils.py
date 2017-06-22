
def set_trainable(model, train):
    model.trainable = train
    for l in model.layers:
        l.trainable = train
