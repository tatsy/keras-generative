from keras import backend as K

def set_trainable(model, train):
    """
    Enable or disable training for the model
    """
    model.trainable = train
    for l in model.layers:
        l.trainable = train


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)
