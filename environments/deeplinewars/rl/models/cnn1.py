from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D

from util.loss_functions import huber_loss_1


def model(state_size, action_size, learning_rate):
    model = Sequential()
    model.add(Conv2D(64, (1, 1), strides=(1, 1), data_format="channels_first", activation="relu", input_shape=state_size))
    model.add(Conv2D(64, (1, 1), strides=(1, 1), data_format="channels_first", activation="relu"))
    model.add(Conv2D(64, (1, 1), strides=(1, 1), data_format="channels_first", activation="relu"))
    #model.add(Conv2D(64, (1, 1), strides=(1, 1), data_format="channels_first", activation="relu"))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    model.compile(optimizer=Adam(lr=learning_rate), loss=huber_loss_1)
    return model