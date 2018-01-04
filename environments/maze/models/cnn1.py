from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam

from maze import util


def model(state_size, action_size, learning_rate):
    model = Sequential()
    model.add(Conv2D(256, (1, 1), strides=(1, 1), activation="relu", input_shape=state_size))
    model.add(Conv2D(256, (1, 1), strides=(1, 1), activation="relu"))
    model.add(Conv2D(256, (1, 1), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    model.compile(optimizer=Adam(lr=learning_rate), loss=util.huber_loss_1)
    return model, "cnn1"