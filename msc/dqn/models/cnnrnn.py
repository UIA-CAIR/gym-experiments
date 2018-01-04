from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, LSTM, TimeDistributed
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.optimizers import Adam

from maze import util


def model(state_size, action_size, learning_rate):
    model = Sequential()
    model.add(ConvLSTM2D(64, (1, 1), return_sequences=True, strides=(1, 1), data_format="channels_first", activation="relu", input_shape=(None, 2, 11, 11)))
    model.add(ConvLSTM2D(64, (1, 1), return_sequences=True, strides=(1, 1), data_format="channels_first", activation="relu"))
    model.add(ConvLSTM2D(64, (1, 1), return_sequences=False, strides=(1, 1), data_format="channels_first", activation="relu"))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(action_size, activation="linear"))

    model.compile(optimizer=Adam(lr=learning_rate), loss=util.huber_loss_1)
    return model