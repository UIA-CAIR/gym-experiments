from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import ConvLSTM2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.layers.core import Flatten, Dense

from util.loss_functions import huber_loss_1


def model(state_size, action_size, learning_rate):

    model = Sequential()
    model.add(ConvLSTM2D(64, (1, 1), return_sequences=True, strides=(3, 3), data_format="channels_first", activation=Activation("relu"), input_shape=(None, ) + state_size)) # (None, 2, 11, 11)
    model.add(ConvLSTM2D(64, (1, 1), return_sequences=True, strides=(2, 2), data_format="channels_first", activation=Activation("relu")))
    model.add(ConvLSTM2D(64, (1, 1), return_sequences=False, strides=(1, 1), data_format="channels_first", activation=Activation("relu")))
    model.add(Flatten())
    model.add(Dense(1024, activation=Activation("relu")))
    model.add(Dense(action_size, activation=Activation("linear")))

    model.compile(optimizer=Adam(lr=learning_rate), loss=huber_loss_1)
    return model