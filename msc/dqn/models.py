from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.layers.core import Flatten, Dense


def capsnet(state_size, action_size, learning_rate):
    n_routing = 3

    x = Input(shape=state_size)
    conv1 = Conv2D(filters=256, kernel_size=1, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=action_size, dim_vector=16, num_routing=n_routing, name='digitcaps')(primarycaps)
    out_caps = Length(name='out_caps')(digitcaps)

    model = Model(inputs=[x], outputs=[out_caps])
    model.compile(optimizer=Adam(lr=learning_rate), loss=util.huber_loss_1)
    return model, "capsule1"


def cnn_dualing(observation_space, action_space, lr):
    # Neural Net for Deep-Q learning Model

    # Image input
    input_layer = Input(shape=observation_space[1:], name='image_input')
    x = Conv2D(64, (8, 8), strides=(1, 1), activation='relu', data_format="channels_first")(
        input_layer)
    x = Conv2D(64, (4, 4), strides=(1, 1), activation='relu', data_format="channels_first")(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', data_format="channels_first")(x)
    x = Reshape((int(x.shape[1]), int(x.shape[2]), int(x.shape[3])))(x)
    x = Flatten()(x)

    # Value Stream
    vs = Dense(512, activation="relu", kernel_initializer='uniform')(x)
    vs = Dense(1, kernel_initializer='uniform')(vs)

    # Advantage Stream
    ad = Dense(512, activation="relu", kernel_initializer='uniform')(x)
    ad = Dense(action_space, kernel_initializer='uniform', activation="linear")(ad)

    policy = Lambda(lambda w: w[0] - K.mean(w[0]) + w[1])([vs, ad])

    model = Model(inputs=[input_layer], outputs=[policy])
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss="mse")

    return model


def cnn(state_size, action_size, learning_rate):
    model = Sequential()
    model.add(Conv2D(32, (1, 1), strides=(1, 1), activation=Activation("relu"), input_shape=state_size))
    model.add(Conv2D(32, (1, 1), strides=(1, 1), activation=Activation("relu")))
    model.add(Conv2D(32, (1, 1), strides=(1, 1), activation=Activation("relu")))
    model.add(Flatten())
    model.add(Dense(512, activation=Activation("relu")))
    model.add(Dense(action_size, activation=Activation("linear")))
    return model

def cnn_rnn(state_size, action_size, learning_rate):
    model = Sequential()
    model.add(ConvLSTM2D(64, (1, 1), return_sequences=True, strides=(1, 1), data_format="channels_first", activation="relu", input_shape=(None, 2, 11, 11)))
    model.add(ConvLSTM2D(64, (1, 1), return_sequences=True, strides=(1, 1), data_format="channels_first", activation="relu"))
    model.add(ConvLSTM2D(64, (1, 1), return_sequences=False, strides=(1, 1), data_format="channels_first", activation="relu"))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(action_size, activation="linear"))

    model.compile(optimizer=Adam(lr=learning_rate), loss=util.huber_loss_1)
    return model