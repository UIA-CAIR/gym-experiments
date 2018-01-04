from keras import Model
import keras.backend as K
from keras.layers import Conv2D, Input, Reshape, Flatten, Dense, Lambda
from keras.optimizers import Adam
from keras.utils import plot_model


def model(observation_space, action_space, lr):
    # Neural Net for Deep-Q learning Model

    # Image input
    input_layer = Input(shape=observation_space[1:], name='image_input')
    x = Conv2D(64, (8, 8), strides=(1, 1), activation='relu', data_format="channels_first")(
        input_layer)
    x = Conv2D(64, (4, 4), strides=(1, 1), activation='relu', data_format="channels_first")(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', data_format="channels_first")(x)
    #x = Conv2D(64, (2, 2), strides=(1, 1), activation='relu', data_format="channels_first")(x)
    #x = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', data_format="channels_first")(x)
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

    plot_model(model, to_file='./output/_dqn_model.png', show_shapes=True, show_layer_names=True)

    return model