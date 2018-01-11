from tensorflow.python.keras import Input

from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, multiply, Reshape, UpSampling2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import plot_model


def model_1(conditional_size):
    action = Input(shape=(conditional_size, ), name="action")
    image = Input(shape=(84, 84, 1), name="image")

    img_conv = Sequential()
    img_conv.add(Conv2D(256, (4, 4), strides=(2, 2), activation="relu", input_shape=(84, 84, 1)))
    img_conv.add(Conv2D(128, (3, 3), strides=(2, 2), activation="relu"))
    img_conv.add(Conv2D(64, (1, 1), strides=(2, 2), activation="relu"))
    img_conv.add(Flatten())
    img_conv.add(Dense(64, activation="relu"))
    img_conv.add(Dense(64, activation="relu"))

    action_s = Sequential()
    action_s.add(Dense(64, activation="relu", input_shape=(conditional_size, )))
    action_s.add(Dense(64, activation="relu"))

    stream_1 = img_conv(image)
    stream_2 = action_s(action)

    x = multiply([stream_1, stream_2])
    x = Dense(128 * 21 * 21, activation="relu")(x)
    x = Reshape((21, 21, 128))(x)
    #x = BatchNormalization(momentum=0.8)(x)

    x = UpSampling2D()(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    #x = BatchNormalization(momentum=0.8)(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    #x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(1, (3, 3), padding='same', activation="relu")(x)

    model = Model(inputs=[action, image], outputs=[x])
    model.compile(
        optimizer=Adam(0.00001),
        loss=['mse'],
        metrics=['accuracy']
    )
    model.summary()

    return model

    #plot_model(model, to_file='./output/conv_deconv_model.eps', show_shapes=True, show_layer_names=False)