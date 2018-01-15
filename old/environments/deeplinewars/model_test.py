from keras import Sequential, Input, Model
from keras.layers import Conv2D, Flatten, Dense, multiply, Reshape, BatchNormalization, UpSampling2D, Activation

action = Input(shape=(13, ), name="action")
image = Input(shape=(84, 84, 1), name="image")

img_conv = Sequential()
img_conv.add(Conv2D(64, (3, 3), strides=(2, 2), activation="relu", input_shape=(84, 84, 1)))
img_conv.add(Conv2D(64, (3, 3), strides=(2, 2), activation="relu"))
img_conv.add(Conv2D(64, (3, 3), strides=(2, 2), activation="relu"))
img_conv.add(Flatten())
img_conv.add(Dense(512))

action_s = Sequential()
action_s.add(Dense(512, input_shape=(13, )))

stream_1 = img_conv(image)
stream_2 = action_s(action)

x = multiply([stream_1, stream_2])
x = Dense(128 * 21 * 21, activation="relu")(x)
x = Reshape((21, 21, 128))(x)
x = BatchNormalization(momentum=0.8)(x)

x = UpSampling2D()(x)
x = Conv2D(128, kernel_size=3, padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(momentum=0.8)(x)

x = UpSampling2D()(x)
x = Conv2D(64, kernel_size=3, padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(momentum=0.8)(x)

x = Conv2D(1, kernel_size=3, padding='same')(x)
x = Activation("tanh")(x)

model = Model(inputs=[action, image], outputs=[x])
model.summary()