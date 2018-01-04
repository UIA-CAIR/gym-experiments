import random

from PIL import Image
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.utils import plot_model
from tqdm import tqdm
from deeplinewars.training_data_loader import load_all_memories
from deeplinewars.rl.Memory import Memory
import numpy as np
from keras import Sequential, Input, Model
from keras.layers import Conv2D, Flatten, Dense, multiply, Reshape, BatchNormalization, UpSampling2D, Activation
import os
from matplotlib import pyplot as plt



class PlotLosses(Callback):
    def __init__(self):
        super(PlotLosses, self).__init__()
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        self.val_acc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))

        self.i += 1

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x, self.val_acc, label="val_acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./output/%s" % "supergan.png")

plot_losses = PlotLosses()

def stitch_generated_image(generated_images, name):
    # 4 x 4 Grid
    n = 4
    margin = 5
    i_w = 84
    i_h = 84
    i_c = 1
    width = n * i_w + (n - 1) * margin
    height = n * i_h + (n - 1) * margin
    stitched_filters = np.zeros((width, height, i_c))

    for i in range(n):
        for j in range(n):
            img = generated_images[i * n + j]

            stitched_filters[
            (i_w + margin) * i: (i_w + margin) * i + i_w,
            (i_h + margin) * j: (i_h + margin) * j + i_h, :] = img

    stitched_filters *= 255

    img = Image.fromarray(stitched_filters[:, :, 0])
    if img.mode != 'RGB':
        img = img.convert('RGB')
    name = 'yes_dqn_gan_%s.png' % name
    img.save('./output/tmp_' + name)
    os.rename('./output/tmp_' + name, './output/' + name)



action = Input(shape=(13, ), name="action")
image = Input(shape=(84, 84, 1), name="image")

img_conv = Sequential()
img_conv.add(Conv2D(256, (4, 4), strides=(2, 2), activation="relu", input_shape=(84, 84, 1)))
img_conv.add(Conv2D(128, (3, 3), strides=(2, 2), activation="relu"))
img_conv.add(Conv2D(64, (1, 1), strides=(2, 2), activation="relu"))
img_conv.add(Flatten())
img_conv.add(Dense(1024, activation="relu"))
img_conv.add(Dense(1024, activation="relu"))

action_s = Sequential()
action_s.add(Dense(1024, activation="relu", input_shape=(13, )))
action_s.add(Dense(1024, activation="relu"))

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

plot_model(model, to_file='./output/dlw_autoencoder_gan.eps', show_shapes=True, show_layer_names=False)

memory = Memory(10000000)
action_size = 13

X_0 = []
X_1 = []
Y = []




train_percent = .60

for item in load_all_memories(4):
    s, a, r, s1, t = item
    a_arr = np.zeros(shape=(action_size, ))
    a_arr[a] = 1
    s = np.reshape(s, tuple(reversed(s.shape[1:])))
    s1 = np.reshape(s1, tuple(reversed(s1.shape[1:])))

    X_1.append(s)
    X_0.append(a_arr)
    Y.append(s1)


n_train = int(len(X_0) * train_percent)
n_test = len(X_0) - n_train

X_train = [np.array(X_0[:n_train]), np.array(X_1[:n_train])]
Y_train = np.array(Y[:n_train])

X_test = [np.array(X_0[n_train:n_train+n_test]), np.array(X_1[n_train:n_train+n_test])]
Y_test = np.array(Y[n_train:n_train+n_test])

for i in range(10000):
    model.fit(X_train, Y_train, epochs=1, validation_data=(X_test, Y_test), callbacks=[plot_losses], verbose=1)

    n = 8
    indexes = random.sample(range(n_test), n)

    X = [np.array([X_test[0][_] for _ in indexes]), np.array([X_test[1][_] for _ in indexes])]
    X_pred = model.predict_on_batch(X)

    real = X[1]
    fake = X_pred[:n]

    print(real.shape, fake.shape)

    real_fake = []
    for j in range(len(real)):
        real_fake.append(real[j])
        real_fake.append(fake[j])

    stitch_generated_image(real_fake, "yes")


"""
for i in range(1000000):


    X_0 = []
    X_1 = []
    Y = []

    for s, a, r, s1, t in batch:
        X_0.append(s)
        X_1.append(a)
        Y.append(s1)

    X = [np.array(X_1), np.array(X_0)]
    Y = [s1 for s, a, r, s1, t in batch]

    loss = model.train_on_batch(X, [np.array(Y)])

    #model.fit(X, [np.array(Y)], validation_data=X_test, Y_test)


    if i % 50 == 0:
        X_pred = model.predict_on_batch(X)

        real = X[1][:8]
        fake = X_pred[:8]

        real_fake = []
        for j in range(len(real)):
            real_fake.append(real[j])
            real_fake.append(fake[j])

        stitch_generated_image(real_fake, "yes")

"""


