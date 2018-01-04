from __future__ import print_function

from PIL import Image
from keras.callbacks import Callback, CSVLogger
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
from deeplinewars.training_data_loader import load_all_memories

class PlotLosses(Callback):
    def __init__(self, name):
        super(PlotLosses, self).__init__()
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []
        self.name = name

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
        plt.plot(self.x, self.losses, label="Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./output/%s_loss.png" % self.name)

        plt.clf()
        plt.plot(self.x, self.acc, label="Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./output/%s_acc.png" % self.name)


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
    img.save('./output/print_%s.png' % name)


class DCGAN():
    def __init__(self):
        self.img_rows = 84
        self.img_cols = 84
        self.channels = 1

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='mse', optimizer=optimizer)

        #self.plot_a_csv = CSVLogger("discriminator.log", append=True)
        #self.plot_d_csv = CSVLogger
        self.plot_d = PlotLosses("discriminator")
        self.plot_a = PlotLosses("adversarial")

        # The generator takes noise as input and generated imgs
        action = Input(shape=(13, ), name="action")
        image = Input(shape=(84, 84, 1), name="image")

        img = self.generator([action, image])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([action, image], valid)
        self.combined.compile(loss='mse', optimizer=optimizer)

    def build_generator(self):

        action = Input(shape=(13, ), name="action")
        image = Input(shape=(84, 84, 1), name="image")

        img_conv = Sequential()
        img_conv.add(Conv2D(256, (4, 4), strides=(2, 2), input_shape=(84, 84, 1)))
        img_conv.add(LeakyReLU(alpha=0.2))
        img_conv.add(Conv2D(128, (3, 3), strides=(2, 2)))
        img_conv.add(LeakyReLU(alpha=0.2))
        img_conv.add(Conv2D(64, (1, 1), strides=(2, 2)))
        img_conv.add(LeakyReLU(alpha=0.2))
        img_conv.add(Flatten())
        img_conv.add(Dense(1024))
        img_conv.add(LeakyReLU(alpha=0.2))
        #img_conv.add(Dense(1024))
        #img_conv.add(LeakyReLU(alpha=0.2))

        action_s = Sequential()
        action_s.add(Dense(1024, input_shape=(13, )))
        action_s.add(LeakyReLU(alpha=0.2))
        #action_s.add(Dense(1024))
        #action_s.add(LeakyReLU(alpha=0.2))

        stream_1 = img_conv(image)
        stream_2 = action_s(action)

        x = multiply([stream_1, stream_2])
        x = Dense(128 * 21 * 21)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((21, 21, 128))(x)
        #x = BatchNormalization(momentum=0.8)(x)

        x = UpSampling2D()(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        #x = BatchNormalization(momentum=0.8)(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        #x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(1, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

        model = Model(inputs=[action, image], outputs=[x])
        return model

        """noise_shape = (100,)

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_shape=noise_shape))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)"""

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        train_percent = .60
        action_size = 13

        X_0 = []
        X_1 = []
        Y = []

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


        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train[0].shape[0], half_batch)

            real_a = X_train[0][idx]
            real_s0 = X_train[1][idx]
            real_s1 = Y_train[idx]

            # Sample noise and generate a half batch of new images
            #noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict([real_a, real_s0])




            # Train the discriminator (real classified as ones and generated as zeros)
            d_history_real = self.discriminator.fit(real_s1, np.ones((half_batch, 1)), callbacks=[self.plot_d], verbose=0)
            d_history_fake = self.discriminator.fit(gen_imgs, np.zeros((half_batch, 1)), callbacks=[self.plot_d], verbose=0)
            d_acc_real = d_history_real.history["acc"][0]
            d_loss_real = d_history_real.history["loss"][0]
            d_acc_fake = d_history_real.history["acc"][0]
            d_loss_fake = d_history_real.history["loss"][0]
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)


            # ---------------------
            #  Train Generator
            # ---------------------
            idx = np.random.randint(0, X_test[0].shape[0], batch_size)
            _real_a = X_test[0][idx]
            _real_s0 = X_test[1][idx]
            _real_s1 = Y_test[idx]
            #noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.fit([_real_a, _real_s0], np.ones((batch_size, 1)), callbacks=[self.plot_a], verbose=0)
            g_loss = g_loss.history["loss"][0]

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, 100*d_acc, g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                real_fake = []

                for j in range(len(gen_imgs)):
                    real_fake.append(real_s0[j])
                    real_fake.append(gen_imgs[j])

                stitch_generated_image(real_fake, "full_gan")

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=40000, batch_size=128, save_interval=50)





