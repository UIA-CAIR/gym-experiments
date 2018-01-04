import numpy as np
from PIL import Image
from keras import Model, Input, Sequential
from keras.layers import GaussianNoise, Dense, Conv2D, LeakyReLU, Dropout, Reshape, Flatten, Concatenate, Activation, \
    BatchNormalization, UpSampling2D, Conv2DTranspose, Embedding, multiply, ZeroPadding2D, Multiply
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import plot_model
import os


import numpy as np
from PIL import Image


class GAN:
    def __init__(
            self,
            batch_size=16,
            state_size=None,
            action_size=None,
            a_lr=0.00001, #4e-4,
            a_decay=3e-16,
            g_lr=5e-3,
            g_decay=6e-8,
            g_dropout=0.4,
            g_momentum=0.9,
            latent_size=100,
            d_lr=5e-3,
            d_decay=6e-8,
            d_dropout=0.3,
            d_depth=64
    ):
        self.BATCH_SIZE = batch_size
        self.state_size = state_size
        self.action_size = action_size

        # Adversarial Parameters
        self.a_lr = a_lr
        self.a_decay = a_decay

        # Generator Parameters
        self.g_lr = g_lr
        self.g_decay = g_decay
        self.g_dropout = g_dropout
        self.g_momentum = g_momentum
        self.latent_size = latent_size

        # Discriminator Parameters
        self.d_lr = d_lr
        self.d_decay = d_decay
        self.d_dropout = d_dropout
        self.d_depth = d_depth

        # Generator
        self.generator_model = self.build_generator_model()
        self.generator_model.compile(loss=["binary_crossentropy"], optimizer=Adam(lr=g_lr, decay=g_decay), metrics=['accuracy'])

        # Discriminator
        self.discriminator_model = self.build_discriminator_model()
        self.discriminator_model.compile(loss=["binary_crossentropy"], optimizer=Adam(lr=d_lr, decay=d_decay), metrics=['accuracy'])


        self.adversarial_model = self.build_adversarial(self.generator_model, self.discriminator_model)
        self.adversarial_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.a_lr, clipvalue=1.0, decay=self.a_decay), metrics=['accuracy'])
        self.adversarial_model.summary()

    def build_adversarial(self, generator, discriminator):
        model = Model(inputs=generator.inputs, outputs=discriminator(generator.output))
        plot_model(model, to_file='./output/_adversarial_model.png', show_shapes=True, show_layer_names=True)
        return model

    def build_discriminator_model(self):
        img_shape = (84, 84, 1)

        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        plot_model(model, to_file='./output/_discriminator_model.png', show_shapes=True, show_layer_names=True)

        return model

    def build_generator_model(self):

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
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        #x = BatchNormalization(momentum=0.8)(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        #x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(1, (3, 3), padding='same')(x)
        x = Activation("relu")(x)

        model = Model(inputs=[action, image], outputs=[x])

        plot_model(model, to_file='./output/_generator_model.png', show_shapes=True, show_layer_names=True)
        return model

    def stitch_generated_image(self, generated_images, name):
        # 4 x 4 Grid
        n = 4
        margin = 5
        i_w = self.state_size[0]
        i_h = self.state_size[1]
        i_c = self.state_size[2]
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
        img.save('./output/dqn_gan_%s.png' % name)



"""
class GAN:
    def __init__(
            self,
            batch_size=16,
            state_size=None,
            action_size=None,
            a_lr=4e-4,
            a_decay=3e-16,
            g_lr=5e-3,
            g_decay=6e-8,
            g_dropout=0.4,
            g_momentum=0.9,
            latent_size=100,
            d_lr=5e-3,
            d_decay=6e-8,
            d_dropout=0.3,
            d_depth=64,
    ):
        self.BATCH_SIZE = batch_size
        self.state_size = state_size
        self.action_size = action_size


        # Adversarial Parameters
        self.a_lr = a_lr
        self.a_decay = a_decay

        # Generator Parameters
        self.g_lr = g_lr
        self.g_decay = g_decay
        self.g_dropout = g_dropout
        self.g_momentum = g_momentum
        self.latent_size = latent_size

        # Discriminator Parameters
        self.d_lr = d_lr
        self.d_decay = d_decay
        self.d_dropout = d_dropout
        self.d_depth = d_depth

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator_model()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        image, action, self.generator = self.build_generator_model()
        self.generator.compile(loss=['binary_crossentropy'],
                               optimizer=optimizer)

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        #noise = Input(shape=state_size)
        #label = Input(shape=(action_size, ))
        #img = self.generator()

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(image)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([image, action], [valid, target_label])
        self.combined.compile(loss=losses,
                              optimizer=optimizer)
        plot_model(self.combined, to_file='./output/_adversarial_model.png', show_shapes=True, show_layer_names=True)

    def train(self):
        if self.memory.count < self.BATCH_SIZE:
            return

        experience_replay = self.memory.get(self.BATCH_SIZE)
        real_state = np.array([s for s, a, r, s1, t in experience_replay])
        real_state1 = np.array([s1 for s, a, r, s1, t in experience_replay])
        real_action = np.array([a for s, a, r, s1, t in experience_replay])

        # Generate fake state and action

        fake_state = self.generator.predict([real_state, real_action])

        self.stitch_generated_image(real_state, "real")
        self.stitch_generated_image(fake_state, "fake")

        # Concatenate fake and real images
        x_0 = np.concatenate((real_state, fake_state))

        y = np.ones([2 * self.BATCH_SIZE, 1])   # First half of image-set is REAL (0)
        y[self.BATCH_SIZE:, :] = 0  # Last half of image-set is FAKE (0)

        d_loss = self.discriminator.train_on_batch([real_state, real_action], y)

        # Generate new noise
        y = np.ones([self.BATCH_SIZE, 1])   # Set all images class to REAL
        noise = np.random.uniform(-1.0, 1.0, size=[self.BATCH_SIZE, self.latent_size])

        a_loss = self.combined.train_on_batch([noise, real_action], y)

        return d_loss, a_loss

        # Compare Q values
        #new_real_q_values = self.q_model.predict(real_s)
        #fake_q_values = self.q_model.predict(fake_s)
        #q_loss_mse = ((fake_q_values - new_real_q_values) ** 2).mean(axis=None)

    def build_adversarial(self, generator, discriminator):

        model = Model(inputs=generator.input, outputs=[discriminator(generator.output)])

        optimizer = RMSprop(lr=self.a_lr, decay=self.a_decay)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        plot_model(model, to_file='./output/_adversarial_model.png', show_shapes=True, show_layer_names=True)
        return model


    def build_discriminator_model(self):

        img_shape = (84, 84, 1)

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        print("discriminator")
        model.summary()

        img = Input(shape=img_shape)

        features = model(img)

        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.action_size+1, activation="softmax")(features)
        model = Model(img, [validity, label])
        plot_model(model, to_file='./output/_discriminator_model.png', show_shapes=True, show_layer_names=True)

        return model

    def build_generator_model(self):
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
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        #x = BatchNormalization(momentum=0.8)(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        #x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(1, (3, 3), padding='same')(x)
        x = Activation("relu")(x)

        model = Model(inputs=[action, image], outputs=[x])
        model.summary()
        plot_model(model, to_file='./output/_generator_model.png', show_shapes=True, show_layer_names=True)
        return image, action, model

    def stitch_generated_image(self, generated_images, name):
        # 4 x 4 Grid
        n = 4
        margin = 5
        i_w = self.state_size[0]
        i_h = self.state_size[1]
        i_c = self.state_size[2]
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
        name = 'dqn_gan_%s.png' % name
        img.save('./output/tmp_' + name)
        os.rename('./output/tmp_' + name, './output/' + name)
"""
