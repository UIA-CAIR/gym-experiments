import random

from PIL import Image
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.utils import plot_model
from tqdm import tqdm

from deeplinewars.gan import GAN
from deeplinewars.training_data_loader import load_all_memories
from deeplinewars.rl.Memory import Memory
import numpy as np
from keras import Sequential, Input, Model
from keras.layers import Conv2D, Flatten, Dense, multiply, Reshape, BatchNormalization, UpSampling2D, Activation
import os
from matplotlib import pyplot as plt



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
        plt.subplot(1, 2, 1)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x, self.val_acc, label="val_acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./output/%s.png" % self.name)

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
    name = 'print_%s.png' % name
    img.save('./output/tmp_' + name)
    os.rename('./output/tmp_' + name, './output/' + name)


memory = Memory(10000000)
action_size = 13

X_0 = []
X_1 = []
Y = []

plot_adversarial = PlotLosses("adversarial")
plot_discriminator = PlotLosses("discriminator")

gan = GAN(action_size=action_size, state_size=(84, 84, 1))

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
batch_size = 16
for i in range(10000):


    indexes = random.sample(range(n_train), batch_size)
    real_a = np.array([X_train[0][x] for x in indexes])
    real_s0 = np.array([X_train[1][x] for x in indexes])
    real_s1 = np.array([Y_train[x] for x in indexes])


    fake_s1 = gan.generator_model.predict_on_batch([real_a, real_s0])

    real_fake = []
    for j in range(batch_size):
        real_fake.append(real_s0[j])
        real_fake.append(fake_s1[j])

    stitch_generated_image(real_fake, "full_gan")
    
    # Concatenate fake and real images
    x_0 = np.concatenate((real_s1, fake_s1))
    
    y = np.ones([2 * batch_size, 1])   # First half of image-set is REAL (0)
    y[batch_size:, :] = 0  # Last half of image-set is FAKE (0)

    gan.discriminator_model.fit([x_0], y, callbacks=[plot_discriminator])

    # Generate new noise
    y = np.ones([batch_size, 1])   # Set all images class to REAL
    indexes = random.sample(range(n_train), batch_size)
    real_a = np.array([X_train[0][x] for x in indexes])
    real_s0 = np.array([X_train[1][x] for x in indexes])
    real_s1 = np.array([Y_train[x] for x in indexes])

    a_loss = gan.adversarial_model.fit([real_a, real_s0], y, callbacks=[plot_adversarial])
