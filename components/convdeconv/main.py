import random
from py_image_stitcher import ImageStitch
from scipy.misc import imresize

from components.util import load_all_memories, PlotLosses
from components.convdeconv.model import model_1
import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


datasets = [("flashrl", "flashrl*.npy", 4), ("deepmaze", "deepmaze*.npy", 4), ("flappybird", "flappybird*.npy", 2), ("deeplinewars", "deeplinewars*.npy", 13)]
train_percent = .60

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

for name, dataset, action_size in datasets:

    plot_losses = PlotLosses(name)
    model = model_1(action_size)

    X_0 = []  # Condition
    X_1 = []  # Image
    Y = []    # Transitioned Image

    for item in load_all_memories(dataset, 20):
        s, a, r, s1, t = item
        a_arr = np.zeros(shape=(action_size, ))

        if name == "flappybird":
            if a == 119: # FLappy bird
                a_arr[0] = 1
            elif a == None:
                a_arr[1] = 1
            s = np.reshape(s, (84, 84, 1))
            s1 = np.reshape(s1, (84, 84, 1))
        elif name == "deepmaze":
            a_arr[a] = 1
            s = np.reshape(s, (80, 80, 3))
            s1 = np.reshape(s1, (80, 80, 3))
            s = rgb2gray(s)
            s1 = rgb2gray(s1)
            s = imresize(s, (84, 84))
            s1 = imresize(s1, (84, 84))
            s = np.reshape(s, (84, 84, 1))
            s1 = np.reshape(s1, (84, 84, 1))
        elif name == "flashrl":
            a_arr[["w", "a", "s", "d"].index(a)] = 1
        else:
            a_arr[a] = 1
            s = np.reshape(s, tuple(reversed(s.shape[1:])))
            s1 = np.reshape(s1, tuple(reversed(s1.shape[1:])))

        X_0.append(a_arr)
        X_1.append(s)

        Y.append(s1)


    n_train = int(len(X_0) * train_percent)
    n_test = len(X_0) - n_train

    X_train = [np.array(X_0[:n_train]), np.array(X_1[:n_train])]
    Y_train = np.array(Y[:n_train])

    X_test = [np.array(X_0[n_train:n_train+n_test]), np.array(X_1[n_train:n_train+n_test])]
    Y_test = np.array(Y[n_train:n_train+n_test])



    for i in range(10000):
        model.fit(X_train, Y_train, epochs=1, validation_data=(X_test, Y_test), callbacks=[plot_losses], verbose=1)

        n = int(((8*6) / 2))
        indexes = random.sample(range(n_test), n)

        X = [np.array([X_test[0][_] for _ in indexes]), np.array([X_test[1][_] for _ in indexes])]
        X_pred = model.predict_on_batch(X)

        real = X[1]
        fake = X_pred[:n]

        print(real.shape, fake.shape)

        stitch = ImageStitch((84, 84), rows=8, columns=6, spacing=(3, 2))

        real_fake = []
        for j in range(len(real)):
            the_real = real[j] * 255
            the_fake = fake[j] * 255
            stitch.add(the_real.astype('uint8'))
            stitch.add(the_fake.astype('uint8'))

        stitch.save(os.path.join(dir_path, "..", "plots", name + "_convdeconv_" + str(i) + ".png"))
        stitch.save(os.path.join(dir_path, "..", "plots", name + "_convdeconv_" + str(i) + ".eps"))



