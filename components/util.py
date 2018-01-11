from tensorflow.python.keras.callbacks import Callback
from matplotlib import pyplot as plt
import os
import glob
import random
from tqdm import tqdm
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


class PlotLosses(Callback):
    def __init__(self, name):
        super(PlotLosses, self).__init__()
        self.name = name
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
        os.makedirs(os.path.join(dir_path, "plots"), exist_ok=True)
        plt.savefig(os.path.join(dir_path, "plots", self.name + "_plot"))


def load_all_memories(training_data_file, limit=None):
    memories = []
    files = glob.glob(os.path.join(dir_path, "training_data", training_data_file))
    if limit is not None:
        files = files[:limit]
    for file in tqdm(files, desc="Loading experience buffer from %s " % training_data_file):
        with open(file, "rb") as f:
            data = np.load(f)
            memories.extend(data)

    return memories