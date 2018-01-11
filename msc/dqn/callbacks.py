from tensorflow.python.keras.callbacks import Callback as KerasCallback
import time

class Callback(KerasCallback):
    """
    on_epoch_begin: called at the beginning of every epoch.
    on_epoch_end: called at the end of every epoch.
    on_batch_begin: called at the beginning of every batch.
    on_batch_end: called at the end of every batch.
    on_train_begin: called at the beginning of model training.
    on_train_end: called at the end of model training.

    """
    def __init__(self):
        super().__init__()
        self.duration = 0
        self._start = None
        self.epoch = 0

    def on_train_begin(self,  logs={}):
        self._start = time.time()

    def on_train_end(self, logs={}):
        self.duration = time.time() - self._start
        self.epoch += 1


class ModelIntervalCheckpoint(Callback):
    def __init__(self, filepath, interval, verbose=0):
        super(ModelIntervalCheckpoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.total_steps = 0

    def on_train_end(self, logs={}):
        super().on_train_end(logs)
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbose > 0:
            print('Step {}: saving model to {}'.format(self.total_steps, filepath))
        self.model.save_weights(filepath, overwrite=True)