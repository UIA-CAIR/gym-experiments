import os
import random
import numpy as np
import os

from tensorflow import Session, ConfigProto
from tensorflow.python.keras import backend as K
from tensorflow.python.keras._impl.keras.callbacks import CSVLogger
from tensorflow.python.keras._impl.keras.optimizers import Adam

from msc.dqn.callbacks import ModelIntervalCheckpoint

K.set_session(Session(config=ConfigProto(inter_op_parallelism_threads=2)))

from msc.dqn.models import cnn

dir_path = os.path.dirname(os.path.realpath(__file__))



class Memory:

    def __init__(self, memory_size):
        self.buffer = []
        self.count = 0
        self.max_memory_size = memory_size

    def _recalibrate(self):
        self.count = len(self.buffer)

    def remove_n(self, n):
        self.buffer = self.buffer[n-1:-1]
        self._recalibrate()

    def add(self, memory):
        self.buffer.append(memory)
        self.count += 1

        if self.count > self.max_memory_size:
            self.buffer.pop(0)
            self.count -= 1

    def get(self, batch_size=1):
        if self.count <= batch_size:
            return np.array(self.buffer)

        return np.array(random.sample(self.buffer, batch_size))


class Agent:
    def __init__(self,
                 observation_space,
                 action_space,
                 model,
                 lr=1e-4,
                 memory_size=10000000,
                 e_start=1.0,
                 e_end=0.0,
                 e_steps=100000,
                 batch_size=16,
                 discount=0.99,
                 use_double=False,
                 ):

        # File definitions
        self.checkpoint_file = os.path.join(dir_path, "checkpoint.hdf5")
        self.logger_file = os.path.join(dir_path, "log.csv")

        self.use_double = use_double

        self.observation_space = observation_space
        self.action_space = action_space

        self.memory = Memory(memory_size)

        # Hyperparameters
        self.LEARNING_RATE = lr
        self.BATCH_SIZE = batch_size
        self.GAMMA = discount

        # Epsilon decent
        self.EPSILON_START = e_start
        self.EPSILON_END = e_end
        self.EPSILON_DECAY = (self.EPSILON_END - self.EPSILON_START) / e_steps
        self.epsilon = self.EPSILON_START

        self.model = model(self.observation_space, self.action_space, self.LEARNING_RATE)
        self.target_model = model(self.observation_space, self.action_space, self.LEARNING_RATE) if self.use_double else None

        # Compile models
        optimizer = Adam(lr=self.LEARNING_RATE)
        loss = "mse"
        metrics = ["accuracy"]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        if self.use_double:
            self.target_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)



        # Callbacks
        checkpoint = ModelIntervalCheckpoint(self.checkpoint_file, interval=50, verbose=1)
        logger = CSVLogger(self.logger_file, separator=',', append=True)
        self.model_callbacks = [checkpoint, logger]

        print("State size is: %s,%s,%s" % self.observation_space)
        print("Action size is: %s" % self.action_space)
        print("Batch size is: %s " % self.BATCH_SIZE)

    def reset(self):
        pass

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def train(self):
        if self.memory.count < self.BATCH_SIZE:
            return

        # Define which models to do updates on
        m1 = self.model
        m2 = self.target_model if self.use_double else self.model

        # Define inputs to model, and which targets (outputs) to predict
        inputs = np.zeros(((self.BATCH_SIZE,) + self.observation_space))
        targets = np.zeros((self.BATCH_SIZE, self.action_space))


        for i, (s, a, r, s1, terminal) in enumerate(self.memory.get(self.BATCH_SIZE)):
            target = r


            if not terminal:
                tar_s1 = m2.predict(s1)
                target = r + self.GAMMA * np.amax(tar_s1[0])

            targets[i] = m2.predict(s)
            targets[i, a] = target
            inputs[i] = s

        history = m1.fit(inputs, targets, epochs=1, callbacks=self.model_callbacks, verbose=0)

        #if self.ddqn and self.episode_train_count % 50 == 0:
        #    self.update_target_model()

    def act(self, state):
        self.epsilon = max(self.EPSILON_END, self.epsilon + self.EPSILON_DECAY)

        # Epsilon exploration
        if np.random.uniform() <= self.epsilon:
            return random.randrange(self.action_space)

        # Exploit Q-Knowledge
        act_values = self.target_model.predict(state) if self.use_double else self.model.predict(state)

        return np.argmax(act_values[0])


if __name__ == "__main__":
    agent = Agent((84, 84, 1), 4, cnn)

    for x in range(1000):
        agent.memory.add((
            np.zeros((1, 84, 84, 1)),
            random.randint(0, 3),
            random.randrange(-1, 1),
            np.zeros((1, 84, 84, 1)),
            False
        ))

    for x in range(10000):
        agent.train()
