import os
import random
import numpy as np

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.models import load_model
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras.engine import Input, Model
from tensorflow.contrib.keras.python.keras.utils.vis_utils import plot_model
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
from tensorflow.contrib.keras.python.keras.layers.core import Flatten, Dense, Lambda, Reshape
from deeplinewars.rl.Memory import Memory


class Agent:
    def __init__(self,
                 observation_space,
                 action_space,
                 lr=1e-4,
                 memory_size=10000000,
                 e_start=1.0,
                 e_end=0.0,
                 e_steps=100000,
                 batch_size=16,
                 discount=0.99,
                 model=None,
                 ddqn=False
                 ):
        os.makedirs("./output", exist_ok=True)
        os.makedirs("./save", exist_ok=True)

        self.ddqn = ddqn


        self.observation_space = observation_space
        self.action_space = action_space

        self.memory = Memory(memory_size)

        # Parameters
        self.LEARNING_RATE = lr
        self.BATCH_SIZE = batch_size
        self.GAMMA = discount

        # Epsilon decent
        self.EPSILON_START = e_start
        self.EPSILON_END = e_end
        self.EPSILON_DECAY = (self.EPSILON_END - self.EPSILON_START) / e_steps
        self.epsilon = self.EPSILON_START

        self.episode_loss = 0
        self.episode_train_count = 0

        self.model = model

        if self.ddqn:
            self.target_model = self.build_model()

        self.model.summary()
        print("State size is: %s,%s,%s,%s" % self.observation_space)
        print("Action size is: %s" % self.action_space)
        print("Batch size is: %s " % self.BATCH_SIZE)

    def reset(self):
        self.episode_loss = 0
        self.episode_train_count = 0

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        raise Exception("implement model!")

    def load(self, name):
        self.model = load_model(name)
        self.target_model = load_model(name)

    def save(self, name):
        self.target_model.save(name)

    def train(self):
        if self.memory.count < self.BATCH_SIZE:
            return

        m1 = self.model
        m2 = self.target_model if self.ddqn else self.model

        inputs = np.zeros(((self.BATCH_SIZE,) + self.observation_space[1:]))
        targets = np.zeros((self.BATCH_SIZE, self.action_space))

        for i, j in enumerate(self.memory.get(self.BATCH_SIZE)):
            s, a, r, s1, terminal = j

            target = r

            if not terminal:
                tar_s1 = m2.predict(s1)
                target = r + self.GAMMA * np.amax(tar_s1[0])

            targets[i] = m2.predict(s)
            targets[i, a] = target
            inputs[i] = s

        history = m1.fit(inputs, targets, epochs=1, callbacks=[], verbose=0)
        self.episode_loss += history.history["loss"][0]

        self.episode_train_count += 1

        if self.ddqn and self.episode_train_count % 50 == 0:
            self.update_target_model()

    def loss(self):
        return 0 if self.episode_train_count == 0 else self.episode_loss / self.episode_train_count

    def act(self, state):
        self.epsilon = max(self.EPSILON_END, self.epsilon + self.EPSILON_DECAY)

        # Epsilon exploration
        if np.random.uniform() <= self.epsilon:
            return random.randrange(self.action_space)

        # Exploit Q-Knowledge
        if self.ddqn:
            act_values = self.target_model.predict(state)
        else:
            act_values = self.model.predict(state)

        return np.argmax(act_values[0])
