import os
from collections import deque
import pickle
from threading import Thread
import numpy as np
from keras.callbacks import TensorBoard
from maze.agent import DQN
import gym
import time
import gym_maze
from tqdm import tqdm
import importlib.util
import keras.backend as K
from maze import util


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, "models")
models = [os.path.join(model_path, x) for x in os.listdir(model_path) if x not in ["__init__.py", "__pycache__"]]
env_list = [
    "maze-arr-9x9-full-deterministic-v0",
    "maze-arr-11x11-full-deterministic-v0",
    "maze-arr-13x13-full-deterministic-v0",
    "maze-arr-15x15-full-deterministic-v0",
    "maze-arr-17x17-full-deterministic-v0",
    "maze-arr-19x19-full-deterministic-v0",
    "maze-arr-25x25-full-deterministic-v0",
    "maze-arr-35x35-full-deterministic-v0",
    # "maze-arr-55x55-full-deterministic-v0"
]

epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000]  # epochs
# From 1 second to 12 hours

training_sets = [
    "1m-training-set"
]


class Experiment(Thread):

    def __init__(self, model_path, env_name, prev_epoch, epoch, training_set):
        super().__init__()

        self.model_path = model_path
        self.model_name = self.model_path.split("/")[-1].replace(".py", "")
        self.skip = False
        self.weight_save_path = os.path.join(dir_path, "weights", "%s_%s_%s_%s.h5" % (env_name, training_set, self.model_name, epoch))
        if os.path.exists(self.weight_save_path):
            print("%s:%s already exists! Skipping..." % (epoch, self.model_name))
            self.skip = True
            return

        self.env_name = env_name
        self.training_set_name = training_set
        self.training_data_path = os.path.join(dir_path, training_set, "training-data_%s.pkl" % self.env_name)
        self.epoch = epoch
        self.prev_epoch = prev_epoch
        self.init_weights = None

        if self.prev_epoch is not None:
            self.init_weights = os.path.join(
                dir_path,
                "weights",
                "%s_%s_%s_%s.h5" % (env_name, training_set, self.model_name, prev_epoch)
            )
        else:
            self.prev_epoch = 0

        spec = importlib.util.spec_from_file_location("models", self.model_path)
        models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models)
        self.model_fn = models.model

    def load_memory(self, agent):
        with open(self.training_data_path, "rb") as file:
            mem = pickle.load(file)
            agent.memory = deque(maxlen=len(mem))
            for item in tqdm(mem):
                s, a, r, s1, t = item
                fixed_item = (
                    np.expand_dims(s, axis=0),
                    a,
                    r,
                    np.expand_dims(s1, axis=0),
                    t
                )
                agent.memory.append(fixed_item)




    def run(self):
        if self.skip:
            return

        env = gym.make(self.env_name)
        env.reset()
        K.clear_session()
        agent = DQN(env.observation_space, env.action_space)
        agent.model, model_name = self.model_fn(env.observation_space, agent.a_space, agent.lr)

        if self.init_weights:
            print("Weights: %s" % self.init_weights)
            agent.model.load_weights(self.init_weights)
        agent.tbcb = TensorBoard(
            log_dir= os.path.join("tensorboard", "%s_%s_%s_%s" % (self.env_name, self.training_set_name, model_name, self.epoch)),
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )

        self.load_memory(agent)

        start_epoch = self.prev_epoch
        end_epoch = self.epoch
        total_epoch = end_epoch - start_epoch

        for epoch in tqdm(range(start_epoch, end_epoch), total=total_epoch, desc="%s:%s" % (end_epoch, self.env_name)):
            agent.train()

        agent.model.save_weights(self.weight_save_path)





for env_name in env_list:

    for model_path in models:

        prev_epoch = None
        for epoch in epochs:

            for training_set in training_sets:
                print(env_name, epoch)
                experiment = Experiment(model_path, env_name, prev_epoch, epoch, training_set)
                experiment.start()
                experiment.join()

            prev_epoch = epoch







