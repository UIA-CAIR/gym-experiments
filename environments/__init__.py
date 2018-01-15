
import random
import numpy as np

from DeepRTS.pyDeepRTS import DeepRTS
from DeepRTS import ensure_data_directory
from scipy.misc import imresize

from environments.DeepLineWars.Game import Game
from environments.DeepLineWars.algorithms.random import Random
from environments.FlappyBird.wrapped_flappy_bird import FlappyBird
from util.imaging import rgb2gray


class Environment:

    def __init__(self):
        self.state = None

    def step(self, action):
        pass

    def reset(self):
        pass

    def _get_state(self):
        raise NotImplementedError("Implement _get_state function!")

    @property
    def observation_space(self):
        raise NotImplementedError("Implement observation_space property!")

    @property
    def action_space(self):
        raise NotImplementedError("Implement action_space property!")

    @property
    def action_space_names(self):
        raise NotImplementedError("Implement action_space_names property!")


class FlappyBirdEnvironment(Environment):

    def __init__(self):
        super().__init__()

        self.env = FlappyBird()
        self.action_names = ["No Action", "Flap"]

        self._reward = None
        self._cum_reward = 0

    def step(self, action):
        super().step(action)

        s1, r, t, = self.env.frame_step(action)
        s1 = rgb2gray(s1)
        s1 = imresize(s1, (84, 84))
        s1 = np.reshape(s1, (1, 84, 84, 1))
        self._reward = r
        self._cum_reward += r
        return s1, r, t, {}

    def victory(self):
        return np.log(self._cum_reward)

    def get_reward(self):
        return self._reward

    def reset(self):
        self._reward = None
        #self._cum_reward = 0
        return self._get_state()

    def _get_state(self):
        s = rgb2gray(self.env.get_state())
        s = imresize(s, (84, 84))
        s = np.reshape(s, (1, 84, 84, 1))
        return s

    @property
    def observation_space(self):
        return self._get_state().shape[1:]

    @property
    def action_space(self):
        return len(self.action_names)

    @property
    def action_space_names(self):
        return self.action_names


class DeepRTSEnvironment(Environment):

    def __init__(self):
        super().__init__()

        ensure_data_directory()
        self.env = DeepRTS(2, True)
        self.player = self.env.addPlayer()
        self.opponent = self.env.addPlayer()
        self.action_names = [
            "Prev Unit",
            "Next Unit",
            "Move Left",
            "Move Right",
            "Move Down",
            "Move UpLeft",
            "Move UpRight",
            "Move DownLeft",
            "Move DownRight",
            "Attack",
            "Harvest",
            "Build 0",
            "Build 1",
            "Build 2",
            "No Action"
        ]
        self.env.setFPS(60)
        self.env.setUPS(10000000)
        self.env.setAPM(10000)
        self.update_times = 5

        self.env.initGUI()
        self.env.start()

    def step(self, action):
        super().step(action)

        self.opponent.queueAction(random.randint(0, self.action_space))
        self.player.queueAction(action)

        for _ in range(self.update_times):
            self.env.tick()
            self.env.update()
        self.env.render()
        self.env.caption()

        s1 = self._get_state()
        t = self.env.checkTerminal()
        r = self.get_reward()

        return s1, r, t, {}

    def victory(self):
        p_score = self.player.getScore()
        opp_score = self.opponent.getScore()
        return True if p_score > opp_score else False

    def get_reward(self):
        p_score = self.player.getScore()
        opp_score = self.opponent.getScore()
        return (p_score - opp_score) / (p_score + opp_score)

    def reset(self):
        super().reset()
        self.env.reset()
        self.env.render()
        return self._get_state()

    def _get_state(self):
        s = self.env.getState()
        s = np.reshape(s, (1, 30, 30, 3))
        return s

    @property
    def observation_space(self):
        return self._get_state().shape[1:]

    @property
    def action_space(self):
        return len(self.action_names)

    @property
    def action_space_names(self):
        return self.action_names


class DeepLineWarsEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self.env = Game()
        self.env.representation = "image_grayscale"
        self.player = self.env.players[0]
        self.opponent = self.env.players[1]
        self.opponent.agents.append(Random(self.env, self.opponent))

    def step(self, action):
        super().step(action)

        s1, r, t, _ = self.env.step(self.player, action)
        s1 = np.reshape(s1, (1, 84, 84, 1))
        self.env.update()
        self.env.render()
        return s1, r, t, _

    def victory(self):
        return self.env.winner == self.player

    def get_reward(self):
        return (self.player.health - self.opponent.health) / 50

    def reset(self):
        super().reset()
        self.env.reset()
        self.env.render()
        return self._get_state()

    def _get_state(self):
        s = self.env.get_state(self.player)
        s = np.reshape(s, (1, 84, 84, 1))
        return s

    @property
    def observation_space(self):
        return self._get_state().shape[1:]

    @property
    def action_space(self):
        return len(self.player.action_space)

    @property
    def action_space_names(self):
        return [x["action"] for x in self.player.action_space]
