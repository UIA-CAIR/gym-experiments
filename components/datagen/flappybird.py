import random

import numpy
from ple.games.flappybird import FlappyBird
from ple import PLE
import os

from scipy.misc import imresize

dir_path = os.path.dirname(os.path.realpath(__file__))

save_path = os.path.join(dir_path, "..", "training_data")
os.makedirs(save_path, exist_ok=True)

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)

p.init()
memores = []
reward = 0.0

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])

for i in range(40000):
    if p.game_over():
        p.reset_game()

    s = p.getScreenRGB()
    action = random.choice(p.getActionSet())
    reward = p.act(action)
    s1 = p.getScreenRGB()

    s = rgb2gray(s)
    s1 = rgb2gray(s1)
    s = imresize(s, (84, 84))
    s1 = imresize(s1, (84, 84))

    memores.append((s, action, reward, s1, False))

numpy.save(os.path.join(save_path, "flappybird_%s.npy" % 1), numpy.array(memores))