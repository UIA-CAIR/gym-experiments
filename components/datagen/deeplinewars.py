import random
import gym
import gym_deeplinewars.envs
import numpy
import time
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

save_path = os.path.join(dir_path, "..", "training_data")
os.makedirs(save_path)

env = gym.make('deeplinewars-stochastic-11x11-v0')
env.set_representation("image_grayscale")

action_space = env.action_space
memories = []
episodes = 1000


for episode in range(episodes):
    s = env.reset()

    terminal = False
    episode_reward = 0
    tick = 0

    while not terminal:
        # Draw environment on screen
        env.render()  # For image you MUST call this

        # Draw action from distribution
        a = random.randint(0, action_space-1)
        s1, r, terminal, _ = env.step(a)

        memories.append((s, a, r, s1, terminal))

        s = s1
        tick += 1

        if tick > 6000:
            memories.clear()
            terminal = True

    print("Episode: %s, Epsilon: %s, Buffer: %s, Loss: %s, p:%s, e:%s" % (
        episode,
        0,
        len(memories),
        0,
        env.player.health, env.player.opponent.health
    ))

    if len(memories) > 0:
        items = numpy.array(memories)
        numpy.save(os.path.join(save_path, "deeplinewars_%s.npy" % episode), items)
        memories.clear()
