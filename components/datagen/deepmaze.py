import random
import gym
import gym_maze
import numpy
import time
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

save_path = os.path.join(dir_path, "..", "training_data")
os.makedirs(save_path, exist_ok=True)

env = gym.make('maze-img-15x15-full-deterministic-v0')

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

        print(s1.shape)

        memories.append((s, a, r, s1, terminal))

        s = s1
        tick += 1

        if tick > 6000:
            memories.clear()
            terminal = True

    print("Episode: %s, Epsilon: %s, Buffer: %s, Loss: %s" % (
        episode,
        0,
        len(memories),
        0,
    ))

    if len(memories) > 0:
        items = numpy.array(memories)
        numpy.save(os.path.join(save_path, "deepmaze_%s.npy" % episode), items)
        memories.clear()
