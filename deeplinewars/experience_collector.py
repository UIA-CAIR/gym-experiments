import random
import gym
import gym_deeplinewars.envs
import numpy
import time

env = gym.make('deeplinewars-shuffle-v0')
env.set_representation("raw_enemy")
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
        #env.render()  # For image you MUST call this

        # Draw action from distribution
        a = random.randint(0, action_space-1)
        s1, r, terminal, _ = env.step(a)

        memories.append((s, a, r, s1, terminal))

        s = s1
        tick += 1

    if env.env.winner == env.player:
        print()

    print("Episode: %s, Epsilon: %s, Buffer: %s, Loss: %s, p:%s, e:%s" % (
        episode,
        0,
        len(memories),
        0,
        env.player.health, env.player.opponent.health
    ))

    items = numpy.array(memories)
    numpy.save("./training_data/match_%s_%s.npy" % (episode, time.time()), items)
    memories.clear()
