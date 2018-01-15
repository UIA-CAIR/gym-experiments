import random
from collections import deque

import gym
import gym_maze
import pickle

envs = [
    "maze-arr-9x9-full-deterministic-v0",
    "maze-arr-11x11-full-deterministic-v0",
    "maze-arr-13x13-full-deterministic-v0",
    "maze-arr-15x15-full-deterministic-v0",
    "maze-arr-17x17-full-deterministic-v0",
    "maze-arr-19x19-full-deterministic-v0",
    "maze-arr-25x25-full-deterministic-v0",
    "maze-arr-35x35-full-deterministic-v0",
    #"maze-arr-55x55-full-deterministic-v0"
]

dataset_size = 10000000

for env_name in envs:
    env = gym.make(env_name)
    memory = deque(maxlen=dataset_size)

    while len(memory) < dataset_size:

        s = env.reset()
        steps = 0
        temp_memory = []
        terminal = False

        while not terminal:
            # Draw environment on screen
            #env.render()  # For image you MUST call this

            # Draw action from distribution
            a = random.randint(0, 3)

            # Perform action in environment
            s1, r, t, _ = env.step(a)
            terminal = t

            memory.append((s, a, r, s1, terminal))

            s = s1

            if steps > 1000:
                temp_memory.clear()
                steps = 0

        memory.extend(temp_memory)

        print("env: %s, memory: %s/%s" % (env_name, len(memory), dataset_size))

    with open("training-data_%s.pkl" % env_name, "wb") as file:
        pickle.dump(list(memory), file)
