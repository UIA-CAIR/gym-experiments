import importlib
import gym
import gym_maze # This is required in order to load gym-maze
import os
import numpy as np
import pickle
from tqdm import tqdm

from maze.agent import DQN

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, "models")
weights_path = os.path.join(dir_path, "weights")
weights_list = [(x, os.path.join(weights_path, x)) for x in sorted(os.listdir(weights_path)) if ".h5" in x]
result_path = os.path.join(dir_path, "results")

os.makedirs(result_path, exist_ok=True)

def load_model_fn(model_fn_path):
    spec = importlib.util.spec_from_file_location("models", model_fn_path)
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    return models.model


if __name__ == '__main__':



    for weights_path in weights_list:
        env_name, training_set, model_name, duration = weights_path[0].split("_")
        duration = int(duration.replace(".h5", ""))
        model_fn_path = os.path.join(model_path, model_name + ".py")
        model_fn = load_model_fn(model_fn_path)
        data = []

        filename = os.path.join(result_path, "results_%s_%s_%s_%s.pkl" % (env_name, training_set, model_name, duration))

        if os.path.isfile(filename):
            continue

        # Reset environment
        env = gym.make(env_name)
        env_timeout = 1000
        steps = 0
        num_games = 1

        # Load agent
        agent = DQN(env.observation_space, env.action_space)
        agent.model, model_name = model_fn(env.observation_space, agent.a_space, agent.lr)
        agent.model.load_weights(weights_path[1])


        for i in tqdm(range(num_games), desc="%s:%s" % (duration, env_name)):

            # Reset game
            s = env.reset()
            s = np.expand_dims(s, axis=0)

            # Set terminal state to false
            terminal = False
            steps = 0

            while not terminal:
                # Draw environment on screen
                #env.render()  # For image you MUST call this

                # Draw action from distribution
                a = agent.act(s)

                # Perform action in environment
                s1, r, t, info = env.step(a)
                s1 = np.expand_dims(s1, axis=0)
                terminal = t

                if steps > env_timeout:
                    terminal = True

                steps += 1

                s = s1

            data.append({"game": i, "steps": steps, "optimal": info["optimal_path"], "train_duration": duration})


        with open(filename, "wb") as file:
            pickle.dump(data, file)