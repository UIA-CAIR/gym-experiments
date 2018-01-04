import json
import pickle

import numpy as np
import requests
from tqdm import tqdm

with open("./10m-training-set/training_data_maze-arr-35x35-full-deterministic-v0.pkl", "rb") as file:
    training_set = pickle.load(file)

for item in tqdm(training_set[3500:]):
    s, a, r, s1, t = item
    s = np.expand_dims(s, axis=0)
    s1 = np.expand_dims(s1, axis=0)

    data = {
        "model_name": "maze_35x35",
        "data": {
            "s": pickle.dumps(s),
            "a": a,
            "r": r,
            "s1": pickle.dumps(s1),
            "t": t,
            "info": {},
            "random": np.random.uniform()
        }
    }

    x = requests.post("http://thor.uia.no:3000/insert", data=pickle.dumps(data))
