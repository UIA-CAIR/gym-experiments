import os
import glob
import numpy as np
from tqdm import tqdm
dir_path = os.path.dirname(os.path.realpath(__file__))


def load_all_memories(limit=None):
    memories = []
    files = glob.glob(os.path.join(dir_path, "training_data/*.npy"))
    if limit is not None:
        files = files[:limit]
    for file in tqdm(files, desc="Loading experience buffer"):
        with open(file, "rb") as f:
            data = np.load(f)
            memories.extend(data)

    return memories