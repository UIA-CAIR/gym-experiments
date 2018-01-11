import random
from FlashRL.lib.Game import Game
import numpy
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

save_path = os.path.join(dir_path, "..", "training_data")
os.makedirs(save_path, exist_ok=True)


#memories = []
#episodes = 1000
#items = numpy.array(memories)
#

memories = []
s = None
a = None

def on_frame(state, type, vnc):
    global s
    global a

    s1 = numpy.reshape(state, (84, 84, 1))
    if s is None:
        s = s1

    a = random.choice(["w", "a", "s", "d"])
    vnc.send_key(a)

    memories.append((s, a, 0, s1, False))
    print(len(memories) / 5000)

    s = s1

    if len(memories) > 5000:
        numpy.save(os.path.join(save_path, "flashrl_%s.npy" % 1), numpy.array(memories))

g = Game("multitask", fps=60, frame_callback=on_frame, grayscale=True, normalized=True)

