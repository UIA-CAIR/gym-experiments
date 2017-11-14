import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
# Example data
names = []
performance = []
colors = []
data = {}


result_path = os.path.dirname(os.path.realpath(__file__))
results = [os.path.join(result_path, x) for x in os.listdir(result_path) if ".pkl" in x]
pkl_results = [(x, pickle.load(open(x, "rb"))) for x in results]

_epochs = [
    #1,
    #10,
    #20,
    #30,
    #40,
    #50,
    #60,
    #70,
    #80,
    #90,
    100,
    #120,
    #140,
    #160,
    #180,
    #200,
    #300,
    #400,
    #500,
    #600,
    #700,
    #800,
    #900,
    1000,
    #2000,
    #3000,
    #4000,
    #5000,
    #6000,
    #7000,
    #8000,
    #9000,
    10000,
    #20000,
    30000
]  # epochs


for result in pkl_results:

    steps_sum =[]
    name = result[0]
    cnt = 0
    optimal = result[1][0]["optimal"]
    for item in result[1]:
        #print(item)
        steps_sum.append(item["steps"])

        cnt += 1

    perf = sum(steps_sum) / cnt
    if perf > 1000:
        perf = 100

    results, game_type, training_set, model, epochs = os.path.basename(name).replace(".pkl", "").split("_")
    game, representation, size, full, randomness, _ = game_type.split("-")

    name = "%s:%s" % (size, epochs)

    if size not in data:
        data[size] = []

    data[size].append((name, perf, int(epochs), model))

#
data_items = sorted(data.items())

for model_type in ["cnn1", "capsule1"]:

    for key, items in data_items:

        color = np.random.rand(3, )
        for item in sorted(items, key=lambda x: (x[2])):

            train_duration = item[2]
            if train_duration not in _epochs:
                continue

            if model_type != item[3]:
                continue

            names.append(item[0])
            performance.append(item[1])
            colors.append(color)

    # Example data
    y_pos = np.arange(len(names))
    #error = np.random.rand(len(names)) xerr=error

    ax.barh(y_pos, performance,  align='center',
            color=colors, ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Steps (Lower is better)')
    ax.set_title('Maze: DQN %s Performance' % model_type)

    plt.savefig("%s_performance.png" % model_type)

    names.clear()
    performance.clear()
    colors.clear()