import gym
import gym_deeplinewars.envs
from environments.deeplinewars.rl.Agent import Agent
from environments.deeplinewars.rl.models import cnn1, ddqn, capsule1, cnnrnn
import matplotlib.pyplot as plt
import numpy as np

plot_loss = []
plot_win_percent = []
plot_wins = []
plot_losses = []
plot_x_axis = []
fig, ax = plt.subplots(nrows=2, ncols=2)


env = gym.make('deeplinewars-stochastic-11x11-v0')
env.set_representation("raw_enemy")

model = cnn1.model(env.observation_space[1:], env.action_space, 1e-4)
agent = Agent(
    env.observation_space,
    env.action_space,
    lr=1e-4,
    memory_size=100000,
    e_start=1.0,
    e_end=0.0,
    e_steps=1000000,
    batch_size=16,
    discount=0.99,
    model=model
)

#for file in tqdm(glob.glob("./training_data/*.npy")):
#    experiences = np.load(file)
#    for experience in experiences:
#        agent.memory.add(experience)


episodes = 10000000

victories = 0
losses = 0
for episode in range(episodes):

    s = env.reset()
    agent.reset()

    terminal = False
    episode_reward = 0
    tick = 0

    while not terminal:
        # Draw environment on screen
        env.render()  # For image you MUST call this

        # Draw action from distribution
        a = agent.act(s)

        # Perform action in environment
        s1, r, terminal, _ = env.step(a)

        # Add to replay buffer
        agent.memory.add((s, a, r, s1, terminal))
        s = s1
        tick += 1



    if r <= 0:
        losses += 1
    else:
        victories += 1

    for x in range(10):
        agent.train()

    plot_loss.append(agent.loss())
    plot_win_percent.append(victories / (victories + losses))
    plot_wins.append(victories)
    plot_losses.append(losses)
    plot_x_axis.append(episode)

    plt.subplot(2, 2, 1)
    plt.plot(plot_x_axis, plot_loss)
    plt.subplot(2, 2, 2)
    plt.plot(plot_x_axis, plot_win_percent)
    plt.subplot(2, 2, 3)
    plt.plot(plot_x_axis, plot_wins)
    plt.plot(plot_x_axis, plot_losses)
    plt.savefig("summary.png")
    plt.close()

    print("Episode: %s, Epsilon: %s, Buffer: %s, Loss: %s, Win/Loss: %s/%s" % (
        episode,
        agent.epsilon,
        agent.memory.count,
        agent.loss(),
        victories,
        losses
    ))

