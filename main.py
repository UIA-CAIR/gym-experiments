import numpy
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dqn.agent import Agent
from dqn.models import cnn, cnn_rnn, capsnet
from environments import DeepLineWarsEnvironment, DeepRTSEnvironment, FlappyBirdEnvironment


class Statistics:

    def __init__(self, action_size, action_names):
        self.action_size = action_size
        self.action_names = action_names
        self.loss = {"x": [], "y": []}
        self.win_percent = {"x": [], "y": []}
        self.action_distribution = {"x": [], "y": []}

        self.episodic_action_distribution = numpy.zeros(action_size, dtype=numpy.int16)

        self.figure = Figure()
        canvas = FigureCanvasAgg(self.figure)

        self.plot_loss = self.figure.add_subplot(2, 2, 1)
        self.plot_win_percent = self.figure.add_subplot(2, 2, 2)
        self.plot_action_distribution = self.figure.add_subplot(2, 2, 3)

        self.win_percent_label = "Win Percent"

    def add_loss(self, epoch, loss):
        self.loss["x"].append(epoch)
        self.loss["y"].append(loss)

    def add_win_percent(self, win_percent, y_axis_name=None):
        if y_axis_name is not None:
            self.win_percent_label = y_axis_name

        self.win_percent["x"].append(len(self.win_percent["x"]) + 1)
        self.win_percent["y"].append(win_percent)

    def add_action(self, a):
        self.episodic_action_distribution[a] = self.episodic_action_distribution[a] + 1

    def next_episode(self, episode):

        population = numpy.sum(self.episodic_action_distribution)
        items = [(x / population) * 100 for x in self.episodic_action_distribution]

        self.action_distribution["y"].append(items)
        self.action_distribution["x"].append(episode)

        self.episodic_action_distribution = numpy.zeros(self.action_size, dtype=numpy.int16)

    def plot(self):
        # Action Distributio

        self.plot_loss.cla()
        self.plot_win_percent.cla()
        self.plot_action_distribution.cla()

        self.plot_loss.grid(True)
        self.plot_loss.set_xlabel("Episode")
        self.plot_loss.set_ylabel("Loss")

        self.plot_win_percent.grid(True)
        self.plot_win_percent.set_xlabel("Episode")
        self.plot_win_percent.set_ylabel(self.win_percent_label)

        self.plot_action_distribution.grid(True)
        self.plot_action_distribution.set_xlabel("Episode")
        self.plot_action_distribution.set_ylabel("Action Frequency")

        self.plot_loss.plot(self.loss["x"], self.loss["y"])
        self.plot_win_percent.plot(self.win_percent["x"], self.win_percent["y"])

        lineObjects = self.plot_action_distribution.plot(
            numpy.array(self.action_distribution["x"]),
            numpy.array(self.action_distribution["y"])
        )
        self.plot_action_distribution.legend(
            lineObjects,
            self.action_names,
            ncol=2,
            loc=9,
            bbox_to_anchor=(1.6, 1.03),
            fontsize=6
        )

        self.figure.tight_layout()

        self.figure.savefig("output.png")
        self.figure.savefig("output.eps")


if __name__ == "__main__":
    train_epochs = 20
    train_interval = 2
    victories = 0
    environments = [
        #(Environment,              e_start e_end   e_steps exp_ep, episodes,   episode_max_length)
        (FlappyBirdEnvironment,     0.1,    0.0001, 1000,   200,    1000,       6000),
        (DeepRTSEnvironment,        1.0,    0.0001, 1000,   10,     100,        6000),
        (DeepLineWarsEnvironment,   1.0,    0.0001, 1000,   10,     100,        6000)
    ]

    for env_signature, e_start, e_end, e_steps, exploration_episodes, episodes, episode_max_length in environments:
        env = env_signature()
        statistics = Statistics(env.action_space, env.action_space_names)
        agent = Agent(
            env.observation_space,
            env.action_space,
            cnn,
            exploration_episodes=exploration_episodes,
            e_steps=e_steps,
            e_start=e_start,
            e_end=e_end
        )

        for episode in range(episodes):
            s = env.reset()
            t = False
            step = 0
            action_distribution = []

            while not t and step < episode_max_length:
                a = agent.act(s)
                statistics.add_action(a)

                s1, r, t, _ = env.step(a)
                r = env.get_reward()

                agent.memory.add((s, a, r, s1, t))

                s = s1
                step += 1

            victory = env.victory()
            if victory is bool:
                victories += 1 if victory else 0
                win_percent = (victories / agent.episode) * 100
                statistics.add_win_percent(win_percent)
            else:
                # In those cases where there are no "victory" (Flappy Bird etc), Use victory as total score
                victories = env.victory()
                statistics.add_win_percent(victories, y_axis_name="Total Reward")
                win_percent = "N/A"

            loss = None
            if agent.episode > agent.exploration_episodes and agent.episode % train_interval == 0:
                for e in range(train_epochs):
                    agent.train()
                    loss = agent.loss
                    statistics.add_loss(agent.epoch, loss)


            episode = agent.episode


            statistics.next_episode(episode)
            statistics.plot()

            print("Episode: %s, Win Ratio: %s, Loss: %s, Memory: %s, Epsilon: %s" %
                  (episode, win_percent, loss, agent.memory.count, agent.epsilon)
                  )
            agent.next_episode()



