import numpy as np
from keras.callbacks import TensorBoard


class DQN:
    def __init__(self, state_size, action_size, batch_size=32, lr=1e-6, dr=0.99):
        self.lr = lr
        self.dr = dr
        self.memory = None
        self.s_space = state_size
        self.a_space = action_size
        self.b_size = batch_size
        self.model = None

        self.accumulative_loss = 0
        self.loss_counter = 0

        self.tbcb = None
        self.losses = []

    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def train(self):
        inputs = np.zeros(((self.b_size,) + self.s_space))
        targets = np.zeros((self.b_size, self.a_space))

        for i, j in enumerate(np.random.choice(len(self.memory), self.b_size, replace=False)):
            state, action, reward, next_state, terminal = self.memory[j]

            target = reward

            if not terminal:
                target = reward + self.dr * np.amax(self.model.predict(next_state)[0])

            targets[i] = self.model.predict(state)
            targets[i, action] = target
            inputs[i] = state

        history = self.model.fit(inputs, targets, epochs=1, verbose=0, callbacks=[])
        self.losses.append(history.history["loss"][0])
