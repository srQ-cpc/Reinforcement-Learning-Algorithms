import numpy as np


class Sarsa:
    def __init__(self, action_space_n, learning_rate=0.01, gamma=0.9, e_greedy=0.9):
        self.action_space_n = action_space_n
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.q_table = {}

    def predict(self, state):
        state = self._check_state(state)
        prob = np.random.uniform()
        if prob < self.e_greedy:
            max_indices = np.where(self.q_table[state] == np.max(self.q_table[state]))[0]
            action = np.random.choice(max_indices)
        else:
            action = np.random.randint(low=0, high=self.action_space_n)
        return action

    def fit(self, state, action, reward, state_, action_):
        state = self._check_state(state)
        state_ = self._check_state(state_)
        q = self.q_table[state][action]
        if state_ == "terminal":
            q_ = 0
        else:
            q_ = self.q_table[state_][action_]
        self.q_table[state][action] += (reward + self.gamma * q_ - q) * self.learning_rate

    def _check_state(self, state):
        state = str(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_n)
        return state
