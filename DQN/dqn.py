import random
import math
import numpy as np
import tensorflow as tf

from collections import deque

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam


class DQN:
    def __init__(
            self,
            state_space,
            action_space,
            learning_rate=0.0001,
            gamma=0.99,
            epsilon_start=0.95,
            epsilon_end=0.01,
            epsilon_decay=500,
            replace_target_iter=200,
            learning_interval=1,
            replay_buffer_size=10000,
            batch_size=32
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.replace_target_iter = replace_target_iter
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.q_network = self.build_network()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.q_target_network = self.build_network()
        self._copy_weights()
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.learning_steps = 0
        self.learning_interval = learning_interval
        self.add_expr_steps = 0
        self.predict_count = 0

    def _copy_weights(self):
        self.q_target_network.set_weights(self.q_network.get_weights())

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        if (self.add_expr_steps % self.learning_interval) != 0:
            return
        self.learning_steps += 1
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, state_, done = map(np.stack, zip(*minibatch))
        q_ = self.q_target_network.predict(state_, verbose=0)
        with tf.GradientTape() as tape:
            q = self.q_network(state, training=True)
            target = q.numpy().copy()
            target[range(self.batch_size), action] = (1 - done) * tf.reduce_max(q_, axis=1) * self.gamma + reward
            loss = tf.losses.MeanSquaredError()(target, q)
        grads = tape.gradient(loss, self.q_network.trainable_weights)
        # print(f"loss: {loss}")
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_weights))

        if self.learning_steps % self.replace_target_iter == 0:
            self._copy_weights()

    def predict(self, state, mode="train"):
        self.predict_count += 1
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.predict_count / self.epsilon_decay)
        # print(f"epsilon: {epsilon}")
        if mode == "train" and np.random.uniform() < epsilon:
            return np.random.randint(0, self.action_space)
        state = np.array([state])
        q = self.q_network.predict(state, verbose=0)
        return np.argmax(q[0])

    def build_network(self):
        input = Input(shape=self.state_space)
        x = Flatten()(input)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(self.action_space, activation='linear')(x)
        model = Model(inputs=input, outputs=output)
        return model

    def add_expr(self, state, action, reward, state_, done):
        self.replay_buffer.append((state, action, reward, state_, done))
        self.add_expr_steps += 1
