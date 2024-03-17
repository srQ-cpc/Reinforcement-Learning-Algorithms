import random
import numpy as np
import tensorflow as tf

from collections import deque

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, Concatenate
from keras.optimizers import Adam


class DDPG:
    def __init__(
            self,
            state_space,
            action_space,
            action_bound,
            learning_rate=0.0001,
            gamma=0.99,
            tau=0.01,
            var=1,
            learning_interval=1,
            replay_buffer_size=1024,
            batch_size=32
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.a_optimizer = Adam(learning_rate=self.learning_rate)
        self.c_optimizer = Adam(learning_rate=self.learning_rate)
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.learning_steps = 0
        self.learning_interval = learning_interval
        self.add_expr_steps = 0
        self.predict_count = 0
        self.actor = self._build_actor_network()
        self.actor_target = self._build_actor_network()
        self.critic = self._build_critic_network()
        self.critic_target = self._build_critic_network()
        self.var = var
        self._update_target_network()

    def _update_target_network(self, tau=1.0):
        w_a, w_c = self.actor.get_weights(), self.critic.get_weights()
        w_a_, w_c_ = self.actor_target.get_weights(), self.critic_target.get_weights()
        for w, w_ in zip(w_a, w_a_):
            np.copyto(w_, (1 - tau) * w_ + tau * w)
        for w, w_ in zip(w_c, w_c_):
            np.copyto(w_, (1 - tau) * w_ + tau * w)
        self.actor_target.set_weights(w_a_)
        self.critic_target.set_weights(w_c_)

    def _build_actor_network(self):
        input = Input(shape=self.state_space)
        x = Flatten()(input)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(self.action_space[0], activation='tanh')(x)
        output = Lambda(lambda a: self.action_bound * a)(x)
        return Model(inputs=input, outputs=output)

    def _build_critic_network(self):
        s_input = Flatten()(Input(shape=self.state_space))
        a_input = Flatten()(Input(shape=self.action_space))
        x = Concatenate(axis=1)([s_input, a_input])
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(1, activation='linear')(x)
        return Model(inputs=[s_input, a_input], outputs=output)

    def choose_action(self, state, mode="train"):
        state = np.array([state])
        action = self.actor.predict(state, verbose=0)[0]
        if mode == "train":
            action = np.clip(np.random.normal(action, self.var), -self.action_bound, self.action_bound)
            if self.add_expr_steps > self.replay_buffer_size:
                self.var *= 0.995
        return action

    def add_expr(self, state, action, reward, state_, done):
        self.replay_buffer.append((state, action, reward, state_, done))
        self.add_expr_steps += 1

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        if (self.add_expr_steps % self.learning_interval) != 0:
            return
        self.learning_steps += 1
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, state_, done = map(np.stack, zip(*minibatch))
        action_ = self.actor_target.predict(state_, verbose=0)
        q_ = self.critic_target.predict([state_, action_], verbose=0)
        q_target = reward + (1 - done) * self.gamma * q_
        with tf.GradientTape() as tape:
            q_pred = self.critic([state, action], training=True)
            loss = tf.losses.MeanSquaredError()(q_pred, q_target)
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.c_optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        with tf.GradientTape() as tape:
            action = self.actor(state, training=True)
            q = self.critic([tf.convert_to_tensor(state), action], training=True)
            loss = -tf.reduce_mean(q)
        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.a_optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))

        self._update_target_network(self.tau)





