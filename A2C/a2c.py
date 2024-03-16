import random
import math
import numpy as np
import tensorflow as tf

from collections import deque

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam


class A2C:
    def __init__(
            self,
            state_space,
            action_space,
            learning_rate=0.00005,
            gamma=0.99
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.actor, self.critic = self._build_network()
        self.v_optimizer = Adam(learning_rate=self.learning_rate)
        self.p_optimizer = Adam(learning_rate=self.learning_rate)

    def _build_network(self):
        input = Input(shape=self.state_space)
        x = Flatten()(input)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        action_prob = Dense(self.action_space, activation='softmax')(x)
        v = Dense(1, activation='linear')(x)
        actor = Model(inputs=input, outputs=action_prob)
        critic = Model(inputs=input, outputs=v)
        return actor, critic

    def learn(self, state, action, reward, state_, done):
        state = np.array([state])
        state_ = np.array([state_])
        v_ = self.critic.predict(state_, verbose=0)[0]
        with tf.GradientTape() as tape_v:
            v_pred = self.critic(state, training=True)[0]
            adv = reward + self.gamma * (1 - done) * v_ - v_pred
            v_loss = adv * adv
        v_grads = tape_v.gradient(v_loss, self.critic.trainable_weights)
        with tf.GradientTape() as tape_p:
            adv = adv.numpy()
            action_prob = self.actor(state, training=True)[0]
            p_loss = -tf.math.log(action_prob[action]) * adv
        p_grads = tape_p.gradient(p_loss, self.actor.trainable_weights)
        self.v_optimizer.apply_gradients(zip(v_grads, self.critic.trainable_weights))
        self.p_optimizer.apply_gradients(zip(p_grads, self.actor.trainable_weights))
        # print(f"p_loss: {p_loss}, v_loss: {v_loss}" )

    def choose_action(self, state):
        state = np.array([state])
        action_prob = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(np.arange(self.action_space), p=action_prob.ravel())
        return action
