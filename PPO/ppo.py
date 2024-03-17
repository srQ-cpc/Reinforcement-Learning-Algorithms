import numpy
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam


class PPO:
    def __init__(
            self,
            state_space,
            action_space,
            learning_rate=0.0001,
            gamma=0.99,
            gae_lambda=0.95,
            clip=0.2,
            batch_size=32,
            update_steps=10
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.actor, self.critic = self._build_network()
        self.actor_old, _ = self._build_network()
        self.v_optimizer = Adam(learning_rate=self.learning_rate)
        self.p_optimizer = Adam(learning_rate=self.learning_rate)
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.state = []
        self.state_ = []
        self.action = []
        self.reward = []

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

    def _copy_weights(self):
        self.actor_old.set_weights(self.actor.get_weights())

    def choose_action(self, state):
        state = np.array([state])
        action_prob = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(np.arange(self.action_space), p=action_prob.ravel())
        return action

    def learn(self, state, action, reward, state_, done):
        self.state.append(state)
        self.state_.append(state_)
        self.action.append(action)
        self.reward.append(reward)
        if len(self.state) == self.batch_size or done:
            self._copy_weights()
            batch_size = len(self.state)
            v_pred = self.critic.predict(np.array(self.state), verbose=0).reshape(-1).tolist()
            if done:
                v_pred.append(0)
            else:
                v_pred.append(self.critic.predict(np.array([self.state_[-1]]), verbose=0)[0][0])
            adv = [0] * batch_size
            last_adv = 0
            for i in range(len(adv) - 1, -1, -1):
                delta = v_pred[i + 1] * self.gamma + self.reward[i] - v_pred[i]
                adv[i] = self.gamma * self.gae_lambda * last_adv + delta
            adv = numpy.array(adv)
            y_r = adv + np.array(v_pred[:-1])
            action_prob_old = self.actor_old.predict(np.array(self.state), verbose=0)[range(batch_size), self.action]
            for i in range(self.update_steps):
                with tf.GradientTape() as tape_p:
                    action_prob = self.actor(np.array(self.state), training=True)
                    indices = np.concatenate(
                        (
                            np.arange(batch_size).reshape([-1, 1]),
                            np.array(self.action).reshape([-1, 1])
                        ),
                        axis=1
                    )
                    action_prob = tf.gather_nd(action_prob, indices)
                    ratio = action_prob / (action_prob_old + 1e-5)
                    p_loss = -tf.reduce_mean(
                        tf.minimum(
                            ratio * adv,
                            tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip) * adv
                        )
                    )
                p_grads = tape_p.gradient(p_loss, self.actor.trainable_weights)
                self.p_optimizer.apply_gradients(zip(p_grads, self.actor.trainable_weights))
                with tf.GradientTape() as tape_v:
                    v = self.critic(np.array(self.state), training=True)
                    v_loss = tf.reduce_mean(tf.losses.MeanSquaredError()(v, y_r.reshape(-1,1)))
                v_grads = tape_v.gradient(v_loss, self.critic.trainable_weights)
                self.v_optimizer.apply_gradients(zip(v_grads, self.critic.trainable_weights))
            self.state.clear()
            self.state_.clear()
            self.action.clear()
            self.reward.clear()
