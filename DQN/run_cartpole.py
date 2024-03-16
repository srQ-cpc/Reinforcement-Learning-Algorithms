import gym
from dqn import DQN


def main():
    env = gym.make("CartPole-v1")
    dqn = DQN(env.observation_space.shape, env.action_space.n)

    for i in range(10000):
        observation, info = env.reset()
        game_reward = 0
        while True:
            action = dqn.predict(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            game_reward += reward
            dqn.add_expr(observation, action, reward, observation_, terminated)
            dqn.learn()
            if terminated or truncated:
                print(f"game {i} finished, reward: {game_reward}")
                break
    env.close()


if __name__ == "__main__":
    main()