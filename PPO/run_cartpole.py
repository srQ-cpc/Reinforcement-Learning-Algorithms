import gym
from ppo import PPO


def main():
    env = gym.make("CartPole-v1")
    ppo = PPO(env.observation_space.shape, env.action_space.n)

    for i in range(10000):
        observation, info = env.reset()
        game_reward = 0
        while True:
            action = ppo.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            game_reward += reward
            ppo.learn(observation, action, reward, observation_, terminated)
            if terminated or truncated:
                print(f"game {i} finished, reward: {game_reward}")
                break
    env.close()


if __name__ == "__main__":
    main()