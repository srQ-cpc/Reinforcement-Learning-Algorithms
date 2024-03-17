import gym
from ddpg import DDPG


def main():
    env = gym.make("Pendulum-v1", render_mode="human")
    ddpg = DDPG(env.observation_space.shape, env.action_space.shape, env.action_space.high)
    for i in range(10000):
        observation, info = env.reset()
        game_reward = 0
        while True:
            action = ddpg.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            game_reward += reward
            ddpg.add_expr(observation, action, reward, observation_, terminated)
            ddpg.learn()
            if terminated or truncated:
                print(f"game {i} finished, reward: {game_reward}")
                break
    env.close()


if __name__ == "__main__":
    main()