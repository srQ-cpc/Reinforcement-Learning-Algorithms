from sarsa import Sarsa
from enviroments.maze_env import Maze


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        action = model.predict(observation)

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on observation
            action_ = model.predict(observation_)

            # RL learn from this transition
            model.fit(observation, action, reward, observation_, action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    model = Sarsa(env.n_actions)

    env.after(100, update)
    env.mainloop()
