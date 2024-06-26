import time

from maze_env import Maze
from RL_brain_classic import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(5000):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)
            # action = RL.choose_action_and_validate(observation, env.action_is_valid)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                env.render()
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # Train maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.55,
                      replace_target_iter=200,
                      memory_size=2000,
                      use_double_DQN=True,
                      is_train=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
    RL.save_to_cache()
