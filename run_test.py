import time
from maze_env import Maze
from RL_brain_classic import DeepQNetwork


def run_maze():
    win_cnt = 0
    lose_cnt = 0
    # play 10 rounds
    for episode in range(10):
        # initial observation
        observation = env.reset()
        action = None
        while True:
            # fresh env
            env.render()
            # wait for human observe
            if env.action_is_valid(action):
                time.sleep(0.2)

            # RL choose action based on observation
            action = RL.choose_action(observation)
            # action = RL.choose_action_and_validate(observation, env.action_is_valid)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                env.render()
                time.sleep(0.1)
                if reward > 0:
                    print("win")
                    win_cnt += 1
                else:
                    print("lose")
                    lose_cnt += 1
                time.sleep(3)
                break

    print("succ rate {0}%".format(win_cnt*100.0/(win_cnt + lose_cnt)))

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # Test maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.95,
                      replace_target_iter=200,
                      memory_size=2000,
                      is_train=False
                      )
    env.after(100, run_maze)
    env.mainloop()
