"""
This code is derived from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5_Deep_Q_Network/RL_brain.py
Just for demonstration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.cost_his = []

    def _build_net(self):
        self.eval_net = nn.Sequential(
            nn.Linear(self.n_features, 10),
            nn.ReLU(),
            nn.Linear(10, self.n_actions)
        )

        self.target_net = nn.Sequential(
            nn.Linear(self.n_features, 10),
            nn.ReLU(),
            nn.Linear(10, self.n_actions)
        )

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)

        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(observation)
            action = torch.argmax(actions_value, dim=1).item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.target_net(torch.tensor(batch_memory[:, -self.n_features:], dtype=torch.float)), self.eval_net(torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float))

        q_target = q_eval.clone().detach()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.item())

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()