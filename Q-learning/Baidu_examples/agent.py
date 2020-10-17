"""
    Author:VolcanicSnow
    Email:liupf2792@gmail.com
    Function：Q_learning Agent
    Version：1.0
    Date：2020/8/14 16:16
"""
import numpy as np


class QLearningAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate,
                 gamma,
                 e_greed):
        self.act_n = act_n          # 动作的维度
        self.lr = learning_rate     # 学习率
        self.gamma = gamma          # 衰减因子
        self.epsilon = e_greed      # 探索，以（1 - epsilon）的概率选择策略动作
        self.Q = np.zeros((obs_n, act_n))   # Q 表格

    def sample(self, obs):
        """
        根据输入的观察值，选择动作
        :param obs: 当前的观察值，即状态 S_t
        :return: action
        """
        if np.random.uniform(0, 1) > self.epsilon:      # 以（1 - epsilon）的概率选择 Q 表中的动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)       # 以 epsilon 的概率随机选取动作
        return action

    def predict(self, obs):
        """
        根据观察值，从 Q 表中选择动作
        :param obs: 当前的观察值
        :return: action
        """
        Q_list = self.Q[obs, :]
        Q_max = np.max(Q_list)
        action_list = np.where(Q_list == Q_max)[0]  # 可能有多个动作最大值一致
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, act, reward, next_obs, done):
        """
        更新 Q 表
        :param reward:
        :param obs:当前的观察值，即 S_t
        :param act: 当前状态所要采取的动作 A_t
        :param next_obs: 下一个观察值，S_t+1
        :param done: episode 是否结束
        :return: 没有返回值
        """
        Q_s_t = self.Q[obs, act]
        if done:    # 回合结束，没有下一个状态
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])    # Q-learning
        self.Q[obs, act] = Q_s_t + self.lr * (target_Q - Q_s_t)             # 更新 Q 表
