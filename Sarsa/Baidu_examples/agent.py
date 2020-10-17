import numpy as np


class SarsaAgent(object):
    def __init__(self,
                obs_n,  # 状态空间
                act_n,  # 动作空间
                gamma,  # 衰减因子
                learning_rate,  # 学习率
                e_greed):   # 探索因子
        self.act_n = act_n
        self.gamma = gamma
        self.lr = learning_rate
        self.epsilon = e_greed
        self.Q = np.zeros((obs_n, act_n))   # 初始化Q表

    def sample(self, obs):
        """
        根据输入的观察值选择动作
        :param obs: 观察值，即当前的状态 S_t
        :return: 动作 A_t
        """
        if np.random.uniform(0, 1) < (1 - self.epsilon):    # 从 Q 表中选择动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    def predict(self, obs):
        """
        根据 输入的观察值，预测输出的动作
        :param obs_n: 当前的观察值，即当前的状态值 S_t
        :return: 动作，A_t
        """
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]   # max_Q 可能对应这多个 action
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, act, reward, next_obs, next_act, done):
        """
        学习过程，更新 Q 表
        :param obs: 当前的观察值 S_t
        :param act: 当前的动作   A_t
        :param reward: 获得的奖励值
        :param next_obs: 下一个观察值，即 S_t+1
        :param next_act: 下一个动作，即 A_t+1
        :param done: episode 是否结束
        """
        Q_s_a = self.Q[obs, act]
        if done:
            target_Q = reward   # 到达终点，没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_act]     # Sarsa
        self.Q[obs, act] += self.lr * (target_Q - Q_s_a)    # 更新 Q 表
