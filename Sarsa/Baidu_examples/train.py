import gym
from gridworld import CliffWalkingWapper, FrozenLakeWapper
from agent import SarsaAgent
import time


def run_episode(env, agent, render=False):
    """
    训练
    :param env: 训练环境
    :param agent: 参与训练的 Agent
    :param render: 环境是否渲染
    :return: 一个回合内总的步数，一个回合内总的奖励
    """
    total_steps = 0     # 一个回合内总的步数
    total_reward = 0    # 一个回合内总的奖励

    obs = env.reset()   # 重置环境，即重新开始一个新的 episode
    action = agent.sample(obs)  # 根据算法来选择一个动作

    while True:
        next_obs, reward, done, _ = env.step(action)    # 执行动作，与环境进行一次交互
        next_action = agent.sample(next_obs)     # 获取下一状态所要采取的动作

        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs
        total_steps += 1
        total_reward += reward
        if render:
            env.render()
        if done:
            break
    return total_steps, total_reward


def test_episode(env, agent):
    """
    测试训练完后的环境
    :param env: 环境
    :param agent: Agent
    :return:
    """
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(1)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    # 初始化 环境
    # 冰湖环境
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)

    # 悬崖环境
    env = gym.make("CliffWalking-v0")
    env = CliffWalkingWapper(env)

    # 初始化 Agent
    agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)

    # 开始训练
    render = False
    for episode in range(500):
        ep_steps, ep_reward = run_episode(env, agent, render)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps,
                                                          ep_reward))
        # 每隔 20 个 episode 看一下效果
        if episode % 20 == 0:
            render = True
        else:
            render = False

    # 训练结束，看一下效果
    test_episode(env, agent)


if __name__ == '__main__':
    main()
