"""
    Author:VolcanicSnow
    Email:liupf2792@gmail.com
    Function：在环境中运行 Agent
    Version：1.0
    Date：2020/8/14 16:16
"""
import gym
from gridworld import CliffWalkingWapper, FrozenLakeWapper
from agent import QLearningAgent
import time


def run_episode(env, agent, render=False):
    total_steps = 0
    total_reward = 0

    obs = env.reset()
    while True:
        action = agent.sample(obs)  # 选择一个动作
        next_obs, reward, done, _ = env.step(action)
        # 训练 Q-learning
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs
        total_steps += 1
        total_reward += reward

        if render:
            env.render()
        if done:
            break
    return total_steps, total_reward


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        total_reward += reward
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)

    agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)

    is_render = False
    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps,
                                                          ep_reward))

        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False
    # 训练结束，查看算法效果
    test_episode(env, agent)


if __name__ == "__main__":
    main()