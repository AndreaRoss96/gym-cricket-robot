import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import RankWarning
from gym_cricket.envs.cricket_env import CricketEnv
from DDPG import DDPG
from utils.OUNoise import OUNoise

env = CricketEnv()
ddpg = DDPG(env)
noise = OUNoise(env.action_space)

num_episodes = 10000
step_per_episode = 500
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(step_per_episode):
        action = ddpg.get_action(state) # invoke the actor nn to generate an action (compute forward)
        action = noise.get_action(action,step)
        reward, new_state, done, info = env.step(action)
        ddpg.replay_buffer.push(state,action,reward,new_state,done)

        if len(ddpg.replay_buffer) > batch_size :
            ddpg.update(batch_size)

        state = new_state
        episode_reward += reward

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

# TODO: Si riesce a farlo live? 
plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()


