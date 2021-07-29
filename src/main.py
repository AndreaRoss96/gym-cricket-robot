import numpy as np
import matplotlib.pyplot as plt
import pywavefront as pw
import torch
from numpy.lib.polynomial import RankWarning

from gym_cricket.envs.cricket_env import CricketEnv
from DDPG import DDPG
from neural_network.actor_nn import Actor
from utils.OUNoise import OUNoise
from utils.auxiliaryFuncs import init_nn

env = CricketEnv()
noise = OUNoise(env.action_space)

num_episodes = 10000
step_per_episode = 500
batch_size = 128
rewards = []
avg_rewards = []

# Set the final Goal @TODO read this from a file
wheels = [0.0] * 8
limbs = [0.0, -np.pi/2, np.pi, -np.pi/2, 0.0, np.pi/2,\
    np.pi, np.pi/2, 0.0,-np.pi/2, np.pi, -np.pi/2, 0.0,\
    np.pi/2, np.pi, np.pi/2, 0.0, 0.0]
goals = np.concatenate([wheels,limbs])
env.set_goal(joint_position=goals)

# Set the terrain @TODO read this from a file
scene = pw.Wavefront('/home/andrea/Downloads/flat.obj')
terrain = np.array(scene.vertices)
terrain = np.reshape(terrain, (4,3,1,1,1))
terrain = torch.FloatTensor(terrain)

# Initialize neural networks
actor, critic, actor_target, critic_target = init_nn(env,terrain,kernel_sizes=[1])

# Initialize DDPG 
ddpg = DDPG(env, actor, critic, actor_target, critic_target, terrain)

for episode in range(num_episodes):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(step_per_episode):
        state = [elem for sub_obs in state for elem in [*sub_obs]] # unpacking the elements from the format returned by cricket env
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


