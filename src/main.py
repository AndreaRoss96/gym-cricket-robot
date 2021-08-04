import numpy as np
import matplotlib.pyplot as plt
import pywavefront as pw
import torch
from numpy.lib.polynomial import RankWarning
import time
import warnings
warnings.filterwarnings("ignore")

from gym_cricket.envs.cricket_env import CricketEnv
from DDPG import DDPG
from neural_network.actor_nn import Actor
from utils.OUNoise import OUNoise
from utils.auxiliaryFuncs import init_nn

env = CricketEnv()
noise = OUNoise(env.action_space)

num_episodes = 1000
step_per_episode = 500
batch_size = 16
rewards = []
avg_rewards = []

# Set the final Goal @TODO read this from a file
wheels = [0.0] * 8
limbs = [0.0, -np.pi/2, np.pi, -np.pi/2, 0.0, np.pi/2,\
    np.pi, np.pi/2, 0.0,-np.pi/2, np.pi, -np.pi/2, 0.0,\
    np.pi/2, np.pi, np.pi/2, 0.0, 0.0]
goals = np.concatenate([wheels,limbs])
env.set_goal(joint_position=goals)
env.set_reward_values(w_X=1,w_Y=1,w_Z=1,early_stop_limit=150)

# Set the terrain @TODO read this from a file
scene = pw.Wavefront('/home/andrea/Downloads/flat.obj')
terrain = np.array(scene.vertices)
terrain = np.reshape(terrain, (4,3,1,1,1))
# terrain = torch.FloatTensor(terrain)

# Initialize neural networks
actor, critic, actor_target, critic_target = init_nn(env,terrain,kernel_sizes=[1])

# Initialize DDPG 
ddpg = DDPG(env, actor, critic, actor_target, critic_target, terrain)

file = open("action_out.txt", "w")
for episode in range(num_episodes):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(step_per_episode):
        # state = [elem for sub_obs in state for elem in [*sub_obs]] # unpacking the elements from the format returned by cricket env
        action = ddpg.get_action(state) # invoke the actor nn to generate an action (compute forward)
        print(action)
        file.write(f'Action {action}\n\n')
        #action = noise.get_action(action,step)
        #file.write(f'Action_noise {action}\n\n')

        reward, new_state, done, info = env.step(action)
        ddpg.replay_buffer.push(state,action,reward,new_state,done)

        if len(ddpg.replay_buffer) > batch_size :
            print("------"*80)
            print('UPDATE')
            print("------"*80)
            ddpg.update(batch_size)

        state = new_state
        episode_reward += reward

        if done :
            print('!'*80)
            break
        time.sleep(1/1)
    rewards.append(episode_reward)
    print('_'*40)
    print(f'episode no: {episode}')
    print(f'episode reward: {episode_reward}')
    n = 10
    print(f'last {n} episode reward: {rewards[-n:]}')
    print('_'*40)
    print()
    
    avg_rewards.append(np.mean(rewards[-10:]))
file.close()

# TODO: Si riesce a farlo live?
ddpg.save_model('/home/andrea/Desktop/project/gym-cricket-robot/src/weights_out') # add read/load directory for the measures of the goal and then use it as a output
plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()


