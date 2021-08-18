import numpy as np
import argparse
import matplotlib.pyplot as plt
import pywavefront as pw
from copy import deepcopy
import torch
from numpy.lib.polynomial import RankWarning
import time
from utils.util import get_output_folder

from gym_cricket.envs.cricket_env import CricketEnv
# from DDPG import DDPG
from ddpg2 import DDPG
from neural_network.actor_nn import Actor
from utils.OUNoise import OUNoise
from utils.auxiliaryFuncs import init_nn

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode',       default='train', type=str, help='support option: train/test')
    parser.add_argument('--env',        default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1',    default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2',    default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate',       default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate',      default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup',     default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount',   default=0.99, type=float, help='')
    parser.add_argument('--bsize',      default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize',     default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau',        default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta',   default=0.0001, type=float, help='noise theta')
    parser.add_argument('--ou_sigma',   default=0.0002, type=float, help='noise sigma')
    parser.add_argument('--ou_mu',      default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output',     default='output', type=str, help='')
    parser.add_argument('--debug',      dest='debug', action='store_true')
    parser.add_argument('--init_w',     default=0.003, type=float, help='')
    parser.add_argument('--train_iter', default=200000,type=int, help='train iters each timestep')
    parser.add_argument('--epsilon',    default=50000,type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed',       default=-1, type=int, help='')
    parser.add_argument('--resume',     default='default',type=str, help='Resuming model path for testing')
    parser.add_argument('--early_stop', default=100, type=int,help='change episode after [early_stop] steps with a non-growing reward')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        # args.resume = 'output/{}-run6'.format(args.env)
        args.resume = 'output/{}-run0'.format(args.env)


    env = CricketEnv()
    noise = OUNoise(env.action_space)

    num_episodes = 10000
    step_per_episode = 400
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
    _, limb_joints, _ = env.cricket.get_joint_ids()
    num_limb_joints = len(limb_joints)
    env.set_reward_values(w_joints= np.full((num_limb_joints,), 1.5),disc_factor=0.5,w_X=1,w_Y=1,w_Z=1,w_theta=1,w_sigma=1)

    # Set the terrain @TODO read this from a file
    scene = pw.Wavefront('/home/andrea/Downloads/flat.obj')
    terrain = np.array(scene.vertices)
    terrain = np.reshape(terrain, (4,3,1,1,1))
    # terrain = torch.FloatTensor(terrain)

    # Initialize neural networks
    actor, critic, actor_target, critic_target = init_nn(
        env,terrain,hidden_layers=[100,50], kernel_sizes=[1])

    # Initialize DDPG 
    ddpg = DDPG(env, actor, critic, actor_target, critic_target, terrain,args)

    # output
    output = 'weights_out0'
    output = get_output_folder(output, 'cricket-v0')

    file = open("action_out.txt", "w")
    for episode in range(num_episodes):
        state = env.reset()
        ddpg.reset(state) # new
        noise.reset() # delete
        episode_reward = 0

        for step in range(step_per_episode):
            # state = [elem for sub_obs in state for elem in [*sub_obs]] # unpacking the elements from the format returned by cricket env
            action = ddpg.select_action(state) #.get_action(state) # invoke the actor nn to generate an action (compute forward)
            # print(action)
            file.write(f'Action {action}\n\n')
            #action = noise.get_action(action,step)
            #file.write(f'Action_noise {action}\n\n')

            reward, new_state, done, info = env.step(action)
            new_state = deepcopy(new_state)
            ddpg.observe(reward,new_state,done)
            # ddpg.replay_buffer.push(state,action,reward,new_state,done)

            if step > args.warmup:
                ddpg.update_policy()
            # if len(ddpg.replay_buffer) > batch_size :
            #     ddpg.update(batch_size)

            state = new_state
            episode_reward += reward

            if done :
                print('!'*80)
                break

        if episode % int(num_episodes/3) == 0:
            ddpg.save_model(output)

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


