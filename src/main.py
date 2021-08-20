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
from ddpg import DDPG
from neural_network.actor_nn import Actor
from utils.OUNoise import OUNoise
from utils.auxiliaryFuncs import init_nn

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch on TORCS with Multi-modal')
    # environment arguments
    parser.add_argument('--mode',           default='train', type=str, help='support option: train/test')
    parser.add_argument('--env',            default='Cricket-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--num_episodes',   default=10000, type=int, help='total training episodes')
    parser.add_argument('--step_episode',   default=400, type=int, help='simulation steps per episode')
    parser.add_argument('--early_stop',     default=100, type=int, help='change episode after [early_stop] steps with a non-growing reward')
    parser.add_argument('--cricket',        default='basic_cricket', type=str, help='[hebi_cricket, basic_cricket] - cricket urdf model you want to load')
    parser.add_argument('--terrain',        default='flat', type=str, help='name of the terrain you want to load')
    # reward function
    parser.add_argument('--w_X',            default=0.5, type=float, help='weight X to compute difference between the robot and the optimal position. Used in the reward function')
    parser.add_argument('--w_Y',            default=0.5, type=float, help='weight Y to compute difference between the robot and the optimal position. Used in the reward function')
    parser.add_argument('--w_Z',            default=0.5, type=float, help='weight Z to compute difference between the robot and the optimal position. Used in the reward function')
    parser.add_argument('--w_theta',        default=0.5, type=float, help='weight theta to compute difference between the robot and the optimal position. Used in the reward function')
    parser.add_argument('--w_sigma',        default=0.5, type=float, help='weight sigma to compute difference between the robot and the optimal position. Used in the reward function')
    parser.add_argument('--disct_factor',   default=0.99, type=float, help='discount factor for learnin in the reward function')
    parser.add_argument('--w_joints',       default=1.5, type=float, help='weight to punish bad joints behaviours in the reward function')
    # neural networks
    parser.add_argument('--hidden1',        default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2',        default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--hidden3',        default=150, type=int, help='hidden num of third fully connect layer')
    parser.add_argument('--hidden4',        default=0, type=int, help='hidden num of fourth fully connect layer')
    parser.add_argument('--hidden5',        default=0, type=int, help='hidden num of fifth fully connect layer')
    parser.add_argument('--conv_hidden1',   default=0, type=int, help='hidden num of first convolutional layer')
    parser.add_argument('--conv_hidden2',   default=0, type=int, help='hidden num of second convolutional layer')
    parser.add_argument('--conv_hidden3',   default=0, type=int, help='hidden num of third convolutional layer')
    parser.add_argument('--conv_hidden4',   default=0, type=int, help='hidden num of fourth convolutional layer')
    parser.add_argument('--conv_hidden5',   default=0, type=int, help='hidden num of fifth convolutional layer')
    parser.add_argument('--kernel_size1',   default=1, type=int, help='num of first kernel for cnn')
    parser.add_argument('--kernel_size2',   default=0, type=int, help='num of second kernel for cnn')
    parser.add_argument('--kernel_size3',   default=0, type=int, help='num of third kernel for cnn')
    parser.add_argument('--kernel_size4',   default=0, type=int, help='num of fourth kernel for cnn')
    # ddpg arguments
    parser.add_argument('--bsize',          default=128, type=int, help='minibatch size')
    parser.add_argument('--rate',           default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate',          default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup',         default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount',       default=0.99, type=float, help='')
    parser.add_argument('--rmsize',         default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length',  default=1, type=int, help='')
    parser.add_argument('--tau',            default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta',       default=0.0001, type=float, help='noise theta')
    parser.add_argument('--ou_sigma',       default=0.0002, type=float, help='noise sigma')
    parser.add_argument('--ou_mu',          default=0.0, type=float, help='noise mu')
    # TODO
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output',         default='output', type=str, help='')
    parser.add_argument('--debug',          dest='debug', action='store_true')
    parser.add_argument('--init_w',         default=0.003, type=float, help='')
    parser.add_argument('--train_iter',     default=200000,type=int, help='train iters each timestep')
    parser.add_argument('--epsilon',        default=50000,type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed',           default=-1, type=int, help='')
    parser.add_argument('--resume',         default='default',type=str, help='Resuming model path for testing')

    # parsing argument
    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        # args.resume = 'output/{}-run6'.format(args.env)
        args.resume = 'output/{}-run0'.format(args.env)


    env = CricketEnv(
        #plane_path='src/gym_cricket/assests/terrains/' + args.terrain + '.urdf',
        cricket_model = args.cricket)
    noise = OUNoise(env.action_space)

    num_episodes = args.num_episodes
    step_per_episode = args.step_episode
    batch_size = 128
    rewards = []
    avg_rewards = []

    ## Set the final Goal @TODO read this from a file
    wheels = [0.0] * 8
    limbs = [0.0, -np.pi/2, np.pi, -np.pi/2, 0.0, np.pi/2,\
        np.pi, np.pi/2, 0.0,-np.pi/2, np.pi, -np.pi/2, 0.0,\
        np.pi/2, np.pi, np.pi/2, 0.0, 0.0]
    goals = np.concatenate([wheels,limbs])
    env.set_goal(joint_position=goals)

    _, limb_joints, _ = env.cricket.get_joint_ids()
    num_limb_joints = len(limb_joints)
    env.set_reward_values(
        w_joints = np.full((num_limb_joints,), args.w_joints),
        disc_factor = 0.5,
        w_X=args.w_X, w_Y=args.w_X, w_Z=args.w_X,
        w_theta=args.w_theta ,w_sigma=args.w_theta)

    scene = pw.Wavefront(r'src/gym_cricket/assests/terrains/' + args.terrain + '.obj')
    terrain = np.array(scene.vertices)
    terrain = np.reshape(terrain, (4,3,1,1,1))
    # terrain = torch.FloatTensor(terrain)

    ## Initialize neural networks
    # hidden layers for fully connected neural network (robot)
    hidden_layers = [args.hidden1,args.hidden2,args.hidden3,args.hidden4,args.hidden5]
    hidden_layers = [layers for layers in hidden_layers if layers is not 0]

    # convolutional layers for convolutional neural network (terrain)
    conv_hidden_layers = [args.conv_hidden1,args.conv_hidden2,args.conv_hidden3,args.conv_hidden4,args.conv_hidden5]
    conv_hidden_layers = [layers for layers in conv_hidden_layers if layers is not 0]

    # kernel sizes for convolutional neural network (terrain)
    kernel_sizes = [args.kernel_size1, args.kernel_size2, args.kernel_size3, args.kernel_size4]
    kernel_sizes = [layers for layers in kernel_sizes if layers is not 0]

    actor, critic, actor_target, critic_target = init_nn(
        env, terrain,
        hidden_layers = hidden_layers,
        conv_layers= conv_hidden_layers,
        kernel_sizes=kernel_sizes)

    # Initialize DDPG 
    ddpg = DDPG(env, actor, critic, actor_target, critic_target, terrain,args)

    # output
    output = 'weights_out0'
    output = get_output_folder(output, 'cricket-v0')

    # file = open("action_out.txt", "w")
    for episode in range(num_episodes):
        state = env.reset()
        ddpg.reset(state) # new
        noise.reset() # delete
        episode_reward = 0

        for step in range(step_per_episode):
            action = ddpg.select_action(state) #.get_action(state) # invoke the actor nn to generate an action (compute forward)
            # file.write(f'Action {action}\n\n')

            reward, new_state, done, info = env.step(action)
            new_state = deepcopy(new_state)
            ddpg.observe(reward,new_state,done)

            if step > args.warmup:
                ddpg.update_policy()

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
    # file.close()

    ddpg.save_model(output) # add read/load directory for the measures of the goal and then use it as a output
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


