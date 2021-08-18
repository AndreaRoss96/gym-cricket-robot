#!/usr/bin/env python3

import numpy as np
import argparse
from copy import deepcopy

from gym_cricket.envs.cricket_env import CricketEnv
from utils.evaluator import Evaluator
from utils.auxiliaryFuncs import init_nn
from utils.util import *
from ddpg2 import DDPG
import pywavefront as pw
import matplotlib.pyplot as plt

# gym.undo_logger_setup()


def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    rewards = []
    avg_rewards = []
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        # env response with next_observation, reward, terminate_info
        reward, observation2, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup:
            agent.update_policy()

        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            print('_'*40)
            print("Evaluating...")
            print('_'*40)
            def policy(x): return agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(
                env, policy, debug=False, visualize=False)
            if debug:
                prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(
                    step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            if debug:
                prGreen('#{}: episode_reward:{} steps:{}'.format(
                    episode, episode_reward, step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            rewards.append(episode_reward)
            avg_rewards.append(np.mean(rewards[-10:]))
            print('_'*40)
            print(f'episode no: {episode}')
            print(f'episode reward: {episode_reward}')
            n = 10
            print(f'last {n} episode reward: {rewards[-n:]}')
            print('_'*40)

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
    return rewards, avg_rewards
    # plt.plot(rewards)
    # plt.plot(avg_rewards)
    # plt.plot()
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.show()


def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    def policy(x): return agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(
            env, policy, debug=debug, visualize=visualize, save=False)
        if debug:
            prYellow('[Evaluate] #{}: mean_reward:{}'.format(
                i, validate_reward))


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

    # Set the final Goal @TODO read this from a file
    wheels = [0.0] * 8
    limbs = [0.0, -np.pi/2, np.pi, -np.pi/2, 0.0, np.pi/2,
             np.pi, np.pi/2, 0.0, -np.pi/2, np.pi, -np.pi/2, 0.0,
             np.pi/2, np.pi, np.pi/2, 0.0, 0.0]
    goals = np.concatenate([wheels, limbs])
    env.set_goal(joint_position=goals)
    env.set_reward_values(w_X=1, w_Y=1, w_Z=1,
                          early_stop_limit=args.early_stop)

    # Set the terrain @TODO read this from a file
    scene = pw.Wavefront('/home/andrea/Downloads/flat.obj')
    terrain = np.array(scene.vertices)
    terrain = np.reshape(terrain, (4, 3, 1, 1, 1))

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    # Initialize neural networks
    actor, critic, actor_target, critic_target = init_nn(
        env, terrain, hidden_layers=[150,100,50], kernel_sizes=[1])

    # Initialize DDPG
    ddpg = DDPG(env, actor, critic, actor_target,
                critic_target, terrain, args)

    evaluate = Evaluator(args.validate_episodes,
                         args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        rewards, avg_rewards = train(args.train_iter, ddpg, env, evaluate, args.validate_steps,
              args.output, max_episode_length=args.max_episode_length, debug=True)#args.debug)
        plt.plot(rewards)
        plt.plot(avg_rewards)
        plt.plot()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()

    elif args.mode == 'test':
        test(args.validate_episodes, ddpg, env, evaluate, args.resume,
             visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
