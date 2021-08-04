from neural_network.actor_nn import Actor
from neural_network.critic_nn import Critic
from torch.autograd import Variable
import torch
import numpy as np
import copy
import os

def init_nn(env, terrain, hidden_layers = [400,300,200], conv_layers = [], kernel_sizes = [150,100,50], action_features_out = 5, init_w=3e-3, output_critic = None):
    """
    initialize the target networks as copies of the original networks
    """
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    terrain_dim = terrain.shape[1]
    terrain_output = terrain.shape[0]

    actor = Actor(
        num_states,
        num_actions,
        terrain_dim,
        terrain_output,
        hidden_layers=hidden_layers,
        conv_layers=conv_layers,
        kernel_sizes=kernel_sizes,
        init_w=init_w)
    actor_target = copy.deepcopy(actor)
    #  Actor(
    #     num_states,
    #     num_actions,
    #     terrain_dim,
    #     terrain_output,
    #     hidden_layers=hidden_layers,
    #     conv_layers=conv_layers,
    #     kernel_sizes=kernel_sizes,
    #     init_w=init_w)
    critic = Critic(
        num_states,
        num_actions,
        terrain_dim,
        terrain_output,
        hidden_layers=hidden_layers,
        conv_layers=conv_layers,
        kernel_sizes=kernel_sizes,
        action_features_out = action_features_out,
        output_critic = output_critic,
        init_w=init_w)
    critic_target = copy.deepcopy(critic)
    # Critic(
    #     num_states,
    #     num_actions,
    #     terrain_dim,
    #     terrain_output,
    #     hidden_layers=hidden_layers,
    #     conv_layers=conv_layers,
    #     kernel_sizes=kernel_sizes,
    #     action_features_out = action_features_out,
    #     output_critic = output_critic,
    #     init_w=init_w)

    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)
    
    return actor,critic,actor_target,critic_target

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=np.float32):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir