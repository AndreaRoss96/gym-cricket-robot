from neural_network.actor_nn import Actor
from neural_network.critic_nn import Critic

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
    actor_target = Actor(
        num_states,
        num_actions,
        terrain_dim,
        terrain_output,
        hidden_layers=hidden_layers,
        conv_layers=conv_layers,
        kernel_sizes=kernel_sizes,
        init_w=init_w)
    critic = Critic(
        num_states + num_actions,
        num_actions,
        terrain_dim,
        terrain_output,
        hidden_layers=hidden_layers,
        conv_layers=conv_layers,
        kernel_sizes=kernel_sizes,
        action_features_out = action_features_out,
        output_critic = output_critic,
        init_w=init_w)
    critic_target = Critic(
        num_states + num_actions,
        num_actions,
        terrain_dim,
        terrain_output,
        hidden_layers=hidden_layers,
        conv_layers=conv_layers,
        kernel_sizes=kernel_sizes,
        action_features_out = action_features_out,
        output_critic = output_critic,
        init_w=init_w)

    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)
    
    return actor,critic,actor_target,critic_target