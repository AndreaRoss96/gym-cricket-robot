# external dependecies
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# internal dependecies
# from neural_network.actor_nn import Actor_nn
# from neural_network.critic_nn import Critic_nn
from utils.buffer import Buffer
# from utils.memory import SequentialMemory,OrnsteinUhlenbeckProcess
from utils.util import *

# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#:~:text=Background,-(Previously%3A%20Introduction%20to&text=Deep%20Deterministic%20Policy%20Gradient%20(DDPG,function%20to%20learn%20the%20policy.
# https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py
# https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b

criterion = torch.nn.MSELoss()

class DDPG:
    def __init__(self, env, actor, critic, actor_target, critic_target, terrain, batch_size = 128,
    gamma=0.99, tau=1e-2, ou_theta = 0.15, ou_sigma = 0.2, ou_mu = 0.0, buffer_maxlen=6000000, 
    critic_learning_rate=1e-3, actor_learning_rate=1e-4, window_length = 1,):
        """
        params:
         - env : gym environment
         - gamma : 
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.terrain = torch.FloatTensor(terrain).unsqueeze(0).to(self.device)
        self.terrain_full = None
        #self.terrain = terrain
        self.env = env
        self.obsv_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]

        # hyperparameters
        self.gamma = gamma  # to update the critic by minimizing the loss
        self.tau = tau      # to update the target networks

        # # init actor and critic (origin and target) networks
        # self.actor,\
        #     self.critic,\
        #         self.actor_target,\
        #             self.critic_target = self.__init_nn(self.obsv_space, self.action_space)
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target

        # batch size
        self.batch_size = batch_size

        # optimizers
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        # hard_update(self.critic_target, self.critic)

        #Create replay buffer
        # self.memory = SequentialMemory(limit=buffer_maxlen, window_length=window_length)
        # self.random_process = OrnsteinUhlenbeckProcess(size=self.action_space, theta=ou_theta, mu=ou_mu, sigma=ou_sigma)
        self.replay_buffer = Buffer(buffer_maxlen)

        if USE_CUDA: self.cuda()

    def get_action(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        # terrain = torch.FloatTensor(self.terrain).unsqueeze(0).to(self.device)
        action = self.actor.forward(state, self.terrain[0])
        action = action.squeeze(0).cpu().detach().numpy()

        return action

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
    
    def update(self, batch_size):
        torch.cuda.empty_cache()
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        # if self.terrain_full == None :
        #     self.terrain_full = []
        #     for _ in range(batch_size) :
        #         #self.terrain_full = torch.cat((self.terrain_full, self.terrain))
        #         self.terrain_full.append(self.terrain.cpu().numpy())
            # self.terrain_full = torch.FloatTensor(self.terrain_full).to(self.device)

        # Critic loss
        curr_Q = self.critic.forward(state_batch, self.terrain[0], action_batch)
        next_actions = self.actor_target.forward(next_state_batch, self.terrain[0])
        next_Q = self.critic_target.forward(next_state_batch, self.terrain[0], next_actions.detach())
        expected_Q = reward_batch + self.gamma * next_Q
        q_loss = F.mse_loss(curr_Q, expected_Q.detach()) # critic loss
        q_loss_arr = q_loss.data.cpu().numpy()
        self.env.push_loss(q_loss_arr)

        # update critic network
        self.critic_optimizer.zero_grad()
        q_loss.backward() 
        self.critic_optimizer.step()

        # actor loss
        action_for_loss = self.actor.forward(state_batch, self.terrain)
        policy_loss = -self.critic.forward(observations=state_batch, terrain= self.terrain, action = action_for_loss).mean()
        
        # update actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
    
    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
    # def __init_nn(self, num_states, num_actions):
    #     """
    #     initialize the target networks as copies of the original networks
    #     """
    #     actor = Actor_nn(num_states, num_actions)
    #     actor_target = Actor_nn(num_states, num_actions)
    #     critic = Critic_nn(num_states + num_actions, num_actions)
    #     critic_target = Critic_nn(num_states + num_actions, num_actions)

    #     for target_param, param in zip(actor_target.parameters(), actor.parameters()):
    #         target_param.data.copy_(param.data)
    #     for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    #         target_param.data.copy_(param.data)
        
    #     return actor,critic,actor_target,critic_target
