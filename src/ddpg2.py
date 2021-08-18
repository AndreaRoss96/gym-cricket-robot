
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from utils.memory import SequentialMemory
from utils.random_process import OrnsteinUhlenbeckProcess
from utils.util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, env, actor, critic, actor_target, critic_target, terrain, args):
        """
        params:
         - env : gym environment
         - gamma : 
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise Exception("cuda not found!")

        self.terrain = torch.FloatTensor(terrain).unsqueeze(0).to(self.device)
        self.terrain_full = None
        #self.terrain = terrain
        self.env = env
        self.obsv_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        #
        # hyperparameters
        self.tau = args.tau      # to update the target networks
        self.discount = args.discount
        self.epsilon = 1.0
        self.depsilon = 1.0 / self.epsilon
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True
        # batch size
        self.batch_size = args.bsize

        # optimizers
        self.actor_optimizer  = Adam(self.actor.parameters(), lr=args.prate) # lr= learning rate
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.prate)

        # hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        # hard_update(self.critic_target, self.critic)

        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.action_space, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        if state_batch.shape == (64,1):
            tmp = []
            for arr in state_batch:
                tmp.append(arr[0][0])
            state_batch = np.array(tmp, dtype=np.float64)
        # Prepare for the target q batch
        next_q_values = self.critic_target(
            to_tensor(next_state_batch, volatile=True),
            self.terrain[0],
            self.actor_target(to_tensor(next_state_batch, volatile=True),self.terrain[0])
        )
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic(
            to_tensor(state_batch),
            self.terrain[0],
            to_tensor(action_batch))
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic(
            to_tensor(state_batch),
            self.terrain[0],
            self.actor(to_tensor(state_batch), self.terrain[0])
        )

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.action_space)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])), self.terrain[0])
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

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

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
