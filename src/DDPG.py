# external dependecies
import torch
import torch.optim as optim
import torch.nn.functional as F
# internal dependecies
from neural_network.actor_nn import Actor_nn
from neural_network.critic_nn import Critic_nn
from utils.buffer import Buffer

# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#:~:text=Background,-(Previously%3A%20Introduction%20to&text=Deep%20Deterministic%20Policy%20Gradient%20(DDPG,function%20to%20learn%20the%20policy.
# https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py
# https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b

class DDPG:
    def __init__(self, env, gamma=0.99, tau=1e-2, buffer_maxlen=50000, critic_learning_rate=1e-3, actor_learning_rate=1e-4):
        """
        params:
         - env : gym environment
         - gamma : 

        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.obsv_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]

        # hyperparameters
        self.gamma = gamma  # to update the critic by minimizing the loss
        self.tau = tau      # to update the target networks

        # init actor and critic (origin and target) networks
        self.actor,\
            self.critic,\
                self.actor_target,\
                    self.critic_target = self.__init_nn(self.obsv_space, self.action_space)
        
        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
    
        self.replay_buffer = Buffer(buffer_maxlen)

    def get_action(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()

        return action
    
    def update(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)

        # Critic loss
        curr_Q = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(next_state_batch)
        next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
        expected_Q = reward_batch + self.gamma * next_Q
        q_loss = F.mse_loss(curr_Q, expected_Q.detach()) # critic loss

        # update critic network
        self.critic_optimizer.zero_grad()
        q_loss.backward() 
        self.critic_optimizer.step()

        # actor loss
        policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()
        
        # update actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def __init_nn(self, num_states, num_actions):
        """
        initialize the target networks as copies of the original networks
        """
        actor = Actor_nn(num_states, num_actions)
        actor_target = Actor_nn(num_states, num_actions)
        critic = Critic_nn(num_states + num_actions, num_actions)
        critic_target = Critic_nn(num_states + num_actions, num_actions)

        for target_param, param in zip(actor_target.parameters(), actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(critic_target.parameters(), critic.parameters()):
            target_param.data.copy_(param.data)
        
        return actor,critic,actor_target,critic_target
