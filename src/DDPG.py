from gym_cricket.envs.cricket_env import CricketEnv
from neural_network.actor_nn import Actor_nn
from neural_network.critic_nn import Critic_nn

# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#:~:text=Background,-(Previously%3A%20Introduction%20to&text=Deep%20Deterministic%20Policy%20Gradient%20(DDPG,function%20to%20learn%20the%20policy.
# https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py

def DDPG_algorith(episodes, num_states, hidden_size, num_actions):
    '''
    Initialize actor and critic networks 𝒜 𝑠, 𝑒|𝜃 𝒜 𝑎𝑛𝑑 𝒞 𝑠, 𝑎, 𝑒|𝜃𝒞 do: with random weights 𝜃 𝒜 𝑎𝑛𝑑 𝜃𝒞
    Initialize target network 𝒜′𝑎𝑛𝑑 𝒞′ with random weights 𝜃 𝒜′ ← 𝜃 𝒜 𝑎𝑛𝑑 𝜃𝒞′ ← 𝜃𝒞
    Initialize replay buffer ℛ
    '''
    actor, critic, actor_target, critic_target = init_nn(num_states, hidden_size, num_actions)
    buffer = init_buffer()
    done = False
    cricket_env = CricketEnv()
    gamma = 0.99
    for episode in range(episodes) :
        '''
        Procedure RobotInit
        Generate a random stable starting state 𝑠 in the unstructured environment
        end procedure RobotInit
        '''
        cricket_env.reset() # reset robot position (randomly) and restart the simulation
        while not done:
            # gererate action at←𝒜(st,e|θ^𝒜)
            current_state = cricket_env.get_state()
            env = cricket_env.get_env()

            # Execute action a t : observe reward r t and new state s t+1
            action = actor.generate_action(current_state,env)
            reward, new_state, done, info  = cricket_env.step(action)

            # Store transition s t , a t , r t , s t+1 in R
            buffer.add((current_state,action,reward,new_state))

            # update policy
            if update_time :
                minibatch = buffer.get_minibatch()
                # sample a random minibatch of N transitions
                for (state_i,action_i,reward_i,new_state_i) in minibatch:
                    y = reward_i + gamma*(1-)

def init_nn(num_states, hidden_size, num_actions):
    """
    initialize the target networks as copies of the original networks
    """
    actor = Actor_nn(num_states, hidden_size, num_actions)
    actor_target = Actor_nn(num_states, hidden_size, num_actions)
    critic = Critic_nn(num_states + num_actions, hidden_size, num_actions)
    critic_target = Critic_nn(num_states + num_actions, hidden_size, num_actions)

    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)
