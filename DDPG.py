from gym_cricket.envs.cricket_env import CricketEnv
from neural_network.actor_nn import Actor_nn

# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#:~:text=Background,-(Previously%3A%20Introduction%20to&text=Deep%20Deterministic%20Policy%20Gradient%20(DDPG,function%20to%20learn%20the%20policy.
# https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py
def DDPG_algorith(episodes:int):
    '''
    Initialize actor and critic networks 𝒜 𝑠, 𝑒|𝜃 𝒜 𝑎𝑛𝑑 𝒞 𝑠, 𝑎, 𝑒|𝜃𝒞 do: with random weights 𝜃 𝒜 𝑎𝑛𝑑 𝜃𝒞
    Initialize target network 𝒜′𝑎𝑛𝑑 𝒞′ with random weights 𝜃 𝒜′ ← 𝜃 𝒜 𝑎𝑛𝑑 𝜃𝒞′ ← 𝜃𝒞
    Initialize replay buffer ℛ
    '''
    actor, critic, actor_target, critic_target = init_nn()
    buffer = init_buffer()
    done = False
    cricket_env = CricketEnv()
    for n in range(episodes) :
        '''
        Procedure RobotInit
        Generate a random stable starting state 𝑠 in the unstructured environment
        end procedure RobotInit
        '''
        cricket_env.reset() # reset robot position (randomly) and restart the simulation
        while not done:
            current_state = cricket_env.get
            action = actor.generate_action()
            reward, new_state = cricket_env.step()
            buffer.add((current_state,action,reward,new_state))
            minibatch = buffer.get_minibatch()
            # sample a random minibatch of N transitions
            for (state_i,action_i,reward_i,new_state_i) in minibatch:



