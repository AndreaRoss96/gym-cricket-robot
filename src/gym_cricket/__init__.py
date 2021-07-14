from gym.envs.registration import register

register(
    id='cricket-v0',
    entry_point='gym_cricket.envs:CricketEnv',
)