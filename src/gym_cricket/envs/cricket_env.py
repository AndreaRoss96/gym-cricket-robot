import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
from gym_cricket.assests.goal import Goal
from gym_cricket.assests.cricket import Cricket
"""
Blog interessanti da cui attingere conoscienza:
- Simple driving car
https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e
---> GitHub: https://github.com/GerardMaggiolino/Gym-Medium-Post/blob/48b512fd8f3616bb24b4c20695266ba8efcc6387/simple_driving/envs/simple_driving_env.py

- Automatic arm
https://www.etedal.net/2020/04/pybullet-panda_2.html
---> GitHub: https://github.com/mahyaret/gym-panda

- Deep Reinforcment Learning on Robot Grasping
https://github.com/BarisYazici/deep-rl-grasping
"""


'''
TODO:
Bro, qua di lavoro ce ne, io ti direi di ridaere una letta a quello che hai scritto
perche' il ragazzone maggiolino fa il fenomeno ma compie movimenti in 2D

Ad ogni modo dovrebbe essere tutto corretto dato che non abbiamo ancora lavorato a nessuna
parte delicata del codice (aka lo step e forse il render), tu riguarda nel dubbio

quindi mi concnetrerei sullo sviluppo dello step, che, in realta', dovrebbe anche considerare
le reti neurali che dovresti implementare sooner or later
'''

class CricketEnv(gym.Env):
    '''
    Gym environment used by the cricket robot to train it self
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        self.client = p.connect(p.GUI) # connect to PyBullet using GUI
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)
        # adjust the view angle of the environment
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-40,
            cameraTargetPosition=[0.55,-0.35,0.2])
        # Let's describe the format of valid actions and observations.
        self.observation_space = spaces.Box(
            # modifica con i dati delle rotazioni e della forza normale
            low=np.array([-10, -10, -1, -1, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([10, 10, 1, 1, 5, 5, 10, 10], dtype=np.float32))
        self.action_space = spaces.Box(
            # modifica con gli intervalli delle torsioni dei joint e delle track
            low=np.array([0, -.6], dtype=np.float32),
            high=np.array([1, .6], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()
        self.reset()


    def step(self, action):
        """
        To return:
         - reward
         - new_state
         - done (True/False)
         - info : string
        """
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        pos,angs,l_vel,a_vel = self.cricket.get_observations()
        

        pass
    
    def reset(self):
        ''' This function is used to reset the PyBullet environment '''
        self.step_counter = 0
        p.resetSimulation()     # reset PyBullet environment
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-10)

        # Plane: to chose the plane I can do a script that changes the path to the desired 
        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        self.cricketUid = p.loadURDF(os.path.join(urdfRootPath, "gym-cricket/gym_cricket/envs/assests/urdfs/cricket_robot.urdf"),useFixedBase=True)
        rest_poses = [range(np.random.uniform(-math.pi,math.pi,p.getNumJoints(self.cricketUid)))]
        # here you can set the position of the joints (randomly is good)
        for i in range(p.getNumJoints(self.cricketUid)):
            p.resetJointState(self.cricketUid,i, rest_poses[i]) # (bodyID, JointIndex,targetValue,targetVel, physicsClient)

        # set the final state of the cricket robot
        goal_state = [] # inserisci la posizione finale di tutti i joint che ti occorrono + la posizione e l'orientation 

        # init Cricket & goal
        self.cricket = Cricket(self.client)
        self.goal = Goal()

        # Che è sta robba zi?
        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return np.array(self.observation).astype(np.float32)


    def render(self, mode='human'):
        # che fa render?
    ...

    def close(self):
    ...

    def seed(self, seed=None):
        '''
        Gym provides seeding utilities we can use to ensure different training and demonstration
        runs are identical. We obtain a randomly seeded numpy random number generator that we’ll
        use for all random operations.
        '''
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]