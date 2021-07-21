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
https://github.com/openai/gym/blob/master/docs/creating-environments.md
Blog interessanti da cui attingere conoscienza:
- Simple driving car
https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e
---> GitHub: https://github.com/GerardMaggiolino/Gym-Medium-Post/blob/48b512fd8f3616bb24b4c20695266ba8efcc6387/simple_driving/envs/simple_driving_env.py

- Automatic arm
https://www.etedal.net/2020/04/pybullet-panda_2.html
---> GitHub: https://github.com/mahyaret/gym-panda

- Deep Reinforcment Learning on Robot Grasping
https://github.com/BarisYazici/deep-rl-grasping
file:///home/andrea/Desktop/repo_interest/deep-rl-grasping/final_report.pdf
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
    
    def set_reward_values(self,w_joints = None,w_error = 100,
                          disc_factor = 0.99, w_t = 0.0625,w_X = 0.5,w_Y = 0.5,
                          w_Z = 0.5,w_psi = 0.5,w_theta = 0.5,
                          w_sigma = 0.5):
        """Set the costants used in the reward function"""
        if w_joints == None :
            _, limb_joints, _ = self.cricket.get_joint_ids
            num_limb_joints = len(limb_joints)
            self.w_joints = np.full((num_limb_joints,), 1)
        else :
            self.w_joints = w_joints
        self.w_error = w_error
        self.disc_factor = disc_factor
        self.w_t = w_t
        self.w_X = w_X
        self.w_Y = w_Y
        self.w_Z = w_Z
        self.w_psi = w_psi
        self.w_theta = w_theta
        self.w_sigma = w_sigma



    def step(self, action):
        """Advance the Task by one step.

        Args:
            action (np.ndarray): The action to be executed.

        Returns: A tuple (reward, new_state, done, info)
         - reward
         - new_state
         - done (True/False)
         - info : string
        """
        self.episode_step += 1
        self.cricket.perform_action(action)
        pos,angs,l_vel,a_vel = self.cricket.get_observations()
        

        pass
    
    def __compute_reward(self, action):
        """Compute the reard based on the defined reward function"""
        # \mathcal{R}_t =-\sum_{i=0}^{no\_track}[\alpha_i(q_{t-1}^i)^2+\beta_i(q_{t-1}^i)^2+\kappa_i|\tau_{t-1}^i|+w_q^i(\Delta q_t^i)^2]-w_\varepsilon (\sum_{i=0}^5\gamma^i\varepsilon_{t-i})^2-w_tt-w_X(\Delta X)^2-w_Y(\Delta Y)^2-w_Z(\Delta Z)^2-w_\psi(\Delta \psi)^2-w_\theta(\Delta \theta)^2-w_\phi(\Delta \phi)^2
        reward = 0
        no_track_sum = 0
        _, limb_ids, _ = self.cricket.get_joint_ids()
        # Penalty if the robot touches itself
        self_collisions = self.cricket.get_joint_collisions()
        for id,coll in self_collisions.items() :
            no_track_sum += self.__joint_penalty(id)**2 # alpha_i(q_{t-1}^i)^2
        # Penalty if the robot touches the environment
        if p.getContactPoints(self.cricketUid,self.planeUid,linkIndexA=-1): # not empty
            for id in limb_ids:
                no_track_sum += self.__joint_penalty(id)**2 # beta_i(q_{t-1}^i)^2
        # reward/penalty based on the direction of the last rotation
        for c, p_action in enumerate(self.previous_actions):
            # kappa_i|\tau_{t-1}^i|
            if (action[c] <=0 and p_action <=0) or (action[c] > 0 and p_action > 0) :
                # if the new action is following the old one --> reward
                no_track_sum -= abs(p_action)
            else :
                # else penalty
                no_track_sum += abs(p_action)
        # Difference with the joints final position
        _, limb_pos = self.cricket.get_joint_positions()
        diff = [(abs(limb) - abs(goal_))**2 for limb, goal_ in zip(limb_pos,self.goal.get_final_joints())]
        no_track_sum += self.w_joints * sum(diff) # w_q^i(\Delta q_t^i)^2

        reward -= no_track_sum

        # The error ε represent the difference between the predicted Q(s,a) and the measured Q(s,a)
        depth = 5
        reward -= self.w_error * sum([self.disc_factor*err for err in self.pred_v_measured[:depth]])**2

        # time passed
        reward -= self.w_t * self.episode_step

        # difference with the robot's final center of mass position and body rotation
        pos,angs, _, _ = self.cricket.get_observations()
        f_pos, f_ang = self.goal.get_final_observation()
        # center of mass
        reward -= self.w_X * (abs(f_pos[0]) - abs(pos[0]))**2
        reward -= self.w_Y * (abs(f_pos[1]) - abs(pos[1]))**2
        reward -= self.w_Z * (abs(f_pos[2]) - abs(pos[2]))**2
        # tortion
        reward -= self.w_psi   * (abs(f_ang[0]) - abs(angs[0]))**2
        reward -= self.w_theta * (abs(f_ang[1]) - abs(angs[1]))**2
        reward -= self.w_sigma * (abs(f_ang[2]) - abs(angs[2]))**2

    def __joint_penalty(self, id):
        is_continous = p.getJointInfo(self.cricketUid,id)[8] == 0.0
        val = p.getJointState(self.cricketUid,id)[0]
        # normalization 
        if is_continous:
            if val > math.pi:
                res = val%math.pi - math.pi
            elif val < math.pi:
                res = val%math.pi + math.pi
            else :
                res = val
        else:
            res = val
        return res


        p.getContactPoints(robot_id, planeId, linkIndexA=-1)
    def reset(self):
        ''' This function is used to reset the PyBullet environment '''
        self.step_counter = 0
        p.resetSimulation()     # reset PyBullet environment
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-10)

        # Plane: to chose the plane I can do a script that changes the path to the desired 
        self.planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

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
        self.previous_actions = np.zeros(shape=(,))
        self.episode_step = 0
        self.pred_v_measured = []

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