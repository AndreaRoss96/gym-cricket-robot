from numpy.core.fromnumeric import std
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data
import math
import numpy as np
from gym_cricket.assests.cricket import Cricket
from gym_cricket.assests.cricketGoal import CricketGoal
from gym_cricket.assests.hebi_cricket import HebiCricket
from gym_cricket.assests.hebi_cricketGoal import HebiCricketGoal
"""
https://github.com/openai/gym/blob/master/docs/creating-environments.md

Repositories that use RL, OpenAI, and PyBullet
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
class CricketEnv(gym.Env):
    '''
    Gym environment for cricket robot (and similar)
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, plane_path = "plane.urdf", cricket_model = "basic_cricket", is_connected = False):
        self.n_loss = 5
        self.gravity = -9.81
        # self.goal = None

        if not is_connected :
            self.__connect()
        
        # adjust the view angle of the environment
        p.resetDebugVisualizerCamera(
            cameraDistance=7,
            cameraYaw=0,
            cameraPitch=-40,
            cameraTargetPosition=[0.,-0.,0.])

        self.plane_path = plane_path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeUid = p.loadURDF(self.plane_path)
                      
        # Plane: to chose the plane I can do a script that changes the path to the desired 
        self.cricket_model = cricket_model
        std_path = r'src/gym_cricket/assests/urdfs/'
        final_path = std_path + self.cricket_model + '.urdf'
        if cricket_model == 'hebi_cricket' :
            self.cricket = HebiCricket(self.client)
        elif cricket_model == 'basic_cricket' :
            self.cricket = Cricket(self.client)
        else :
            raise ValueError('Check the name of the cricket model')
        
        # Defining OBSERVATION space A.K.A. NN inputs
        ## position X,Y,Z
        high_pos = np.inf * np.ones(3)
        low_pos = -high_pos
        ## orientation ??, ??, ??
        high_ang = np.pi * np.ones(3)
        low_ang = - high_ang
        ## velocity linear and angular
        high_vel = np.concatenate((self.cricket.max_lvel,self.cricket.max_avel))
        low_vel = np.concatenate((self.cricket.min_lvel,self.cricket.min_avel))
        ## leg joints
        high_leg, low_leg = self.cricket.get_joint_limits()
        ## tracks
        high_track, low_track = self.cricket.get_track_limits()
        ## normal forces
        high_nf,low_nf = self.cricket.get_normal_forces_limits(self.gravity)
        ## Computed Loss
        high_loss, low_loss = np.full((self.n_loss,), np.inf), np.zeros((self.n_loss,))
        # Let's describe the format of valid actions and observations.
        self.observation_space = spaces.Box(
            low=np.array(np.concatenate((low_pos,low_ang,low_vel,low_leg,low_track,low_nf,low_loss)), dtype=np.float32),
            high=np.array(np.concatenate((high_pos,high_ang,high_vel,high_leg,high_track,high_nf,high_loss)), dtype=np.float32))
        
        # Defining ACTION space A.K.A. NN outputs
        high_lim, low_lim = self.cricket.get_action_limits()
        high_vel_lim, low_vel_lim = self.cricket.get_action_velocities_limits()
        low = np.array(np.concatenate((low_lim, low_vel_lim)), dtype=np.float32)
        high = np.array(np.concatenate((high_lim, high_vel_lim)), dtype=np.float32)
        self.action_space = spaces.Box(
            low=low,
            high=high)
        self.np_random, _ = gym.utils.seeding.np_random()

        dim = len(self.cricket.get_action_limits())
        self.previous_actions = np.zeros(shape=(dim,))
        self.episode_step = 0
        self.loss = np.zeros((self.n_loss,), dtype=np.float32)
        self.previous_reward = - np.inf
        self.early_stop = 0

        # self.reset()
    
    def __connect(self):
        self.client = p.connect(p.GUI) # p.connect(p.DIRECT) # connect to PyBullet using GUI

    def set_reward_values(self,w_joints = None,w_error = 100,
                          disc_factor = 0.99, w_t = 0.0625,w_X = 0.5,w_Y = 0.5,
                          w_Z = 0.5,w_psi = 0.5,w_theta = 0.5,
                          w_sigma = 0.5, early_stop_limit = 250):
        """Set the costants used in the reward function"""
        if w_joints is not None :
            self.w_joints = w_joints
        else :
            _, limb_joints, _ = self.cricket.get_joint_ids()
            num_limb_joints = len(limb_joints)
            self.w_joints = np.full((num_limb_joints,), 1)
        self.w_error = w_error
        self.disc_factor = disc_factor
        self.w_t = w_t
        self.w_X = w_X
        self.w_Y = w_Y
        self.w_Z = w_Z
        self.w_psi = w_psi
        self.w_theta = w_theta
        self.w_sigma = w_sigma
        self.early_stop_limit = early_stop_limit

    def step(self, action):
        """Advance the Task by one step.

        Args:
            action (np.ndarray): The action to be executed.

        Returns: A tuple (reward, new_state, done, info)
         - reward   : float
         - new_state: [(x,y,z),(psi,theta,sigma),(vx,vy,vz),(vpsi,vtheta,vsigma),(f1,f2,f3,f4)]
         - done     : Boolean
         - info     : string
        """
        try :
            self.goal
        except AttributeError:
            raise AttributeError("Robot's goal not defined.\
                \n\nUse the function \"set_goal(goal)\"")

        self.episode_step += 1
        self.cricket.perform_action(action)
        p.stepSimulation(self.client)
        # new state
        state_keys = ["pos","angs","l_vel","a_vel","limb_pos",\
                        "track_pos","normal_forces","loss_state"]
        current_state = self.current_state()
        dict_state = dict(zip(state_keys,current_state))
        current_state = self.__unpack_state(current_state)

        # reward
        reward = self.__compute_reward(
            action,
            dict_state["pos"],
            dict_state["angs"],
            dict_state["limb_pos"])
        #done
        done = False
        info = ""
        self.previous_actions = action
        if reward < self.previous_reward :
            self.early_stop +=1
            if self.early_stop >= self.early_stop_limit :
                print("!"*80)
                print("early stop")
                done = True
                info = "reward doesn't grow"
        else :
            self.early_stop = 0
        self.previous_reward = reward

        return reward, current_state, done, info
    
    def __compute_reward(self,action,pos,angs,limb_pos):
        """Compute the reward based on the defined reward function"""
        # \mathcal{R}_t =-\sum_{i=0}^{no\_track}[\alpha_i(q_{t-1}^i)^2+\beta_i(q_{t-1}^i)^2+\kappa_i|\tau_{t-1}^i|+w_q^i(\Delta q_t^i)^2]-w_\varepsilon (\sum_{i=0}^5\gamma^i\varepsilon_{t-i})^2-w_tt-w_X(\Delta X)^2-w_Y(\Delta Y)^2-w_Z(\Delta Z)^2-w_\psi(\Delta \psi)^2-w_\theta(\Delta \theta)^2-w_\phi(\Delta \phi)^2
        reward = 0
        no_track_sum = 0
        _, limb_ids, _ = self.cricket.get_joint_ids()

        # Penalty if the robot touches itself
        self_collisions = self.cricket.get_joint_collisions()
        for id,coll in self_collisions.items() :
            index = np.where(limb_ids == id)
            no_track_sum += (limb_pos[index[0][0]])**2
            #no_track_sum += self.__joint_penalty(id)**2 # alpha_i(q_{t-1}^i)^2
        
        # Penalty if the robot touches the environment
        if p.getContactPoints(self.cricketUid,self.planeUid,linkIndexA=-1): # not empty
            for id in limb_ids:
                index = np.where(limb_ids == id)
                no_track_sum += (limb_pos[index[0][0]])**2
                #no_track_sum += self.__joint_penalty(id)**2 # beta_i(q_{t-1}^i)^2
        
        # reward/penalty based on the direction of the last rotation
        for c, p_action in enumerate(self.previous_actions):
            # kappa_i|\tau_{t-1}^i|
            if (action[c] <0 and p_action <0) or (action[c] > 0 and p_action > 0) or (p_action == 0):
                # if the new action is following the old one --> reward
                no_track_sum -= abs(p_action)
            else :
                # else penalty
                no_track_sum += abs(p_action)

        # Difference with the joints final position
        diff = [(abs(limb) - abs(f_limb))**2 for limb, f_limb in zip(limb_pos,self.goal.get_final_joints())]
        no_track_sum += np.dot(self.w_joints,diff) # w_q^i(\Delta q_t^i)^2

        reward -= no_track_sum

        # The error ?? represent the difference between the predicted Q(s,a) and the measured Q(s,a)
        reward -= self.w_error * sum([self.disc_factor * err for err in self.loss[-self.n_loss:]])**2

        # time passed
        reward -= self.w_t * self.episode_step

        # difference with the robot's final center of mass position and body rotation
        f_pos, f_ang = self.goal.get_final_observation()

        # center of mass
        reward -= self.w_X * (abs(f_pos[0]) - abs(pos[0]))**2
        reward -= self.w_Y * (abs(f_pos[1]) - abs(pos[1]))**2
        reward -= self.w_Z * (abs(f_pos[2]) - abs(pos[2]))**2
        
        # tortion
        reward -= self.w_psi   * (abs(f_ang[0]) - abs(angs[0]))**2
        reward -= self.w_theta * (abs(f_ang[1]) - abs(angs[1]))**2
        reward -= self.w_sigma * (abs(f_ang[2]) - abs(angs[2]))**2

        return reward
    
    def current_state(self):
        """Return the current state of the robot"""
        pos,angs,l_vel,a_vel = self.cricket.get_observations()
        track_pos, limb_pos = self.cricket.get_joint_positions()
        normal_forces = self.cricket.get_normal_forces(self.planeUid)
        loss_state = self.loss
        return [pos,angs,l_vel,a_vel,limb_pos,track_pos,normal_forces,loss_state]

    def reset(self):
        ''' This function is used to reset the PyBullet environment '''
        self.step_counter = 0

        p.resetSimulation(physicsClientId=self.client)     # reset PyBullet environment
        self.__init__(self.plane_path, self.cricket_model, is_connected=True)
        p.setGravity(0,0,self.gravity, physicsClientId=self.client)
        self.cricketUid, _ = self.cricket.get_ids()

        rest_poses = np.random.uniform(-math.pi,math.pi,p.getNumJoints(self.cricketUid))
        # here you can set the position of the joints (randomly is good)
        self.cricket.set_joint_position(rest_poses)

        return self.__unpack_state(self.current_state())

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]

        return rgb_array

    def close(self):
        p.disconnect()

    def seed(self, seed=None):
        '''
        Gym provides seeding utilities we can use to ensure different training and demonstration
        runs are identical. We obtain a randomly seeded numpy random number generator that we???ll
        use for all random operations.
        '''
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def set_goal(self, joint_position, base_position = None):
        if base_position == None :
            if self.cricket_model == 'hebi_cricket' :
                self.goal = HebiCricketGoal(joint_position, self.gravity, self.plane_path)
            elif self.cricket_model == 'basic_cricket' :
                self.goal = CricketGoal(joint_position, self.gravity, self.plane_path)
        else :
            if self.cricket_model == 'hebi_cricket' :
                self.goal = HebiCricketGoal(joint_position, self.gravity, self.plane_path, base_position = base_position)
            elif self.cricket_model == 'basic_cricket' :
                self.goal = CricketGoal(joint_position, self.gravity, self.plane_path, base_position = base_position)

    def push_loss(self, loss):
        loss = np.array(loss, dtype=np.float32)
        np.append(self.loss[-(self.n_loss-1):], loss) # append the loss to the list without exceding the limit number

    def __unpack_state(self, state):
        """
        Unpacking the elements from the format returned by cricket

        Input: 
            position, orientation,velocity,....
            (x, y, z), (psi,omega,mu), (vx,vy,vz), ...
        Return:
            [x,y,z,psi,omega,mu,vx,vy,vz, ...]
        """
        return np.array([elem for sub_obs in state for elem in [*sub_obs]], dtype=np.float32)