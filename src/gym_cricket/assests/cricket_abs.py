from abc import ABC, abstractmethod 
import pybullet as p
import numpy as np
from numpy.core.defchararray import join

# pybullet doc:
#       https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3

class Cricket_abs(ABC):
    """
    Abstract class to define different type of cricket robots
    """

    @abstractmethod
    def perform_action(self, action):
        '''
        The action passed to the robot are the angles of the joints and the rotation of
        the wheel tracks

        [angle1, angle2, ..., anglen, speed1, speed2, ..., anglen]

        action:
            0-3 the value of the tracks movement 
            4... the tortion of all the other joints
        '''
        pass

    @abstractmethod
    def get_joint_limits(self):
        pass

    @abstractmethod
    def get_normal_forces_limits(self,gravity):
        pass

    @abstractmethod
    def get_normal_forces(self, planeId : str):
        pass

    @abstractmethod
    def get_action_velocities_limits(self):
        pass

    def set_joint_position(self, positions):
        for i in range(p.getNumJoints(self.cricket)):
            p.resetJointState(self.cricket, i, positions[i], physicsClientId = self.client)

    def get_observations(self):
        '''Return:
            - position (x,y,z)
            - orientation (roll around X, pitch around Y, yaw around Z)
            - velocity (two 3 values vetors: linear velocity [x,y,z], angular velocity [ωx,ωy,ωz])
           of the robot in the simulation'''
        pos, ang = p.getBasePositionAndOrientation(self.cricket, self.client)

        # convert the quaternion (δ,Θ,ψ,ω) to Euler (δ,Θ,ψ)
        angs = p.getEulerFromQuaternion(ang, physicsClientId=self.client) # roll, pitch, yaw

        # Get the velocity of the robot
        l_vel, a_vel = p.getBaseVelocity(self.cricket, self.client) # linear & angular velocity

        return pos,angs,l_vel,a_vel

    def get_ids(self):
        '''Return the basic PyBullet information of the robot
        
        Return:
         - ID
         - Client
        '''
        return self.cricket, self.client

    def get_joint_ids(self):
        '''Return the joints indexes:
         - track
         - limb
         - fixed
        '''
        return self.wheel_ids,\
            self.limb_ids, self.fixed_ids

    def get_joint_positions(self):
        """
         - getJointState(robot_id, jointIndex)
        [*jointPosition*, jointVelocity, jointReactionForces, appliedJointMotorTorque]
        """
        track_pos = [p.getJointState(self.cricket, wheel[0], physicsClientId=self.client)[0] # wheel[0] gets the jointIndex
            for track in self.track_joints for wheel in track]
        track_pos = self.__normalize(track_pos)

        limb_pos = [p.getJointState(self.cricket, joint[0], physicsClientId=self.client)[0]
            for joint in self.limb_joints]
        limb_pos = self.__normalize(limb_pos)
        # should I return two dictionaries? --> dict(zip(list1,list2))
        return track_pos, limb_pos
    
    def get_joint_velocities(self):
        track_vel = [p.getJointState(self.cricket, wheel[0])[1]
            for track in self.track_joints for wheel in track]
        limb_vel = [p.getJointState(self.cricket, joint[0])[1]
            for limb in self.limb_joints for joint in limb]
        return track_vel, limb_vel
    
    def get_joint_collisions(self):
        collision = {}
        for id in self.limb_ids:
            aa, bb = p.getAABB(self.cricket,id,self.client) # return the bounding box of the body (-1) starting from the center of mass
            obs = p.getOverlappingObjects(aa, bb, self.client)
            safe_obs = self.collision_safe.get(id)
            if sorted(obs) not in safe_obs :
                collision.update({id : list(set(obs) - set(safe_obs))})
        return collision

    def get_track_limits(self):
        high_lim, low_lim = np.full((self.num_wheel,), np.pi), np.full((self.num_wheel,),-np.pi)
        return high_lim,low_lim

    def get_action_limits(self):
        dim = len(self.limb_ids) + len(self.track_joints)
        high_lim = np.full((dim,), np.pi)
        low_lim = -high_lim
        return high_lim,low_lim

    def print_info(self):
        '''
        Prints the main robot characteristics 
        '''
        times = 40
        print('='*times)
        print(f'ID: {self.cricket}')
        print(f'clinet: {self.client}')
        print(f'Body mass: {p.getDynamicsInfo(self.cricket,-1,self.client)}')
        print('tracks: ')
        for count, track in enumerate(self.track_joints):
            print(f'{count} - {track}')
            print('-'*times)
        print('_'*times)
        print('limb joints: ')
        for count, joint in enumerate(self.limb_joints):
            print(f'{count} - {joint}')
            print('-'*times)
        print('_'*times)
        print('fixed joints: ')
        for count, joint in enumerate(self.fixed_joints):
            print(f'{count} - {joint}')
            print('-'*times)
        print('_'*times)
        print('Link states')
        print('='*times)
