from numpy.core.defchararray import join
import pybullet as p
import numpy as np
import os
from gym_cricket.assests.cricket_abs import Cricket_abs

class Cricket(Cricket_abs):
    def __init__(self, client, strating_position=[], base_position = [0,0,0.5], normal_forces = 4) -> None:
        """
        Input:
        - clinet : costant
        PyBullet phisic client
        - starting_position : np.array
        starting position for the joints
        - base_position :
         [x,y,z] position of the robot at the beginning of the simulation


        variables: 
        self.cricket: 
            cricket_id from URDF file
        self.track_joints and self.wheel_ids:
            joints related to the tracks --> list of lists [[wheel1,wheel2],[..],...]
        self.limb_joints,self.limb_ids:
            all the other non-fixed joints (knees, shoulders, and so on)
        """
        self.client = client
        f_path='urdfs/cricket_robot.urdf'
        f_name = os.path.join(os.path.dirname(__file__), f_path)
        self.cricket = p.loadURDF(fileName = f_name,
                                  basePosition = base_position,
                                  physicsClientId=self.client
                                  )
                                  #flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.track_joints,self.wheel_ids,\
            self.limb_joints,self.limb_ids,\
                self.fixed_joints, self.fixed_ids = self.__find_joints()   # completes the above joint lists 

        self.num_wheel = 8
        num_tracks = len(self.track_joints)
        num_joints = len(self.limb_ids)
        
        #Starting positions
        if strating_position == []:
            self.wheel_position = np.zeros((self.num_wheel,))
            self.limb_positions = np.zeros((num_joints))
        else :
            self.wheel_position = np.array(strating_position[:self.num_wheel])
            self.limb_positions = np.array(strating_position[self.num_wheel:])

        for joint_id, joint_pos in zip(self.limb_ids, self.limb_positions):
            p.resetJointState(
                self.cricket,
                joint_id,
                joint_pos,
                0,
                self.client
            )

        self.mass = 100
        # Starting velocities 0
        self.track_velocities = np.zeros(self.num_wheel)
        self.limb_velocities = np.zeros(len(self.limb_positions))

        # Max and Min linear velocity - vx,vy,vz
        velocity = 100
        self.max_lvel = np.array([velocity,velocity,velocity])
        self.min_lvel = np.array([0,0,0])
        # Max and Min angular velocity
        self.max_avel = np.array([velocity,velocity,velocity])
        self.min_avel = np.array([0,0,0])

        # Normal forces
        self.n_normal_f = normal_forces

        # Store all the collision between links and their fixed accessories (which are considered collision by PyBullet)
        self.collision_safe = {}
        for i in self.limb_ids:
            aa, bb = p.getAABB(self.cricket,i,self.client) # return the bounding box of the body (-1) starting from the center of mass
            obs = p.getOverlappingObjects(aa, bb, self.client)
            self.collision_safe.update({i : sorted(obs)})
    
    # def set_joint_position(self, positions):
    #     for i in range(p.getNumJoints(self.cricket)):
    #         p.resetJointState(self.cricket, i, positions[i], physicsClientId = self.client)


    def perform_action(self, action):
        '''
        The action passed to the robot are the angles of the joints and the rotation of
        the wheel tracks

        [angle1, angle2, ..., anglen, speed1, speed2, ..., anglen]

        action:
            0-3 the value of the tracks movement 
            4... the tortion of all the other joints
        '''
        half = int(len(action)/2)
        action = np.array([[a,b] for a,b in zip(action[:half],action[half:])])
        # action -> [(angle1, speed1), (angle2, speed2), ..., (anglen,speedn)]
        # Tracks movement
        wheel_angles = [val for val in action[:4,0] for _ in (0,1)] # take all the first values of the first 4 pairs two time --> e.g. [[1,2],[3,4],[5,6],[7,8]] >> became >> [1,1,3,3,5,5,7,7]
        wheel_velocities = [val for val in action[:4,1] for _ in (0,1)]
        self.wheel_position = np.add(self.wheel_position, wheel_angles)
        self.track_velocities = np.add(self.track_velocities, wheel_velocities)
        p.setJointMotorControlArray(
            self.cricket,
            jointIndices = self.wheel_ids,
            controlMode = p.POSITION_CONTROL,
            targetPositions = self.wheel_position,
            targetVelocities = self.track_velocities,
            physicsClientId = self.client
            )
        
        # Limbs movement 
        limb_angles = [val for val in action[4:,0]]
        limb_velocites = [val for val in action[4:,1]]
        self.limb_positions = np.add(self.limb_positions, limb_angles)
        self.limb_velocities = np.add(self.limb_velocities, limb_velocites)
        p.setJointMotorControlArray(
            self.cricket,
            jointIndices = self.limb_ids,
            controlMode = p.POSITION_CONTROL,
            targetPositions = self.limb_positions,
            targetVelocities = self.limb_velocities,
            physicsClientId = self.client
            )

    # def get_observations(self):
    #     '''Return:
    #         - position (x,y,z)
    #         - orientation (roll around X, pitch around Y, yaw around Z)
    #         - velocity (two 3 values vetors: linear velocity [x,y,z], angular velocity [ωx,ωy,ωz])
    #        of the robot in the simulation'''
    #     pos, ang = p.getBasePositionAndOrientation(self.cricket, self.client)

    #     # convert the quaternion (δ,Θ,ψ,ω) to Euler (δ,Θ,ψ)
    #     angs = p.getEulerFromQuaternion(ang, physicsClientId=self.client) # roll, pitch, yaw

    #     # Get the velocity of the robot
    #     l_vel, a_vel = p.getBaseVelocity(self.cricket, self.client) # linear & angular velocity

    #     return pos,angs,l_vel,a_vel

    # def get_ids(self):
    #     '''Return the basic PyBullet information of the robot
        
    #     Return:
    #      - ID
    #      - Client
    #     '''
    #     return self.cricket, self.client
    
    # def get_joint_ids(self):
    #     '''Return the joints indexes:
    #      - track
    #      - limb
    #      - fixed
    #     '''
    #     return self.wheel_ids,\
    #         self.limb_ids, self.fixed_ids

    def __find_joints(self):
        '''Completes the joints in the cricket robot'''
        number_of_joints = p.getNumJoints(self.cricket, physicsClientId=self.client)
        track = []
        track_joints, limb_joints, fixed_joints = [], [], []
        wheel_ids, limb_ids, fixed_ids = [], [], []
        for joint_number in range(number_of_joints):
            joint_info = p.getJointInfo(self.cricket, joint_number, physicsClientId=self.client)
            # [jointIndex, jointName (bytes), jointType, ...]
            if joint_info[2] == 0 and "track" in joint_info[1].decode("utf-8"): # if the joint is revolute/continous and contains the word "track"
                # the joint is one of the wheels
                track.append(joint_info)
                wheel_ids.append(joint_info[0])
                if len(track) == 2:
                    track_joints.append(track)
                    track = []
            elif joint_info[2] == 0:
                # the joint is revolute/continuos
                limb_ids.append(joint_info[0])
                limb_joints.append(joint_info)
            else :
                fixed_ids.append(joint_info[0])
                fixed_joints.append(joint_info)
        return np.array(track_joints, dtype=object), np.array(wheel_ids, dtype=object),\
                np.array(limb_joints, dtype=object), np.array(limb_ids, dtype=object),\
                 np.array(fixed_joints, dtype=object), np.array(fixed_ids, dtype=object)
    
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
    
    def __normalize(self, position_list):
        '''Helps get_joint_position to normalize the returned values'''
        res = []
        for position in position_list:
            if position % (2*np.pi) > np.pi or position == (2*np.pi) :
                res.append(position%np.pi-np.pi )
            else :
                res.append(position%np.pi)
        return res
    
    # def get_joint_velocities(self):
    #     """
    #      - getJointState(robot_id, jointIndex)
    #     [jointPosition, *jointVelocity*, jointReactionForces, appliedJointMotorTorque]
    #     """
    #     track_vel = [p.getJointState(self.cricket, wheel[0])[1]
    #         for track in self.track_joints for wheel in track]
    #     limb_vel = [p.getJointState(self.cricket, joint[0])[1]
    #         for limb in self.limb_joints for joint in limb]
    #     return track_vel, limb_vel

    # def get_joint_collisions(self):
    #     collision = {}
    #     for id in self.limb_ids:
    #         aa, bb = p.getAABB(self.cricket,id,self.client) # return the bounding box of the body (-1) starting from the center of mass
    #         obs = p.getOverlappingObjects(aa, bb, self.client)
    #         safe_obs = self.collision_safe.get(id)
    #         if sorted(obs) not in safe_obs :
    #             collision.update({id : list(set(obs) - set(safe_obs))})
    #     return collision

    def get_normal_forces(self, planeId : str):
        """Get all the normal forces between the robot and palneId
        
        If it's not possible to obtain 4 normal forces an array full of zeros will complete it
        if there are more than 4 normal forces per track, it will choose randomly

        Return:
         - a list of 4 lists (one per track) of normal forces, for each list:
            4 normal forces along the wheels and the track between the wheels
        """
        contact_points = []
        for count, wheel_id in enumerate(self.wheel_ids):
            contact_points += [c_point[9] for c_point in p.getContactPoints(self.cricket, planeId, linkIndexA=wheel_id)]

            if count + 1 % 2 != 0 : # update trackId once each two iterations
                track_id = wheel_id + 1
                contact_points += [c_point[9] for c_point in p.getContactPoints(self.cricket, planeId, linkIndexA=track_id)]

        if len(contact_points) < self.n_normal_f :
            contact_points += [0.0] * (self.n_normal_f - len(contact_points))
        elif len(contact_points) > self.n_normal_f :
            contact_points = sorted(contact_points, reverse=True)[:self.n_normal_f]

        return contact_points

    def get_joint_limits(self):
        high_lim, low_lim = [] ,[]
        for joint in self.limb_joints:
            if joint[8] == 0 : # continous joint
                high_lim.append(np.pi)
                low_lim.append(-np.pi)
            else:
                high_lim.append(np.pi/2)
                low_lim.append(-np.pi/2)
        return np.array(high_lim), np.array(low_lim)

    def get_track_limits(self):
        high_lim, low_lim = np.full((self.num_wheel,), np.pi), np.full((self.num_wheel,),-np.pi)
        return high_lim,low_lim

    # def get_normal_forces_limits(self,gravity):
    #     mass = 10000 # grams --> there's a way to get the mass
    #     max_nf, min_nf = np.full((self.n_normal_f,),mass*gravity), np.zeros((self.n_normal_f))
    #     return max_nf,min_nf

    # def get_action_limits(self):
    #     dim = len(self.limb_ids) + len(self.track_joints)
    #     high_lim = np.full((dim,), np.pi)
    #     low_lim = -high_lim
    #     return high_lim,low_lim

    def get_action_velocities_limits(self):
        dim = len(self.limb_ids) + len(self.track_joints)
        high_lim = np.full((dim,), 100)
        low_lim = np.zeros((dim,))
        return high_lim,low_lim
    
    # def print_info(self):
    #     '''
    #     Prints the main robot characteristics 
    #     '''
    #     times = 40
    #     print('='*times)
    #     print(f'ID: {self.cricket}')
    #     print(f'clinet: {self.client}')
    #     print(f'Body mass: {p.getDynamicsInfo(self.cricket,-1,self.client)}')
    #     print('tracks: ')
    #     for count, track in enumerate(self.track_joints):
    #         print(f'{count} - {track}')
    #         print('-'*times)
    #     print('_'*times)
    #     print('limb joints: ')
    #     for count, joint in enumerate(self.limb_joints):
    #         print(f'{count} - {joint}')
    #         print('-'*times)
    #     print('_'*times)
    #     print('fixed joints: ')
    #     for count, joint in enumerate(self.fixed_joints):
    #         print(f'{count} - {joint}')
    #         print('-'*times)
    #     print('_'*times)
    #     print('Link states')
    #     print('='*times)

