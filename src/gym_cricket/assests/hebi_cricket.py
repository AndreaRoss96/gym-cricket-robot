from numpy.core import overrides
from cricket_abs import Cricket_abs
from numpy.core.defchararray import join
import pybullet as p
import numpy as np
import os

class HebiCricket(Cricket_abs):
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
        f_name = os.path.join(os.path.dirname(__file__), f_path)
        self.cricket = p.loadURDF(fileName = f_name,
                                  basePosition = base_position,
                                  physicsClientId=self.client
                                  )
                                  #flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.track_joints,self.wheel_ids,\
            self.limb_joints,self.limb_ids,\
                self.fixed_joints, self.fixed_ids = self.__find_joints()   # completes the above joint lists 

        self.num_wheel = len(self.wheel_ids) # 4
        num_joints = len(self.limb_ids)

        # Starting positions
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

    def  __find_joints(self):
        '''Completes the joints in the cricket robot'''
        number_of_joints = p.getNumJoints(self.cricket, physicsClientId=self.client)
        track_joints, limb_joints, fixed_joints = [], [], []
        wheel_ids, limb_ids, fixed_ids = [], [], []
        
        for joint_number in range(number_of_joints):
            joint_info = p.getJointInfo(self.cricket, joint_number, physicsClientId=self.client)
            if joint_info[2] == 0 and "pulley" in joint_info[1].decode("utf-8"):
                # if the joint is revolute/continous and contains the word "pulley"
                wheel_ids.append(joint_info[0])
                track_joints.append(joint_info)
            elif joint_info[2] == 0:
                # the joint is continous
                limb_ids.append(joint_info[0])
                limb_joints.append(joint_info)
            else:
                fixed_ids.append(joint_info[0])
                fixed_joints.append(joint_info)
        return np.array(track_joints, dtype=object), np.array(wheel_ids, dtype=object),\
                np.array(limb_joints, dtype=object), np.array(limb_ids, dtype=object),\
                 np.array(fixed_joints, dtype=object), np.array(fixed_ids, dtype=object)

    @overrides
    def perform_action(self, action):
        half = int(len(action)/2)
        action = np.array([[a,b] for a,b in zip(action[:half],action[half:])])
        # action -> [(angle1, speed1), (angle2, speed2), ..., (anglen,speedn)]
        # Tracks movement
        wheel_angles = [val for val in action[:self.num_wheel,0]]
        wheel_velocities = [val for val in action[:self.num_wheel,1]]
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
        limb_angles = [val for val in action[self.num_wheel:,0]]
        limb_velocites = [val for val in action[self.num_wheel:,1]]
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

    @overrides
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
        return track_pos, limb_pos
    
    def __normalize(self, position_list):
        '''Helps get_joint_position() to normalize the returned values'''
        res = []
        for position in position_list:
            if position % (2*np.pi) > np.pi or position == (2*np.pi) :
                res.append(position%np.pi-np.pi )
            else :
                res.append(position%np.pi)
        return res

    @overrides
    def get_normal_forces_limits(self, gravity):
        mass = 22.569 * 100 # from kilos to grams
        max_nf, min_nf = np.full((self.n_normal_f,),mass*gravity), np.zeros((self.n_normal_f))
        return max_nf,min_nf
        
    @overrides
    def get_normal_forces(self, planeId : str):
        """Get all the normal forces between the robot and palneId
        
        If it's not possible to obtain 4 normal forces an array full of zeros will complete it
        if there are more than 4 normal forces per track, it will choose randomly

        Return:
         - a list of 4 lists (one per track) of normal forces, for each list:
            4 normal forces along the wheels and the track between the wheels
        """
        contact_points = []
        for wheel_id in self.wheel_ids:
            contact_points += [c_point[9] for c_point in p.getContactPoints(self.cricket, planeId, linkIndexA=wheel_id)]
        
        for limb_joint in self.limb_joints:
            if "track" in limb_joint[1]:
                contact_points += [c_point[9] for c_point in p.getContactPoints(self.cricket, planeId, linkIndexA=limb_joint[0])]

        if len(contact_points) < self.n_normal_f :
            contact_points += [0.0] * (self.n_normal_f - len(contact_points))
        elif len(contact_points) > self.n_normal_f :
            contact_points = sorted(contact_points, reverse=True)[:self.n_normal_f]

        return contact_points

