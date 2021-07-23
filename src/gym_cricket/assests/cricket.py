from numpy.core.defchararray import join
import pybullet as p
import numpy as np
import os

class Cricket:
    def __init__(self, client, strating_position=None, joint_speed = 0, c_rolling = 0.2, c_drag = 0.01, c_throttle = 20) -> None:
        """
        Input:
        - clinet : costant
        PyBullet phisic client
        - starting_position : np.array
        starting position for the joints


        variables: 
        self.cricket: 
            cricket_id from URDF file
        self.track_joints and self.track_ids:
            joints related to the tracks --> list of lists [[wheel1,wheel2],[..],...]
        self.limb_joints,self.limb_ids:
            all the other non-fixed joints (knees, shoulders, and so on)
        """
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'urdfs/cricket_robot.urdf')
        self.cricket = p.loadURDF(fileName = f_name,
                                  basePosition = [0,0,0.5],
                                  physicsClientId=client
                                  )
                                  #flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.track_joints,self.track_ids,\
            self.limb_joints,self.limb_ids,\
                self.fixed_joints, self.fixed_ids,\
                    self.cont_ids, self.revo_ids = self.__find_joints()   # completes the above joint lists 
        
        num_wheel = 8
        num_tracks = len(self.track_joints)
        num_joints = len(self.limb_ids)
        
        #Starting positions
        if strating_position == None:
            self.track_positions = np.zeros((num_tracks,))
            self.limb_positions = np.zeros((num_joints))
        else :
            self.track_positions = np.array(strating_position[:num_wheel])
            self.limb_positions = np.array(strating_position[num_wheel:])

        # Starting velocities 0
        self.track_velocities = np.zeros(num_wheel)
        self.limb_velocities = np.zeros(len(self.limb_positions))

        # Max and Min linear velocity
        self.max_lvel = np.array([100,100,100])
        self.min_lvel = np.array([0,0,0])
        # Max and Min angular velocity
        self.max_avel = np.array([100,100,100])
        self.min_avel = np.array([0,0,0])

        # Normal forces
        self.n_normal_f = 4

        # Joint speed
        self.joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the car
        self.c_throttle = 20

        # Store all the collision between links and their fixed accessories (which are considered collision by PyBullet)
        self.collision_safe = {}
        for i in self.limb_ids:
            aa, bb = p.getAABB(self.cricket,i,self.client) # return the bounding box of the body (-1) starting from the center of mass
            obs = p.getOverlappingObjects(aa, bb, self.client)
            self.collision_safe.update({i : sorted(obs)})

    def perform_action(self, action):
        '''
        The action passed to the robot are the angles of the joints and the rotation of
        the wheel tracks

        [(angle,velocity), (angle,velocity), ...]

        Domanda: should I need to add the acceleration of the tortion as well?

        action:non voglio essere nell'intreccio delle loro vite wathcman dr manhattan
            0-3 the value of the tracks movement 
            4... the tortion of all the other joints
        '''
        angles = action[:,0]
        velocities = action [:,1]
        """
        SOLUZZIONE PROPOSTA:
        Si performa un'azione per uno specifico time stemp (like frame to frame),
        Quando il tempo termina, si valutano i risultati (reward, reti and so on), e si genera una nuova azione
        """
        # Tracks movement
        wheel_angles = [val for val in action[:4,0] for _ in (0,1)] # take all the first values of the first 4 pairs two time --> e.g. [[1,2],[3,4],[5,6],[7,8]] >> became >> [1,1,3,3,5,5,7,7]
        wheel_velocities = [val for val in action[:4,1] for _ in (0,1)]
        self.track_positions = np.add(self.track_positions, wheel_angles)
        self.track_velocities = np.add(self.track_velocities, wheel_velocities)
        p.setJointMotorControlArray(
            self.cricket,
            jointIndices = self.track_ids,
            controlMode = p.POSITION_CONTROL,
            targetPositions = self.track_positions,
            targetVelocities = self.track_velocities ,
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

    def get_observations(self):
        '''Return:
            - position (x,y,z)
            - orientation (roll around X, pitch around Y, yaw around Z)
            - velocity (two 3 values vetors: linear velocity [x,y,z], angular velocity [ωx,ωy,ωz])
           of the robot in the simulation'''
        pos, ang = p.getBasePositionAndOrientation(self.cricket, self.client)

        # convert the quaternion (δ,Θ,ψ,ω) to Euler (δ,Θ,ψ)
        angs = p.getEulerFromQuaternion(ang) # roll, pitch, yaw

        # Get the velocity of the robot
        l_vel, a_vel = p.getBaseVelocity(self.cricket, self.client) # linear & angular velocity

        return (pos,angs,l_vel,a_vel)

    def get_ids(self):
        '''Return the basic PyBullet information of the robot'''
        return self.cricket, self.client
    
    def get_joint_ids(self):
        '''Return the joints indexes (track, limb, fixed)'''
        return self.track_ids,\
            self.limb_ids, self.fixed_ids

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


    def __find_joints(self):
        '''Completes the joints in the cricket robot'''
        number_of_joints = p.getNumJoints(self.cricket)
        track = []
        track_joints, limb_joints, fixed_joints = [], [], []
        track_ids, limb_ids, fixed_ids = [], [], []
        cont_ids, revo_ids = [], []
        for joint_number in range(number_of_joints):
            joint_info = p.getJointInfo(self.cricket, joint_number)
            # [jointIndex, jointName (bytes), jointType, ...]
            if joint_info[2] == 0 and "track" in joint_info[1].decode("utf-8"): # if the joint is revolute/continous and contains the word "track"
                # the joint is one of the wheels
                track.append(joint_info)
                track_ids.append(joint_info[0])
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
        return np.array(track_joints), np.array(track_ids),\
                np.array(limb_joints), np.array(limb_ids),\
                 np.array(fixed_joints), np.array(fixed_ids),\
                  np.array(cont_ids),np.array(revo_ids)
    
    def get_joint_positions(self):
        """
         - getJointState(robot_id, jointIndex)
        [*jointPosition*, jointVelocity, jointReactionForces, appliedJointMotorTorque]
        """
        track_pos = [p.getJointState(self.cricket, wheel[0])[0] # wheel[0] gets the jointIndex
            for track in self.track_joints for wheel in track]
        track_pos = self.__normalize(track_pos)

        limb_pos = [p.getJointState(self.cricket, joint[0])[0]
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
    
    def get_joint_velocities(self):
        """
         - getJointState(robot_id, jointIndex)
        [jointPosition, *jointVelocity*, jointReactionForces, appliedJointMotorTorque]
        """
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

    def get_normal_forces(self, planeId : str):
        """Get all the normal forces between the robot and palneId
        
        If it's not possible to obtain 4 normal forces an array full of zeros will complete it
        if there are more than 4 normal forces per track, it will choose randomly

        Return:
         - a list of 4 lists (one per track) of normal forces, for each list:
            4 normal forces along the wheels and the track between the wheels
        """
        for wheel_1,wheel_2 in self.track_ids:
            track_id = wheel_1 + 1
            contact_points = [c_point[9] for c_point in p.getContactPoints(self.cricket, planeId, linkIndexA=wheel_1)]
            contact_points += [c_point[9] for c_point in p.getContactPoints(self.cricket, planeId, linkIndexA=wheel_2)]
            contact_points += [c_point[9] for c_point in p.getContactPoints(self.cricket, planeId, linkIndexA=track_id)]

            if len(contact_points) < self.n_normal_f:
                contact_points += [0] * (self.n_normal_f - len(contact_points))
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
        high_lim, low_lim = np.full((4,), np.pi), np.full((4,),-np.full)
        return high_lim,low_lim

    def get_normal_forces_limits(self,gravity):
        mass = 10000 # grams --> there's a way to get the mass
        max_nf, min_nf = np.full((self.n_normal_f,),mass*gravity), np.zeros((self.n_normal_f))
        return max_nf,min_nf

    def get_action_limits(self):
        dim = len(self.limb_ids) + len(self.track_ids)
        high_lim = np.full((dim,), np.pi)
        low_lim = -high_lim
        return high_lim,low_lim


"""
USEFUL DOC:

setJointMotorControlArray:
[
    required - bodyUniqueId - int body unique id as returned from loadURDF etc.
    required - jointIndices - list of int index in range [0..getNumJoints(bodyUniqueId) (note that link index == joint index)
    required - controlMode - int POSITION_CONTROL, VELOCITY_CONTROL,
                                    TORQUE_CONTROL, PD_CONTROL. (There is also
                                    experimental STABLE_PD_CONTROL for stable(implicit) PD
                                    control, which requires additional preparation. See
                                    humanoidMotionCapture.py and pybullet_envs.deep_mimc for
                                    STABLE_PD_CONTROL examples.)
    optional - targetPositions - list of float in POSITION_CONTROL the targetValue is target position of
                                    the joint
    optional - targetVelocities - list of float in PD_CONTROL, VELOCITY_CONTROL and
                                    POSITION_CONTROL the targetValue is target velocity of the
                                    joint, see implementation note below.
    optional - forces - list of float in PD_CONTROL, POSITION_CONTROL and
                            VELOCITY_CONTROL this is the maximum motor force used to
                            reach the target value. In TORQUE_CONTROL this is the
                            force/torque to be applied each simulation step.
    optional - positionGains - list of float See implementation note below
    optional - velocityGains - list of float See implementation note below
    optional - physicsClientId - int if you are connected to multiple servers, you can pick one. 
]
______________________________________________________________________________________________________

-------------------------------------------------------------------------
method          | implementation    | component                         |
-------------------------------------------------------------------------
POSITION_CONTROL| constraint        | velocity and position constraint  | error = position_gain*(desired_position-actual_position)+velocity_gain*(desired_velocity-actual_velocity)
VELOCITY_CONTROL| constraint        | pure velocity constraint          | error = desired_velocity - actual_velocity
"""