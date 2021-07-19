import pybullet as p
import numpy as np
import os
import math
'''
TODO:
 - riesci a calcolare la forza normale sui tracks? oppure la calcolo in un altro file?
'''
class Cricket:
    def __init__(self, client, strating_position, joint_speed = 0, c_rolling = 0.2, c_drag = 0.01, c_throttle = 20) -> None:
        """
        Input:
        - clinet : costant
        PyBullet phisic client
        - starting_position : np.array
        starting position for the joints


        variables: 
        self.cricket: 
            cricket_id from URDF file
        self.track_joints and self.track_idxs:
            joints related to the tracks --> list of lists [[wheel1,wheel2],[..],...]
        self.limb_joints,self.limb_idxs:
            all the other non-fixed joints (knees, shoulders, and so on)
        """
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'urdfs/cricket_robot.urdf')
        self.cricket = p.loadURDF(fileName = f_name,
                                  basePosition = [0,0,0.1],
                                  physicsClientId=client)
        self.track_joints,self.track_idxs,self.limb_joints,self.limb_idxs = self.__find_joints()   # completes the above joint lists 
        
        #Starting positions
        self.track_positions = np.array(strating_position[:8])
        self.limb_positions = np.array(strating_position[8:])

        # Starting velocities 0
        self.track_velocities = np.zeros(8)
        self.limb_velocities = np.zeros(len(self.limb_positions))

        # Joint speed
        self.joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the car
        self.c_throttle = 20
    
    def perform_action(self, action):
        '''
        The action passed to the robot are the angles of the joints and the rotation of
        the wheel tracks

        [(angle,velocity), (angle,velocity), ...]

        Domanda: should I need to add the acceleration of the tortion as well?

        action:
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
            jointIndices = self.track_idxs,
            controlMode = p.POSITION_CONTROL,
            targetPositions = self.track_positions,
            targetVelocities = self.track_velocities ,
            physicsClientId = self.client
            )
        
        # Limbs movement 
        limb_angles = [val for val in action[4:,0]]
        limb_velocites = [val for val in action[:4,1]]
        self.limb_positions = np.add(self.limb_positions, limb_angles)
        self.limb_velocities = np.add(self.limb_velocities, limb_velocites)
        p.setJointMotorControlArray(
            self.cricket,
            jointIndices = self.limb_idxs,
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
        '''Return the PyBullet information of the robot'''
        return self.cricket, self.client

    def print_info(self):
        '''
        Prints the main robot characteristics 
        '''
        times = 40
        print('='*times)
        print(f'ID: {self.cricket}')
        print(f'clinet: {self.client}')
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
        print('Link states')
        print('='*times)


    def __find_joints(self):
        '''Completes the joints in the cricket robot'''
        number_of_joints = p.getNumJoints(self.cricket)
        track = []
        track_joints, limb_joints = [], []
        track_idxs, limb_idxs = [], []
        for joint_number in range(number_of_joints):
            joint_info = p.getJointInfo(self.cricket, joint_number)
            # [jointIndex, jointName (bytes), jointType, ...]
            if joint_info[2] == 0 and "track" in joint_info[1].decode("utf-8"): # if the joint is revolute/continous and contains the word "track"
                # the joint is one of the wheels
                track.append(joint_info)
                track_idxs.append(joint_info[0])
                if len(track) == 2:
                    track_joints.append(track)
                    track = []
            elif joint_info[2] == 0:
                # the joint is revolute/continuos
                limb_idxs.append(joint_info[0])
                limb_joints.append(joint_info)
        return track_joints, track_idxs, limb_joints, limb_idxs
    
    def get_joint_positions(self):
        """
         - getJointState(robot_id, jointIndex)
        [*jointPosition*, jointVelocity, jointReactionForces, appliedJointMotorTorque]
        """
        track_pos = [p.getJointState(self.cricket, wheel[0])[0] # wheel[0] gets the jointIndex
            for track in self.track_joints for wheel in track]
        limb_pos = [p.getJointState(self.cricket, joint[0])[0]
            for limb in self.limb_joints for joint in limb]
        return track_pos, limb_pos
    
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