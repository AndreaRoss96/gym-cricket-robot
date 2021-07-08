import pybullet as p
import os
import math
'''
TODO:
 - finisci la funzione perform_action
'''
class Cricket:
    def __init__(self, client) -> None:
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'urdfs/cricket_robot.urdf')
        self.cricket = p.loadURDF(fileName = f_name,
                                  basePosition = [0,0,0.1],
                                  physicsClientId=client)
        self.track_joints = [] # joints related to the tracks
        self.limb_joints = [] # all the other non-fixed joints (knees, shoulders, and so on)
        self.__find_joints() # completes the above joint lists 
        # Joint speed
        self.joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the car
        self.c_throttle = 20

    def __find_joints(self):
        number_of_joints = p.getNumJoints(self.cricket)
        track = []
        for joint_number in range(number_of_joints):
            joint_info = p.getJointInfo(self.cricket, joint_number)
            # [jointIndex, jointName (bytes), jointType, ...]
            if joint_info[2] == 0 and "track" in joint_info[1].decode("utf-8"): # if the joint is revolute/continous and contains the word "track"
                # the joint is one of the wheels
                track.append(joint_info)
                if len(track) == 2:
                    self.track_joints.append(track)
                    track = []
            elif joint_info[2] == 0:
                # the joint is revolute/continuos
                self.limb_joints.append(joint_info)
    
    def perform_action(self, action):
        '''
        The action passed to the robot are the angles of the joints and the rotation of
        the wheel tracks
        '''

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
        for count, joint in enumerate(self.limb_joints):
            print(f'{count} - {joint}')
            print('-'*times)
        print('='*times)
