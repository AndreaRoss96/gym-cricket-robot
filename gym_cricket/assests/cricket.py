import pybullet as p
import os
import math
'''
TODO:
 - fai una funzione privata che cerca nel urdf quali sono i joint dedicati alle ruote
   e quali sono i joint dedicati ai movimenti delle braccia
   
 - finisci la funzione perform_action
'''
class Cricket:
    def __init__(self, client) -> None:
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'urdfs/cricket_robot.urdf')
        self.cricket = p.loadURDF(fileName = f_name,
                                  basePosition = [0,0,0.1],
                                  physicsClientId=client)
        self.track_joints = ['''joints related to the 4 tracks
                                they should move all in pair. I mean 
                                the front-left track has two wheels
                                and so on for the other tracks''']
        self.limb_joints = ['''all the other joints (shoulders, knees and so on)''']
        # Joint speed
        self.joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the car
        self.c_throttle = 20

    def get_ids(self):
        '''Return the PyBullet information of the robot'''
        return self.cricket, self.client
    
    def perform_action(self, action):
        pass

    def get_observations(self):
        '''Return:
            - position (x,y,z)
            - orientation (roll around X, pitch around Y, yaw around Z)
            - velocity (two 3 values vetors: linear velocity [x,y,z], angular velocity [ωx,ωy,ωz])
           of the robot in the simulation'''
        pos, ang = p.getBasePositionAndOrientation(self.cricket, self.client)
        ##pos = pos[:2] # position # -- useful for movement in 2D

        # convert the quaternion (δ,Θ,ψ,ω) to Euler (δ,Θ,ψ)
        angs = p.getEulerFromQuaternion(ang) # roll, pitch, yaw
        ##ori = (math.cos(angs[2]), math.sin(angs[2])) # orientation

        # Get the velocity of the robot
        l_vel, a_vel = p.getBaseVelocity(self.cricket, self.client) # linear & angular velocity

        return (pos,angs,l_vel,a_vel)
        
