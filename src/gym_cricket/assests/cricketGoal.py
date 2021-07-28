import numpy as np
import pybullet as p

from gym_cricket.assests.cricket import Cricket

class CricketGoal(Cricket):
    def __init__(self, joint_position, planeId, client= None, base_position = [0,0,0.5]) -> None:
        """
        Define the optimal final position for the robot.

        Input:
         - Joint_position : np.array -> [0:4] tracks position, [4:] limb positions
                optimal position for each joint
         - planeId : int -> pyBullet uniwue Id for the robot
                Used to understand the normal forces
         - client
                Simulation client
        """
        if not isinstance(planeId,int):
            raise ValueError("PlaneId needs to be an integer")

        if client == None :
            client = p.connect(p.DIRECT)
        super().__init__(client=client, strating_position=joint_position, base_position=base_position)

        for _ in range(0,100):
            p.stepSimulation()

        _, limb_pos = self.get_joint_positions()
        self.final_limb = limb_pos

        f_pos,f_angs,_,_ = self.get_observations()
        self.final_pos = f_pos
        self.final_angs = f_angs
        p.disconnect()

    def get_final_joints(self):
        return self.final_limb

    def get_final_observation(self):
        return self.final_pos, self.final_angs