import numpy as np
import pybullet as p
import time

from gym_cricket.assests.cricket import Cricket

class CricketGoal(Cricket):
    def __init__(self, joint_position, planeId, client= None, base_position = [0,0,0.5]) -> None:
        if client == None :
            client = p.connect(p.DIRECT)
        super().__init__(client=client, strating_position=joint_position, base_position=base_position)
        
        if not isinstance(planeId,int):
            raise ValueError("PlaneId needs to be an integer")
        
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