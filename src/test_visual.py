import pybullet as p
import os
import time
import numpy as np
from gym_cricket.assests.cricket import Cricket
from gym_cricket.assests.cricketGoal import CricketGoal
import pybullet_data

cid = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81, physicsClientId=cid)
planeId = p.loadURDF("plane.urdf")

f_path = "/home/andrea/Desktop/HebiCricket/ros_packages/hebi_description/urdf/kits/cricket.urdf"
cricket = p.loadURDF(f_path)
#cricket = Cricket(cid, f_path=f_path)

wheels = [0.0] * 8
limbs = [0.0, -np.pi/2, np.pi, -np.pi/2, 0.0,\
np.pi/2, np.pi, np.pi/2, 0.0,\
-np.pi/2, np.pi, -np.pi/2, 0.0,\
np.pi/2, np.pi, np.pi/2, 0.0, 0.0]

goals = np.concatenate([wheels,limbs])


# cricket = CricketGoal(goals, -9.81, cid, base_position=[0,2,0.3])
# cricket_id, _ = cricket.get_ids()

# for goal, joint in zip(goals, cricket.limb_ids):
#   p.resetJointState(
#     cricket_id,
#     joint,
#     goal,
#     0,
#     cid
#   )

# while 1:
#   time.sleep(1./100.)
#   # print(cricket.get_final_joints())
#   # print(cricket.get_final_observation())
#   p.stepSimulation()

for i in range(p.getNumJoints(cricket, cid)):
  print('---'*60)
  print(p.getJointInfo(cricket,i,cid))
