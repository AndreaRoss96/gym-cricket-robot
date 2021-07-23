import pybullet as p
import os
import time
import numpy as np
from gym_cricket.assests.cricket import Cricket
import pybullet_data

cid = p.connect(p.GUI)

cricket = Cricket(cid)
cricket_id, _ = cricket.get_ids()
goals = [0.0, -np.pi/2, np.pi, -np.pi/2, 0.0,\
np.pi/2, np.pi, np.pi/2, 0.0,\
-np.pi/2, np.pi, -np.pi/2, 0.0,\
np.pi/2, np.pi, np.pi/2, 0.0, 0.0]

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81, physicsClientId=cid)
planeId = p.loadURDF("plane.urdf")

for goal, joint in zip(goals, cricket.limb_ids):
  p.resetJointState(
    cricket_id,
    joint,
    goal,
    0,
    cid
  )

while 1:
  time.sleep(1./240.)
  p.stepSimulation()