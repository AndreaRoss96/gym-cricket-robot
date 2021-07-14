from pybullet_utils import pd_controller_stable
#from pybullet_envs.deep_mimic.env import humanoid_pose_interpolator
import math
import numpy as np
import pybullet as p
import time
import pybullet_data
from gym_cricket.assests.cricket import Cricket

physicsClient = p.connect(p.GUI)
angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
throttle = p.addUserDebugParameter('Throttle', 0, 20, 0)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81, physicsClientId=physicsClient) 
file_path = "/home/andrea/Desktop/pybullet_cricket/cricket_description/urdf/cricket_robot_xacro.urdf" #"/home/andrea/Desktop/hebi_cricket_urdf/hebi_description/urdf/kits/cricket.urdf"
planeId = p.loadURDF("plane.urdf")

robot = Cricket(physicsClient)
print("="*80)
print(robot.get_observations())
print("="*80)
robot_id = robot.get_ids()[0]

number_of_joints = p.getNumJoints(robot_id) # 41
print('*'*80)
print(number_of_joints)
for joint_number in range(number_of_joints):
    #print("*"*80)
    info = p.getJointInfo(robot_id, joint_number)
    #print(info[1:3])

robot.print_info()

# for _ in range(300000): 
#     pos, ori = p.getBasePositionAndOrientation(robot)
#     p.applyExternalForce(robot, 0, [50, 0, 0], pos, p.WORLD_FRAME)
#     p.stepSimulation()

wheel_indices = [1, 3, 4, 5,6,7,8,9]
hinge_indices = [0, 2]
counter = 0
flag = True
mvm = 0
while True:
    user_angle = p.readUserDebugParameter(angle)
    user_throttle = p.readUserDebugParameter(throttle)
    for joint_index in wheel_indices:
        p.setJointMotorControl2(robot_id, joint_index,
                                p.VELOCITY_CONTROL,
                                targetVelocity=user_throttle)
    for joint_index in wheel_indices:
        p.setJointMotorControl2(robot_id, joint_index,
                                p.POSITION_CONTROL, 
                                targetPosition=user_angle)
    counter +=1

    #print(f'observation: {robot.get_observations()}')
    print(f"{counter}: state joints")
    print(f'limb 0: {p.getJointState(robot_id, 10)}')
    print(f'limb 1: {p.getJointState(robot_id, robot.limb_joints[1][0])}')
    mvm += 0.5
    action_1 = np.array(
        [
            [mvm,0.5],
            [0.0,0.1],
            [0.0,0.0]
        ]
    )
    if flag :
        robot.perform_action(action_1)
        if p.getJointState(robot_id, robot.limb_joints[0][0])[0] > 1.566:
            print("**"*100)
            flag = False
    else :
        robot.perform_action([1.57,0])
        # if p.getJointState(robot_id, robot.limb_joints[0][0])[0] < -1.566 :
        #     break

    time.sleep(1./10.)
    p.stepSimulation()

# #print(f'observation: {robot.get_observations()}')
# print(p.getJointInfo(robot_id, robot.limb_joints[1][0]))
# #robot.perform_action([1.57,0.7])
# ori = p.getQuaternionFromEuler([3.14,0.,0.])
# targetPosJoints = p.calculateInverseKinematics(robot_id, 7, [0.9,0.1,1.6], targetOrientation = ori)
# print(f'targetjointpos {targetPosJoints}')
# p.setJointMotorControlArray(robot_id, range(24), p.POSITION_CONTROL, targetPositions = targetPosJoints)