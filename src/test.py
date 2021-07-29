from numpy.core.defchararray import array
from pybullet_utils import pd_controller_stable
#from pybullet_envs.deep_mimic.env import humanoid_pose_interpolator
import math
import numpy as np
import pybullet as p
import time
import pybullet_data
from gym_cricket.assests.cricket import Cricket

physicsClient = p.connect(p.GUI)
# angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
# throttle = p.addUserDebugParameter('Throttle', 0, 20, 0)
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=0,
    cameraPitch=-40,
    cameraTargetPosition=[0.55,-0.35,0.2])

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81, physicsClientId=physicsClient) 
file_path = "/home/andrea/Desktop/pybullet_cricket/cricket_description/urdf/cricket_robot_xacro.urdf" #"/home/andrea/Desktop/hebi_cricket_urdf/hebi_description/urdf/kits/cricket.urdf"
planeId = p.loadURDF("plane.urdf")

def contact_points_printer(contact_points, msg) :
    print(f'n contact points on {msg}: {len(contact_points)}')
    for count, contact_point in enumerate(contact_points):
        print(f'{count} - {contact_point[9]}')

##########################
### STARTING POSITIONS ###
##########################
# Flat
zero_starting_pos = np.zeros((24,))
# Random
rand_starting_pos = np.random.uniform(low=-3.14, high=3.14, size=(24,))
# All legs up 
allup_1 = np.full((8,), 0) # wheels
allup_2 = np.full((1,), 1) # r_front_shoulder_to_body
allup_3 = np.full((3,), 0) # others
allup_4 = np.full((1,), 1) # r_back_shoulder to body
allup_5 = np.full((3,), 0) # others
allup_6 = np.full((1,), -1) # l_front_shoulder_to_body
allup_7 = np.full((3,), 0) # others 
allup_8 = np.full((1,), -1) # l_back_shoulder
allup_9 = np.full((3,), 0) # others 
allUp_starting_pos = np.concatenate((allup_1,allup_2,allup_3,allup_4,allup_5,allup_6,allup_7,allup_8,allup_9))
# All legs down
allup_1 = np.full((8,), 0) # wheels
allup_2 = np.full((1,), -1) # r_front_shoulder_to_body
allup_3 = np.full((3,), 0) # others
allup_4 = np.full((1,), -1) # r_back_shoulder to body
allup_5 = np.full((3,), 0) # others
allup_6 = np.full((1,), 1) # l_front_shoulder_to_body
allup_7 = np.full((3,), 0) # others 
allup_8 = np.full((1,), 1) # l_back_shoulder
allup_9 = np.full((3,), 0) # others 
allDown_starting_pos = np.concatenate((allup_1,allup_2,allup_3,allup_4,allup_5,allup_6,allup_7,allup_8,allup_9))
# Quadrupede
allup_1 = np.full((8,), 0) # wheels
allup_2 = np.full((1,), 0) # r_front_shoulder_to_body
allup_3 = np.array([-1,1.55,-0.56])
# r_front_shoulder2_to_shoulder # r_front_shoulder2_to_knee # r_front_ankle # r_back_shoulder to body
allup_4 = np.full((1,), 0) # others
allup_5 = np.array([1,-1.55,0.56])
allup_6 = np.full((1,), 0)
allup_7 = np.array([-1,1.55,-0.56])
allup_8 = np.full((1,), 0) # others 
allup_9 = np.array([1,-1.55,0.56])
quadrupede_starting_pos = np.concatenate((allup_1,allup_2,allup_3,allup_4,allup_5,allup_6,allup_7,allup_8,allup_9))
# collide
coll_1 = np.full((10,), 0)
coll_2 = np.full((1,), -3.14) # r_front_shoulder2_to_knee
coll_3 = np.full((13,), 0)
coll_starting_pos = np.concatenate((coll_1,coll_2,coll_3))

# Start
robot = Cricket(physicsClient,quadrupede_starting_pos)
##########################
## STOP STARTING POS #####
##########################

print("="*80)
print(robot.get_observations())
print("="*80)
robot_id = robot.get_ids()[0]

number_of_joints = p.getNumJoints(robot_id) # 41
print('*'*80)
print(f'n_joints {number_of_joints}')
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
mvm_ = 0
while True:
    print("*-*"*40)
    print(robot.get_normal_forces(planeId))
    print("*-*"*40)
    # user_angle = p.readUserDebugParameter(angle)
    # user_throttle = p.readUserDebugParameter(throttle)
    # for joint_index in wheel_indices:
    #     p.setJointMotorControl2(robot_id, joint_index,
    #                             p.VELOCITY_CONTROL,
    #                             targetVelocity=user_throttle)
    # for joint_index in wheel_indices:
    #     p.setJointMotorControl2(robot_id, joint_index,
    #                             p.POSITION_CONTROL, 
    #                             targetPosition=user_angle)
    counter +=1

    #print(f'observation: {robot.get_observations()}')
    # print(f"{counter}: state joints")
    # print(f'limb 0: {p.getJointState(robot_id, 10)}')
    # print(f'limb 1: {p.getJointState(robot_id, robot.limb_joints[1][0])}')
    mvm = 0.0
    mvm_ = 0.03
    action_1 = np.array(
        [
            [mvm_,0.],[mvm_,0.0],[mvm_,0.0],
            [mvm_,0.],[0.0,0.0],[0.0,0.0],
            [mvm,0.],[0.0,0.0],[0.0,0.0],
            [mvm,0.],[0.0,0.0],[0.0,0.0],
            [mvm,0.],[0.0,0.0],[0.0,0.0],
            [mvm,0.],[0.0,0.0],[0.0,0.0],
            [mvm,0.],[0.0,0.0]
        ]
    )
    bella = "*" * 20
    
    for i in range(7,8):
        # print(f"{bella} get AABB {bella}")
        aa, bb = p.getAABB(robot_id,i,physicsClient) # return the bounding box of the body (-1) starting from the center of mass
        # print(f'AA:{aa}')
        # print(f'BB:{bb}')
        # print(f"{i} - {bella} getOverlappingObjects {bella}")
        # obs = p.getOverlappingObjects(aa, bb, physicsClient)
        # to_print = []
        # excl = 10000
        # for e in obs:
        #     if e[1] != 6 and e[1] != 7 and e[1] != 8 :
        #         to_print.append(e)
        # print(f'OBS {to_print}')
    print()
    print(f"{bella} getContactPoints {bella}")
    t_pos, l_pos = robot.get_joint_positions()
    print(t_pos[0])

    # for i in range(0,43):
    #     # print(p.getCollisionShapeData(robot_id,i,physicsClient))
    #     print(p.getContactPoints(robot_id, robot_id,i))

    # print("___"*50)
    # contact_points_printer(p.getContactPoints(robot_id, planeId, linkIndexA=10), 'wheel_1')
    # contact_points_printer(p.getContactPoints(robot_id, planeId, linkIndexA=11), 'track')
    # contact_points_printer(p.getContactPoints(robot_id, planeId, linkIndexA=12), 'wheel_2')
    # print("___"*50)
    # if aa[2] < 0 :
    #     print("touch!!!!!!!!!!")
    #     time.sleep(1000)
    if flag :
        robot.perform_action(action_1)
        if p.getJointState(robot_id, robot.limb_joints[0][0])[0] > 1.566:
            # print("**"*100)
            flag = False
    else :
        robot.perform_action(action_1)
        # if p.getJointState(robot_id, robot.limb_joints[0][0])[0] < -1.566 :
        #     break

    time.sleep(1./1.)
    p.stepSimulation()

# #print(f'observation: {robot.get_observations()}')
# print(p.getJointInfo(robot_id, robot.limb_joints[1][0]))
# #robot.perform_action([1.57,0.7])
# ori = p.getQuaternionFromEuler([3.14,0.,0.])
# targetPosJoints = p.calculateInverseKinematics(robot_id, 7, [0.9,0.1,1.6], targetOrientation = ori)
# print(f'targetjointpos {targetPosJoints}')
# p.setJointMotorControlArray(robot_id, range(24), p.POSITION_CONTROL, targetPositions = targetPosJoints)