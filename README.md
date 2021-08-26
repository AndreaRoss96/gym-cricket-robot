# gym-cricket-robot

## Requirements
<b>lang</b>
 
python 3.6.4 or newer

C++ (to compile HEBI robotics packages if needed)

<b>libraries (not std)</b>
 * gym
 * pybullet
 * numpy
 * torch 
 * pywavefront
 * matplotlib
 * argparse
 * setuptools
 * pathlib
 
## How to run
run ```python main.py```

**flags:**
**environment arguments**
 * ```--mode``` support option: train/test
 * ```--env``` open-ai gym environment
 * ```--num_apisode``` total training episodes
 * ```--step_episode``` simulation steps per episode
 * ```--early_stop``` change episode after [early_stop] steps with a non-growing reward
 * ```--cricket``` [hebi_cricket, basic_cricket] - cricket urdf model you want to load
 * ```--terrain``` name of the terrain you want to load 
**reward function**
 * ```--w_X``` weight X to compute difference between the robot and the optimal position. Used in the reward function
 * ```--w_Y``` weight Y to compute difference between the robot and the optimal position. Used in the reward function
 * ```--w_Z``` weight Z to compute difference between the robot and the optimal position. Used in the reward function
 * ```--w_theta``` weight theta to compute difference between the robot and the optimal position. Used in the reward function
 * ```--w_sigma``` weight sigma to compute difference between the robot and the optimal position. Used in the reward function
 * ```--disct_factor``` discount factor for learnin in the reward function
 * ```--w_joints``` weight to punish bad joints behaviours in the reward function
 * ```--
 * ```--
