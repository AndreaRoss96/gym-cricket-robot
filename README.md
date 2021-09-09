# gym-cricket-robot

The code has been tested on Ubuntu 18.04 with ROS Melodic and the following languages and libraries.

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

 **neural networks**
 
 * ```--hidden1``` hidden num of first fully connect layer
 * ```--hidden2``` hidden num of second fully connect layer
 * ```--hidden3``` hidden num of third fully connect layer
 * ```--hidden4``` hidden num of fourth fully connect layer
 * ```--hidden5``` hidden num of fifth fully connect layer
 * ```--conv_hidden1``` hidden num of first convolutional layer
 * ```--conv_hidden2``` hidden num of second convolutional layer
 * ```--conv_hidden3``` hidden num of third convolutional layer
 * ```--conv_hidden4``` hidden num of fourth convolutional layer
 * ```--conv_hidden5``` hidden num of fifth convolutional layer
 * ```--kernel_size1``` num of first kernel for cnn
 * ```--kernel_size2``` num of second kernel for cnn
 * ```--kernel_size3``` num of third kernel for cnn
 * ```--kernel_size4``` num of fourth kernel for cnn


 **ddpg arguments**
 
 
 * ```--bsize```   minibatch size
* ```--rate``` learning rate
* ```--prate``` policy net learning rate (only for DDPG)
* ```--warmup``` time without training but only filling the replay memory
* ```--discount``` discount factor
* ```--rmsize``` memory size
* ```--window_length``` 
* ```--tau``` moving average for target network
* ```--ou_theta``` noise theta
* ```--ou_sigma``` noise sigma
* ```--ou_mu``` noise mu 


**To be optimized**


* ```--validate_episodes``` how many episode to perform during validate experiment
* ```--max_episode_length``` how many steps to perform a validate experiment
* ```--validate_steps``` train iters each timestep
* ```--output``` linear decay of exploration policy
* ```--debug```  Resuming model path for testing
* ```--init_w```         
* ```--train_iter```
* ```--epsilon```    
* ```--seed```       
* ```--resume``` 
      
