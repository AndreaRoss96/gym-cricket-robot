# gym-cricket-robot

The code has been tested on Ubuntu 18.04 with ROS Melodic and the following languages and libraries.
in the following ReadMe there are some instruction on how to run and test the developped code.

## Structure

```
├───res
│       ...
└───src
    │   ddpg.py
    │   main.py
    │   out_rew.txt
    │   setup.py
    │   test.py
    │
    ├───gym_cricket
    │   │   __init__.py
    │   │
    │   ├───assests
    │   │   │   cricket.py
    │   │   │   cricketGoal.py
    │   │   │   cricket_abs.py
    │   │   │   hebi_cricket.py
    │   │   │   hebi_cricketGoal.py
    │   │   │
    │   │   ├───terrains
    │   │   │   ├───flat
    │   │   │   │     ...
    │   │   │   └───slope
    │   │   │         ...
    │   │   └───urdfs
    │   │       │   cricket_robot.urdf
    │   │       │   cricket_robot.urdf.xacro
    │   │       │
    │   │       └───hebiCricket
    │   │           ├───CAD
    │   │           │   ...
    │   │           ├───MATLAB Code
    │   │           │   ...
    │   │           └───ros_packages
    │   │               ├───hebi_cpp_api_ros_examples
    │   │               ├───hebi_description
    │   │               └───hebi_gazebo
    │   └───envs
    │           cricket_env.py
    │           __init__.py
    │
    ├───neural_network
    │       actor_nn.py
    │       critic_nn.py
    │
    ├───utils
    │       auxiliaryFuncs.py
    │       buffer.py
    │       evaluator.py
    │       memory.py
    │       OUNoise.py
    │       random_process.py
    │       util.py
    │
    └───weights_out
            actor.pkl
            critic.pkl
```


## Requirements
### Languages:
 
* python 3.6.4 or newer
* C++ (to compile HEBI robotics packages if needed)

### Libraries (not std)
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
run ```python main.py``` (the ```python``` command might change based on your version and the number of python versions installed)

### Flags:
#### Environment arguments
 
 * ```--mode``` support option: train/test
 * ```--env``` open-ai gym environment
 * ```--num_apisode``` total training episodes
 * ```--step_episode``` simulation steps per episode
 * ```--early_stop``` change episode after [early_stop] steps with a non-growing reward
 * ```--cricket``` [hebi_cricket, basic_cricket] - cricket urdf model you want to load
 * ```--terrain``` name of the terrain you want to load (to be implemented)

#### Reward function


 * ```--w_X``` weight X to compute difference between the robot and the optimal position.
 * ```--w_Y``` weight Y to compute difference between the robot and the optimal position. 
 * ```--w_Z``` weight Z to compute difference between the robot and the optimal position. 
 * ```--w_theta``` weight theta to compute difference between the robot and the optimal position.
 * ```--w_sigma``` weight sigma to compute difference between the robot and the optimal position.
 * ```--disct_factor``` discount factor for learning in the reward function
 * ```--w_joints``` weight to punish bad joints behaviours in the reward function

 #### Neural networks
 
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


 #### Ddpg arguments
 
 
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


#### To be optimized


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
     
## How to change terrain
The flat terrain is the flag ```terrain``` default value. To use another terrain you need to put the files required (*.mtl, *.obj, *.urdf) insede a folder with the same name of the terrai, under the subfolder ```src/gym_cricket/assests/terrains/```

e.g.
I want to use a new terrain named "slope"
* I create a new folder under ```src/gym_cricket/assests/terrains/``` named ```slope```
* I put all the useful files inside that folder: at least slope.mtl, slope.obj, slope.urdf (and in case other useful files)
* I run the script with the correct flag ```python main.py --terrain="slope"```


## How to customize the neural networks
The default neural networks* are the following:
* FC 400x300x150 --> it is possible to add layers **up to a total of 5 hidden layers** (by changing the code it is easy to add more)
* 3D-CNN --> the default three dimensional CNN is very simple composed just by the input layer and a squared kernel 1x1x1. If you want to add more convolutional layer you also need to mind the respective **kernel sizes**. Again the maximum number of hidden layers is 5.

\* to understand how the neural networks work give a look to the report with the explanation of the algorithm

