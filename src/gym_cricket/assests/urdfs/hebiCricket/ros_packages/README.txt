Getting started

Copy these packages into a ROS Melodic workspace.
Make you you have installed the required hebi-cpp-api packages

sudo apt install ros-melodic-hebi-cpp-api

you can then build the workspace.

To start the ROS example for robot control, run

roslaunch hebi_cpp_api_examples cricket.launch

This configuration allows you to control the position, velocitity, and torque command of each module through two methods.
1. By generating a trajectory through provided waypoints (publish to /joint_waypoints). Generally using trajectories will result in smoother motion.
OR
2. By directly commanding the desired joint state (publish to /joint_target). Given a stream of messages published on this topic, the node will try to command all joints to match the most recently received message. There is a safety parameter called 'message_timeout', and if a new message has not been received for that duration (by default this is set to 0.5 seconds) the target is canceled and the modules go into a compliant state. To disable this feature, comment out message_timeout in the launch file.


For simulation, run

roslaunch hebi_gazebo cricket_simulation.launch

The launch file will start the simulator and the controlling group node as described above.

The urdf for the cricket can be found in hebi_description/urdf/kits/cricket.xacro
