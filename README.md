# Preference-based Inverse Reinforcement Learning Approach for Robot Locomotion under Suboptimal Learning Conditions
**Bachelor's Thesis in Robotics**

## Setup


### System:
- Linux Mint 19.2 
- Python 3.6.8 (Note: anything higher than Ubuntu 16.4 does the job as well)
- I recommend using a virtual python environment

### Requirements:

Please refer to the requirements.txt file for the dependencies (Note: Not all requirements are needed, there have to be done some checks of what is indispensable)
Especially for setting up the snake mujoco environment, have a look at [this README file](README_Lemke.md). 

```bash
    pip install -r requirements.txt
```

## Usage

### Short Description

This project consists of three major steps to perform IRL. 

1. run ppo on the ground truth and get trajectories (list of observations) and the timestep 

2. Depending on the used loss function, we preprocess the dataset, eg. form pairs for binary crossentropy and triplets for triplet loss. Next we learn the reward function; here we use an ensemble of networks to stabilize training.

3. Lastly, we run ppo again but now using the learnt reward function und see whether we can recover it.

### Script Templates

All commands are run inside the [gym_mujoco_planar_snake](gym_mujoco_planar_snake) directory.



## Author
Andreas Binder, Information Systems (B.Sc.), Technical University Munich


