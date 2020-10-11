# PrefNet: Inverse Reinforcement Learning Approach for Robot Locomotion under Suboptimal Learning Conditions
**Bachelor's Thesis in Robotics**

## Installation

We recommend to install using [Anaconda](https://www.anaconda.com). Afterwards, you need to perform to steps to use this repository.

1. Create conda environment
```bash
    conda env create -f snake_env.yml
```
2. Activate environment
```bash
    conda activate snake_env
```

Furthermore, we use the Mujoco physics engine for the simulation. You will need to create a folder called `.mujoco` in your home directory and put your [license key](https://www.roboti.us/license.html) there. Also download mujoco200 or mjpro150 [here](https://www.roboti.us/index.html) and unzip it inside `.mujoco`, both versions should work.
If you encounter any problem, file an issue or have a look at https://github.com/openai/mujoco-py. 

You might also need to set some system variables in .bashrc/.zshrc:
```bash
    # example for mjpro150, mujoco 200 analogously
    export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mjpro150
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/.mujoco/mjpro150/bin
```

## Content

This repository contains the [source code](./pref_net/src) to reproduce the results for my thesis. All hyperparameters are set in the [configs file](./pref_net/configs.yml). Make sure that you reference the configs file when executing a python script.

Additionally, my [thesis](./thesis) and my [final presentation](./presentation) are stored here.

## Usage

You can download the training data in different splits here. TODO
Alternatively, you can run create_dataset. 


```bash
    python pref_net/src/create_dataset.py ----path_to_configs PATH/TO/CONFIGS
```

Once you either downloaded or saved the data, you need to reference train.npy in the configuration file and set your hyperparameters.
In case you want to run PPO on top of the learnt reward function, you have to set run_RL to true.
```bash

    python pref_net/src/create_dataset.py ----path_to_configs PATH/TO/CONFIGS
```

Once you either downloaded or saved the data, you need to reference train.npy in the configuration file and set your hyperparameters.
In case you want to run PPO on top of the learnt reward function, you have to set run_RL to true.
```bash

    python pref_net/src/train.py ----path_to_configs PATH/TO/CONFIGS
```

The ```visualize.py``` file lets you visualize a PPO run. 


```bash
    python pref_net/src/visualize.py ----path_to_configs PATH/TO/CONFIGS
```


```bash
    python pref_net/src/visualize.py ----path_to_configs PATH/TO/CONFIGS
```


TODO experiments
TODO extrapolate
TODO evaluation




## Author
Andreas Binder, Information Systems (B.Sc.), Technical University Munich

