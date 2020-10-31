# PrefNet: Inverse Reinforcement Learning Approach for Robot Locomotion under Suboptimal Learning Conditions
**Bachelor's Thesis in Robotics**

## Installation

We recommend to install using [Anaconda](https://www.anaconda.com). Afterwards, you need to perform to steps to use this repository.

1. Create conda environment
```bash
    conda env create -f environment.yml # environment.lock.yml for exact reproduction
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

Download the data [here](https://drive.google.com/drive/folders/1f9TB0Cu2I_MI_XHtFz8YpHjHQNeEs833?usp=sharing). 
Overview of the folder:
```
├── agent                               <- contains the index and data files of the policy that were used to create the training data.                                   
│   ├── 000005000.data-00000-of-00001       The policy was checkpointed every figth episode, which equals 5000 timesteps.       
│   ├── 000005000.index                     The policy was trained for 1.5 Mio timesteps, thus this directory contains 300 policies of varying quality. 
│   ├── ...                                 If you run create dataset, it will create one trajectory of length 1000 for every saved policy.
│   ├── 001500000.data-00000-of-00001       The standard configuration will save the first 200 of those into train.npy and the other 100 into extrapolate.npy
│   └── 001500000.index                 
├── train.npy                           <- contains policy rollouts of length 1000 for the first 200 policies  
├── extrapolate.npy                     <- contains policy rollouts of length 1000 for the last 100 policies, used for testing 
└── subtrajectories.npy                 <- the actual training data, created from train.npy
```

### Data Creation

`subtrajectories.npy` is the actual training data. You can use it directly and skip the rest of this section.

If you want to create the data yourself, you can run `create_dataset.py`. Then, you need to adjust two parameters in the `create_dataset` stage in the `configs.yml` file.
The `model_dir` parameter is the path to the agent files you downloaded. The `data_dir` parameter is the location the generated `subtrajectories.npy` gets saved.
```bash
    python pref_net/src/create_dataset.py --path_to_configs PATH/TO/CONFIGS
```

### Preference Learning

In the `train` stage in the `configs.yml` file, you can set the parameters for the reward learning and the policies you could train on top of it.
First, define the `data_dir` parameter which points to the `subtrajectories.npy` file. `save_dir` represents the directory the reward model and policies trained with it are stored. Concerning monitoring, `tensorboard_dir` is the directory for the tensorboard logs. Lastly, `ranking_method` is the ranking method the network is trained with. You can choose between Pair, Triplet and NaiveTriplet.
```bash
    python pref_net/src/train.py --path_to_configs PATH/TO/CONFIGS
```

## Author
Andreas Binder, Information Systems (B.Sc.), Technical University Munich

