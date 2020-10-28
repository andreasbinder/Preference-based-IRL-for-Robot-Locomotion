# PrefNet: Inverse Reinforcement Learning Approach for Robot Locomotion under Suboptimal Learning Conditions
**Bachelor's Thesis in Robotics**

## Installation

We recommend to install using [Anaconda](https://www.anaconda.com). Afterwards, you need to perform to steps to use this repository.

1. Create conda environment
```bash
    conda env create -f environment.yml # environment.lock.yml
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

Either 

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


TODO experiments
TODO extrapolate
TODO evaluation

## Project Overview

```
├── AUTHORS.rst             <- List of developers and maintainers.
├── CHANGELOG.rst           <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yaml        <- The conda environment file for reproducibility.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── bhm_at_scale        <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```


## Author
Andreas Binder, Information Systems (B.Sc.), Technical University Munich

