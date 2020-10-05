import numpy as np

import os.path as osp
import tensorflow as tf

from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy
#from baselines.ppo1.pposgd_simple import learn
from baselines import logger

import gym, logging
from gym.core import ObservationWrapper
import os

from src.common.env_wrapper import prepare_env
from src.common import my_tf_util
from src.common.misc_util import Configs
#from src.benchmark.info_collector import InfoCollector, InfoDictCollector

from baselines.common.mpi_running_mean_std import RunningMeanStd

from baselines.common.distributions import make_pdtype

from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

class ModelSaverWrapper(ObservationWrapper):

    def __init__(self, env, model_dir, save_frequency_steps):
        ObservationWrapper.__init__(self, env=env)


        self.save_frequency_steps = save_frequency_steps
        self.total_steps = 0
        self.total_steps_save_counter = 0
        self.total_episodes = 0


        self.model_dir = model_dir

    def reset(self, **kwargs):
        self.total_episodes += 1

        # todo start saving after 100k timesteps
        if self.total_steps_save_counter == self.save_frequency_steps or self.total_steps == 1:
            buffer = 9

            len_total_steps = len(str(self.total_steps))

            zeros = (buffer - len_total_steps) * "0"

            file_name = osp.join(self.model_dir, zeros + str(self.total_steps))

            my_tf_util.save_state(file_name)

            logger.log('Saved model to: ' + file_name)

            self.total_steps_save_counter = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps += 1
        self.total_steps_save_counter += 1

        return self.env.step(action)

    def observation(self, observation):
        return observation

