from baselines.common.running_mean_std import RunningMeanStd
import numpy as np


rms = RunningMeanStd(shape=())

x = np.array([10])

rms.update(x)
