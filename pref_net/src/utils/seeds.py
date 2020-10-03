import tensorflow as tf
import torch
import numpy as np
import random

def set_seeds(seed):

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Set seed to %i"%seed)