from sklearn.metrics import ndcg_score

from scipy.stats.stats import pearsonr

import tensorflow as tf
# skip warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from pref_net.src.utils import seeds as seeds



import torch
import torch.nn as nn

#  TODO
# pairnet vs tripletnet
# predictions on 100 samples from extrapolation

import numpy as np



class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, traj):
        sum_rewards = 0

        r = self.model(traj)

        sum_rewards += torch.sum(r)

        return sum_rewards

def get_oridnal_ranking(array):

    order = array.argsort()

    orinal_array = array.copy()

    for index, order_index in enumerate(order):
        orinal_array[order_index] = index

    #orinal_array = np.array([ orinal_array for index, order_index in enumerate(order) ])

    return orinal_array


def get_ranks(array):
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks

def get_ranking_prediction():
    pass

def load_net(path):
    net = Net(27)
    net.load_state_dict(torch.load(path))

    return net

def sample(full_episodes, n):

    n_episodes = len(list(full_episodes))

    starts = np.random.randint(0, n_episodes, size=n)
    starts.sort()

    samples = [full_episodes[index] for index in starts]

    return samples

seed = 0
seeds.set_seeds(seed)

#data_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_subopt/data/2020-10-07_14-33-48/extrapolate.npy"
data_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/original_datasets/extrapolate.npy"

with open(data_path, 'rb') as f:
    data = np.load(f, allow_pickle=True)

# 20 nice
num_samples = 100

#samples = sample(data, num_samples)

samples = list(data[:num_samples])

#triplet_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_opt/2020-10-07_23-23-44_InitialTriplet_results2/model"

triplet_path = "/tmp/pi_opt/2020-10-10_13-22-55_InitialTriplet/model"
#"/home/andreas/Documents/pbirl-bachelorthesis/log/original_datasets/2020-10-09_16-51-34_InitialTriplet/model"

#pair_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_opt/2020-10-08_13-29-39_Pair/model"
pair_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/original_datasets/2020-10-10_00-00-31_Pair/model"

triplet_net, pair_net = load_net(triplet_path), load_net(pair_path)


triplet_predictions = np.array([triplet_net(torch.tensor(inp[0]).float()) for inp in samples])
pair_predictions = np.array([pair_net(torch.tensor(inp[0]).float()) for inp in samples])

triplet_ranks = get_ranks(triplet_predictions)
pair_ranks = get_ranks(pair_predictions)

optimal = np.arange(num_samples)

random = np.arange(num_samples)

np.random.shuffle(random)

worst = optimal[::-1]

print("Score Triplet")
print(ndcg_score(optimal.reshape(1, num_samples), triplet_ranks.reshape(1, num_samples)))
print("Score Pair")
print(ndcg_score(optimal.reshape(1, num_samples), pair_ranks.reshape(1, num_samples)))
print("Score Random")
print(ndcg_score(optimal.reshape(1, num_samples), random.reshape(1, num_samples)))
print("Score Worst")
print(ndcg_score(optimal.reshape(1, num_samples), worst.reshape(1, num_samples)))
print("Score Best")
print(ndcg_score(optimal.reshape(1, num_samples), optimal.reshape(1, num_samples)))

#pearsonr()
