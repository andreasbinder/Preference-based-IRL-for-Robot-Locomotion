from sklearn.metrics import ndcg_score

from scipy.stats.stats import pearsonr

import tensorflow as tf
# skip warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from pref_net.src.utils import seeds as seeds

seed = 0
seeds.set_seeds(seed)

import torch
import torch.nn as nn

#  TODO
# pairnet vs tripletnet
# predictions on 100 samples from extrapolation

import numpy as np


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import torch
import torch.nn as nn
from pref_net.src.common.misc_util import Configs

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


def predict(input, net):
    tensor = torch.tensor(input).float()

    output = net(tensor)

    return output

def load_data(path):

    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True)

    return data




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_configs', type=str,
                        default="/home/andreas/Documents/pbirl-bachelorthesis/pref_net/configs.yml")
    args = parser.parse_args()

    configs = Configs(args.path_to_configs)
    configs = configs.data["extrapolation"]


    # Seed
    seed = configs["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Net
    input_dim = 27
    pair_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_opt/2020-10-08_13-29-39_Pair/model"
    pair_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/original_datasets/2020-10-10_00-00-31_Pair/model"
    net = Net(input_dim=input_dim)
    net.load_state_dict(torch.load(pair_path))



    triplet_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_opt/2020-10-07_23-23-44_InitialTriplet_results2/model"
    triplet_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/original_datasets/2020-10-09_16-51-34_InitialTriplet/model"
    triplet_net = Net(input_dim=input_dim)
    triplet_net.load_state_dict(torch.load(triplet_path))

    # Retrieve Data
    data_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_subopt/data/2020-10-07_14-33-48/train.npy"
    data_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/original_datasets/extrapolate.npy"

    train_episodes, extrapolation_episodes = load_data(data_path), load_data(data_path)

    n = configs["num_samples"]

    train_episodes, extrapolation_episodes = sample(train_episodes, n), sample(extrapolation_episodes, n)

    train_timesteps = [episode[1] for episode in train_episodes]
    #extrapolation_timesteps = [episode[1] for episode in extrapolation_episodes]

    train_predictions = [predict(episode[0], net).item() for episode in train_episodes]
    triplet_train_predictions = [predict(episode[0], triplet_net).item() for episode in train_episodes]
    #extrapolation_predictions = [predict(episode[0], net).item() for episode in extrapolation_episodes]

    train_distance = [sum(episode[2])  for episode in train_episodes]
    #extrapolation_distance = [sum(episode[2]) for episode in extrapolation_episodes]

    print("Correlation Triplet")
    print(np.corrcoef(train_distance, triplet_train_predictions))
    print("Correlation Pair")
    print(np.corrcoef(train_distance, train_predictions))

    plt.scatter(train_timesteps, triplet_train_predictions, color='b', label='Predictions on Training Data')
    # plt.scatter(extrapolation_timesteps, extrapolation_predictions, color="r", label='Predictions on Unseen Data')

    plt.scatter(train_timesteps, train_distance, color='y', label='Distance of the Training Data')
    # plt.scatter(extrapolation_timesteps, extrapolation_distance, color="g", label='Distance of the Unseen Data')

    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Predicted and Actual Distance")
    plt.show()

    import sys

    sys.exit()
    # Plot
    plt.title("Extrapolation")

    #joint = np.concatenate((train_predictions, extrapolation_predictions))

    # highest value
    #scale = np.concatenate((train_distance, extrapolation_distance), axis=0).max()
    scale = 8
    #print(scale)
    # TODO 5 rausnehmen
    '''joint = (np.array(joint) - np.array(joint).min()) / \
                        np.abs(np.array(joint).max() - np.array(joint).min()) * scale #* 6

    train_predictions, extrapolation_predictions = joint[:n], joint[n:]'''

    #train_predictions = (np.array(train_predictions) - np.array(train_predictions).min()) / np.abs(np.array(train_predictions).max() - np.array(train_predictions).min()) * 8

    '''# statistics
    print("#"*5, "Mean", "#"*5)
    print("Distance %f"%(np.array(train_distance).mean()))
    print("Prediction %f"%(np.array(train_predictions).mean()))
    print("Distance %f" % (np.array(extrapolation_distance).mean()))
    print("Prediction %f" % (np.array(extrapolation_predictions).mean()))'''

    #train_distance, extrapolation_distance = np.exp(train_distance), np.exp(train_distance)
    #train_predictions, extrapolation_predictions = np.exp(train_predictions), np.exp(train_distance)

    # rescale
    # x_new = (x-min) / range * scalar
    '''train_predictions = (np.array(train_predictions) - np.array(train_predictions).min()) / \
                        np.abs(np.array(train_predictions).max() - np.array(train_predictions).min()) * 5

    extrapolation_predictions = (np.array(extrapolation_predictions) - np.array(extrapolation_predictions).min()) / \
                        np.abs(np.array(extrapolation_predictions).max() - np.array(extrapolation_predictions).min()) * 5'''

    plt.scatter(train_timesteps, train_predictions, color='b', label='Predictions on Training Data')
    #plt.scatter(extrapolation_timesteps, extrapolation_predictions, color="r", label='Predictions on Unseen Data')

    plt.scatter(train_timesteps, train_distance, color='y', label='Distance of the Training Data')
    #plt.scatter(extrapolation_timesteps, extrapolation_distance, color="g", label='Distance of the Unseen Data')

    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Predicted and Actual Distance")
    plt.show()

    # TODO
    # plot more sophisticated: labels, legends











data_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_subopt/data/2020-10-07_14-33-48/extrapolate.npy"

with open(data_path, 'rb') as f:
    data = np.load(f, allow_pickle=True)

# 20 nice
num_samples = 100

samples = sample(data, num_samples)

triplet_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_opt/2020-10-07_23-23-44_InitialTriplet_results2/model"
pair_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_opt/2020-10-08_13-29-39_Pair/model"
triplet_net, pair_net = load_net(triplet_path), load_net(pair_path)


triplet_predictions = np.array([triplet_net(torch.tensor(inp[0]).float()).item() for inp in samples])
pair_predictions = np.array([pair_net(torch.tensor(inp[0]).float()).item() for inp in samples])



time_steps = np.array([sum(ts[2]) for ts in samples])


scale = time_steps.max()
# rescale
triplet_predictions = (triplet_predictions - triplet_predictions.min()) / \
                        np.abs(triplet_predictions.max() - triplet_predictions.min()) * scale

pair_predictions = (pair_predictions - pair_predictions.min()) / \
                        np.abs(pair_predictions.max() - pair_predictions.min()) * scale


print("Score Triplet")
print(np.corrcoef(triplet_predictions, time_steps))
print("Score Pair")
print(np.corrcoef(pair_predictions, time_steps))
print("Score Random")

import matplotlib.pyplot as plt

steps = np.array([ts[1] for ts in samples])

plt.scatter(steps, triplet_predictions+time_steps.min(), c="red")
plt.scatter(steps, time_steps, c="blue")
plt.show()


'''triplet_ranks = get_ranks(triplet_predictions)
pair_ranks = get_ranks(pair_predictions)

optimal = np.arange(num_samples)

random = np.arange(num_samples)

np.random.shuffle(random)

print("Score Triplet")
print(ndcg_score(optimal.reshape(1, num_samples), triplet_ranks.reshape(1, num_samples)))
print("Score Pair")
print(ndcg_score(optimal.reshape(1, num_samples), pair_ranks.reshape(1, num_samples)))
print("Score Random")
print(ndcg_score(optimal.reshape(1, num_samples), random.reshape(1, num_samples)))'''

#pearsonr()
