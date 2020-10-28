from sklearn.metrics import ndcg_score

from scipy.stats.stats import pearsonr

import tensorflow as tf
# skip warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from pref_net.src.utils import seeds as seeds
from pref_net.src.utils.configs import Configs

import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_configs', type=str,
                        default="/home/andreas/Documents/pbirl-bachelorthesis/pref_net/configs.yml")
    args = parser.parse_args()

    configs = Configs(args.path_to_configs)
    configs = configs.data["ranking_quality"]


    # Seed
    seed = configs["seed"]
    torch.manual_seed(2)
    np.random.seed(seed)
    #seeds.set_seeds(seed)

    data_path = configs["data_path"]
    with open(data_path, 'rb') as f:
        data = np.load(f, allow_pickle=True)

    # num samples
    num_samples = configs["num_samples"]

    # TODO
    samples = list(data[:num_samples])

    # path to triplet net
    triplet_path = configs["triplet_path"]

    # path to pair net
    pair_path = configs["pair_path"]

    triplet_net, pair_net = load_net(triplet_path), load_net(pair_path)

    # predictions
    triplet_predictions = np.array([triplet_net(torch.tensor(inp[0]).float()) for inp in samples])
    pair_predictions = np.array([pair_net(torch.tensor(inp[0]).float()) for inp in samples])

    # get ranks
    triplet_ranks = get_ranks(triplet_predictions)
    pair_ranks = get_ranks(pair_predictions)

    # benchmarks
    optimal = np.arange(num_samples)
    worst = optimal[::-1]
    random =  np.random.choice(np.arange(0, num_samples), replace=False, size=(num_samples,))

    # scores
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
