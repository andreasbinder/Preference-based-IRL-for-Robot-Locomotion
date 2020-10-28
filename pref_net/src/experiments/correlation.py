from sklearn.metrics import ndcg_score

from scipy.stats.stats import pearsonr

import tensorflow as tf
# skip warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''from pref_net.src.utils import seeds as seeds

seed = 0
seeds.set_seeds(seed)'''

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
    configs = configs.data["correlation"]

    #from pref_net.src.utils import seeds as seeds
    # Seed
    seed = configs["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Net
    input_dim = 27
    #pair_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_opt/2020-10-08_13-29-39_Pair/model"
    pair_path = configs["pair_net_path"]
    net = Net(input_dim=input_dim)
    net.load_state_dict(torch.load(pair_path))



    #triplet_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_opt/2020-10-07_23-23-44_InitialTriplet_results2/model"
    triplet_path = configs["triplet_net_path"]
    triplet_net = Net(input_dim=input_dim)
    triplet_net.load_state_dict(torch.load(triplet_path))

    formula = lambda x: np.exp(x/100) * 4000 + 100000# distance np.exp(x) * 2000

    # Retrieve Data
    #data_path = "/home/andreas/Documents/pbirl-bachelorthesis/log/pi_subopt/data/2020-10-07_14-33-48/train.npy"
    data_path = configs["data_path"]

    train_episodes = load_data(data_path)

    paths = [
        "/home/andreas/Documents/pbirl-bachelorthesis/log/original_datasets/2020-10-10_00-00-31_Pair/model",
        "/home/andreas/Documents/pbirl-bachelorthesis/log/original_datasets/2020-10-13_12-47-12_Triplet/model",
        "/home/andreas/Documents/pbirl-bachelorthesis/log/original_datasets/2020-10-09_16-51-34_InitialTriplet_result/model"
    ]

    net_templates = [Net(input_dim=input_dim) for _ in range(len(paths))]

    nets = []
    for net_template, path in zip(net_templates, paths):
        net_template.load_state_dict(torch.load(path))
        nets.append(net_template)

    #nets = [net_template.load_state_dict(torch.load(path)) for net_template, path in zip(net_templates, paths)]

    train_timesteps = [episode[1] for episode in train_episodes]
    train_reward = np.array([sum(episode[3]) for episode in train_episodes])

    train_predictions = np.array([[predict(episode[0], net).item() for episode in train_episodes]
                                   for net in nets
                                   ])


    #print(np.corrcoef(train_reward, train_predictions[0]))
    #print(train_predictions.shape)

    net_labels = ["Pairnet Predictions", "Tripletnet Predictions", "NaiveTripletnet Predictions"]

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    fig.suptitle('Ground-truth Reward vs Predictions')

    ax1.scatter(train_timesteps, train_reward, c='b', label='Ground-truth Reward')
    ax1_twinx = ax1.twinx()
    ax1_twinx.scatter(train_timesteps, train_predictions[0], c='r',label=net_labels[0])

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twinx.get_legend_handles_labels()
    ax1_twinx.legend(lines + lines2, labels + labels2, loc=4)

    ax2.scatter(train_timesteps, train_reward, c='b', label='Ground-truth Reward')
    ax2_twinx = ax2.twinx()
    ax2_twinx.scatter(train_timesteps, train_predictions[1], c='r',label=net_labels[1])

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twinx.get_legend_handles_labels()
    ax2_twinx.legend(lines + lines2, labels + labels2, loc=4)

    ax3.scatter(train_timesteps, train_reward, c='b', label='Ground-truth Reward')
    ax3_twinx = ax3.twinx()
    ax3_twinx.scatter(train_timesteps, train_predictions[2], c='r',label=net_labels[2])

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twinx.get_legend_handles_labels()
    ax3_twinx.legend(lines + lines2, labels + labels2, loc=4)

    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Ground-truth Reward')
    ax1_twinx.set_ylabel('Predictions')

    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Ground-truth Reward')
    ax2_twinx.set_ylabel('Predictions')

    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Ground-truth Reward')
    ax3_twinx.set_ylabel('Predictions')

    plt.show()
    '''ax2 = ax1.twinx()
    #ax1.plot(x, y1, 'g-')
    ax1_scatter = ax1.scatter(x, y1, c='b', label='Ground-truth Reward')
    #ax2.plot(x, y2, 'b-')
    ax2_scatter = ax2.scatter(x, y2, c='r',label='Predicted Reward' )'''



    import sys
    sys.exit()
    # TODO
    train_predictions = np.array([predict(episode[0], net).item() for episode in train_episodes])
    triplet_train_predictions = np.array([predict(episode[0], triplet_net).item() for episode in train_episodes])
    #extrapolation_predictions = [predict(episode[0], net).item() for episode in extrapolation_episodes]

    train_reward = np.array([sum(episode[3])  for episode in train_episodes])
    #extrapolation_distance = [sum(episode[2]) for episode in extrapolation_episodes]

    '''train_predictions = (np.array(train_predictions) - np.array(train_predictions).min()) / \
                        np.abs(np.array(train_predictions).max() - np.array(train_predictions).min()) * max(
        train_reward)

    triplet_train_predictions = (np.array(triplet_train_predictions) - np.array(triplet_train_predictions).min()) / \
                                np.abs(np.array(triplet_train_predictions).max() - np.array(
                                    triplet_train_predictions).min()) * max(train_reward)'''


    '''train_reward = formula(train_reward)
    train_predictions = formula(train_predictions)
    triplet_train_predictions = formula(triplet_train_predictions)'''

    print("Correlation Triplet")
    print(np.corrcoef(train_reward, triplet_train_predictions))
    print("Correlation Pair")
    print(np.corrcoef(train_reward, train_predictions))

    from scipy import stats

    spear = stats.spearmanr(train_reward, triplet_train_predictions).correlation
    spear_pair = stats.spearmanr(train_reward, train_predictions).correlation

    print("Triplet Spear")
    print(spear)

    print("Pair Spear")
    print(spear_pair)

    import numpy as np
    import matplotlib.pyplot as plt

    '''    plt.scatter(train_reward, triplet_train_predictions)
    plt.ylabel("Predicted Reward")
    plt.xlabel("Ground-truth Reward")

    plt.show()

    import sys
    sys.exit()'''



    x = train_timesteps
    y1 = train_reward
    y2 = triplet_train_predictions

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    #ax1.plot(x, y1, 'g-')
    ax1_scatter = ax1.scatter(x, y1, c='b', label='Ground-truth Reward')
    #ax2.plot(x, y2, 'b-')
    ax2_scatter = ax2.scatter(x, y2, c='r',label='Predicted Reward' )


    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Ground-truth Reward')
    ax2.set_ylabel('Predicted Reward')
    '''ax1.legend(loc=2)
    ax2.legend(loc=6)'''
    #plt.legend()
    axes = ax1_scatter + ax2_scatter
    labels = [ax.get_label() for ax in axes]
    ax1.legend(axes, labels)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.show()


    import sys
    sys.exit()

    # TODO
    '''print("Correlation Triplet")
    print(np.corrcoef(train_reward, triplet_train_predictions))
    print("Correlation Pair")
    print(np.corrcoef(train_reward, train_predictions))

    plt.scatter(train_timesteps, triplet_train_predictions, color='b', label='Predictions on Training Data')'''
    # plt.scatter(extrapolation_timesteps, extrapolation_predictions, color="r", label='Predictions on Unseen Data')

    #plt.scatter(train_timesteps, train_reward, color='y', label='Distance of the Training Data')
    # plt.scatter(extrapolation_timesteps, extrapolation_distance, color="g", label='Distance of the Unseen Data')

    ##########################
    train_predictions = (np.array(train_predictions) - np.array(train_predictions).min()) / \
                            np.abs(np.array(train_predictions).max() - np.array(train_predictions).min()) * max(train_reward)

    triplet_train_predictions = (np.array(triplet_train_predictions) - np.array(triplet_train_predictions).min()) / \
                            np.abs(np.array(triplet_train_predictions).max() - np.array(triplet_train_predictions).min()) * max(train_reward)


    '''train_reward = formula(train_reward)
    train_predictions = np.clip(formula(train_predictions), 0, np.inf)
    triplet_train_predictions = np.clip(formula(triplet_train_predictions), 0, np.inf)
    '''

    train_reward = np.array(train_reward)
    train_predictions = np.array(train_predictions)
    triplet_train_predictions = np.array(triplet_train_predictions)

    train_reward = formula(train_reward)
    train_predictions = formula(train_predictions)
    triplet_train_predictions = formula(triplet_train_predictions)


    plt.scatter(train_timesteps, train_reward, color='g', label='Ground-truth Reward')

    plt.scatter(train_timesteps, train_predictions, color='b', label='Predictions of Pairnet')

    plt.scatter(train_timesteps, triplet_train_predictions, color='r', label='Predictions of Tripletnet')


    plt.plot(train_timesteps, train_timesteps, color='y')
    ##########################
    '''print(np.corrcoef(train_reward, triplet_train_predictions))
    print(np.corrcoef(train_reward, train_predictions))'''

    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Rescaled Ground-truth Reward")
    plt.show()





