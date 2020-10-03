import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import torch
import torch.nn as nn
from src.common.misc_util import Configs

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
    net = Net(input_dim=input_dim)
    net.load_state_dict(torch.load(configs["net_path"]))

    # Retrieve Data
    train_episodes, extrapolation_episodes = load_data(configs["train_path"]), load_data(configs["extrapolation_path"])

    train_timesteps = [episode[1] for episode in train_episodes]
    extrapolation_timesteps = [episode[1] for episode in extrapolation_episodes]

    train_distance = [sum(episode[2])  for episode in train_episodes]
    extrapolation_distance = [sum(episode[2]) for episode in extrapolation_episodes]

    # Plot
    plt.title("Extrapolation")

    plt.scatter(train_timesteps, train_distance, color='b')
    plt.scatter(extrapolation_timesteps, extrapolation_distance, color="r")

    plt.show()

    # TODO
    # plot more sophisticated: labels, legends






