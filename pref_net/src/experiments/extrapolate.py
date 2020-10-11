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

    n = configs["num_samples"]

    train_episodes, extrapolation_episodes = sample(train_episodes, n), sample(extrapolation_episodes, n)

    train_timesteps = [episode[1] for episode in train_episodes]
    extrapolation_timesteps = [episode[1] for episode in extrapolation_episodes]

    train_predictions = [predict(episode[0], net).item() for episode in train_episodes]
    extrapolation_predictions = [predict(episode[0], net).item() for episode in extrapolation_episodes]

    train_distance = [sum(episode[2])  for episode in train_episodes]
    extrapolation_distance = [sum(episode[2]) for episode in extrapolation_episodes]

    # Plot
    plt.title("Extrapolation")

    joint = np.concatenate((train_predictions, extrapolation_predictions))

    # highest value
    #scale = np.concatenate((train_distance, extrapolation_distance), axis=0).max()
    scale = 8
    #print(scale)
    # TODO 5 rausnehmen
    joint = (np.array(joint) - np.array(joint).min()) / \
                        np.abs(np.array(joint).max() - np.array(joint).min()) * scale #* 6

    train_predictions, extrapolation_predictions = joint[:n], joint[n:]

    '''# statistics
    print("#"*5, "Mean", "#"*5)
    print("Distance %f"%(np.array(train_distance).mean()))
    print("Prediction %f"%(np.array(train_predictions).mean()))
    print("Distance %f" % (np.array(extrapolation_distance).mean()))
    print("Prediction %f" % (np.array(extrapolation_predictions).mean()))'''

    # try to get straight line by using
    '''train_distance, extrapolation_distance = np.exp(train_distance), np.exp(extrapolation_distance)
    train_predictions, extrapolation_predictions = np.exp(train_predictions), np.exp(extrapolation_predictions)'''

    # rescale
    # x_new = (x-min) / range * scalar
    '''train_predictions = (np.array(train_predictions) - np.array(train_predictions).min()) / \
                        np.abs(np.array(train_predictions).max() - np.array(train_predictions).min()) * 5

    extrapolation_predictions = (np.array(extrapolation_predictions) - np.array(extrapolation_predictions).min()) / \
                        np.abs(np.array(extrapolation_predictions).max() - np.array(extrapolation_predictions).min()) * 5'''

    plt.scatter(train_timesteps, train_predictions, color='b', label='Predictions on Training Data')
    plt.scatter(extrapolation_timesteps, extrapolation_predictions, color="r", label='Predictions on Unseen Data')

    plt.scatter(train_timesteps, train_distance, color='y', label='Distance of the Training Data')
    plt.scatter(extrapolation_timesteps, extrapolation_distance, color="g", label='Distance of the Unseen Data')

    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Predicted and Actual Distance")
    plt.show()

    # TODO
    # plot more sophisticated: labels, legends






