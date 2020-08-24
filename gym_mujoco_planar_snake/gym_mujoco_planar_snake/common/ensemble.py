from gym_mujoco_planar_snake.common.documentation import *
from gym_mujoco_planar_snake.common.losses import *
from gym_mujoco_planar_snake.common.results import *

import os


import torch
import torch.nn as nn
import torch.optim as optim


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

    def cum_return(self, traj):
        # return of one traj
        sum_rewards = 0
        # sum_abs_rewards = 0
        r = self.model(traj)
        sum_rewards += torch.sum(r)
        # sum_abs_rewards += torch.sum(torch.abs(r))

        return sum_rewards

    def forward(self, traj_i, traj_j):
        cum_r_i = self.cum_return(traj_i)
        cum_r_j = self.cum_return(traj_j)

        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0).unsqueeze(0)


class Ensemble(object):

    def __init__(self, args, net_save_path):
        self.num_nets = args.num_nets
        self.ranking_loss = args.ranking_loss
        self.net_save_path = net_save_path

        self.hparams = Params(args.hparams_path)
        self.lr = self.hparams.lr
        self.epochs = self.hparams.epochs
        self.split_ratio = 0.8
        # TODO change for hopper
        self.input_dim = 27

        # TODO Test Phase only for 2 and 3
        assert self.ranking_loss in [2, 3], "Test Phase only for 2 and 3"

    def preprocess_data(self, raw_data):
        data = build_trainset(raw_data, self.ranking_loss)
        data = split_dataset_for_nets(data, self.num_nets)
        return data

    def fit(self, raw_data):

        whole_dataset = self.preprocess_data(raw_data)

        nets = [Net(self.input_dim)] * self.num_nets

        for index, (net, dataset) in enumerate(zip(nets, whole_dataset)):
            self.train(index, net, dataset)

    def train(self, index, net, dataset):

        train_set, val_set = split_into_train_val(dataset, self.split_ratio)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=self.lr)

        running_loss = 0.0
        running_val_loss = 0.0

        # training
        print("Start Training")
        for epoch in range(self.epochs):

            running_loss = 0.0
            for item in train_set:
                inputs, label = item

                loss = self.forward_pass(net, optimizer, criterion, inputs, label)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            ##########
            running_val_loss = 0.0
            for item in train_set:
                inputs, label = item

                loss = self.forward_pass(net, optimizer, criterion, inputs, label)

                # print statistics
                running_val_loss += loss.item()

            ##########

            template = 'Epoch {}, Loss: {:10.4f}, Validation Loss: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / len(train_set),
                running_val_loss / len(val_set)
                # counter / test_labels.size
            )
            print(stats)

        self.hparams.dict["final_train_loss"] = running_loss / len(train_set)
        self.hparams.dict["final_val_loss"] = running_val_loss / len(val_set)
        self.save(index=index, net=net)

    def forward_pass(self, net, optimizer, criterion, inputs, label):
        traj_i, traj_j = torch.from_numpy(inputs[0]).float(), torch.from_numpy(inputs[1]).float()
        y = torch.tensor([label])

        optimizer.zero_grad()

        rewards = net(traj_i, traj_j)

        loss = criterion(rewards, y)

        return loss

    def save(self, index, net, result_name="hparams_and_results.json"):

        #path = os.path.join(self.net_save_path, self.time)
        #os.mkdir(path)
        torch.save(net.state_dict(), os.path.join(self.net_save_path, "model_"+str(index)))
        self.hparams.save(os.path.join(self.net_save_path, "hparams_and_results"+str(index)+".json"))

