import tensorflow as tf

# tf.compat.v1.enable_eager_execution()

from tensorflow import keras

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gym_mujoco_planar_snake.common.reward_nets import *

from time import ctime


import os

class Trainer:

    def __init__(self, hparams,
                 save_path,
                 use_tensorboard=False,
                 Save=True
                 ):
        self.hparams = hparams
        self.save_path = save_path
        self.results = {
            "loss": 0.,
            "accuracy": 0.,
            "test_loss": 0.,
            "test_accuracy": 0.
        }
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.accuracy = None
        self.train_loss = None

        # initialize validation metrics
        self.validation_loss = None
        self.validation_accuracy = None

        # initialize test metrics
        self.test_loss = None
        self.test_accuracy = None

        self.train_summary_writer = None
        self.val_summary_writer = None
        self.test_summary_writer = None
        self.hparams_summary_writer = None

        # documentation
        self.use_tensorboard = use_tensorboard
        self.Save = Save

        self.time = ctime()

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    def save(self, net, result_name="hparams_and_results.json"):

        path = os.path.join(self.save_path, self.time)
        os.mkdir(path)
        torch.save(net.state_dict(), os.path.join(path, "model"))
        self.hparams.save(os.path.join(path, result_name))



    def initialize_tensorboard(self, time):

        if self.use_tensorboard:
            from tensorboardX import SummaryWriter
            import datetime

            # log_dir = 'gym_mujoco_planar_snake/log/tensorboard/PyTorch/'
            log_dir = "gym_mujoco_planar_snake/log/PyTorch_Models/tensorboard"

            # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            current_time = time
            train_log_dir = log_dir + current_time + '/train'
            val_log_dir = log_dir + current_time + '/val'
            # test_log_dir = log_dir + current_time + '/test'

            self.train_summary_writer = SummaryWriter(train_log_dir)
            self.val_summary_writer = SummaryWriter(val_log_dir)
            # self.test_summary_writer = SummaryWriter(test_log_dir)

            # TODO hparams
            self.hparams_summary_writer = SummaryWriter(log_dir)

    def add_to_tensorboard(self, mode, epoch):

        if self.use_tensorboard:

            if mode == 'train':
                self.train_summary_writer.add_scalar('train_loss', self.train_loss.result().numpy(), epoch)
                self.train_summary_writer.add_scalar('train_accuracy', self.accuracy.result().numpy(), epoch)
            elif mode == 'val':
                self.val_summary_writer.add_scalar('val_loss', self.validation_loss.result().numpy(), epoch)
                self.val_summary_writer.add_scalar('val_accuracy', self.validation_accuracy.result().numpy(), epoch)
            elif mode == 'test':
                self.test_summary_writer.add_scalar('test_loss', self.test_loss.result().numpy(), 1)
                self.test_summary_writer.add_scalar('test_accuracy', self.test_accuracy.result().numpy(), 1)

    # TODO include batchsize
    # TODO divide running loss by data size
    def fit_pair(self, dataset):
        # get hyperparameters and data
        # batch_size = self.hparams["batch_size"]
        batch_size = self.hparams.dict["batch_size"]
        time = self.time
        # lr = self.hparams["lr"]
        lr = self.hparams.dict["lr"]
        # epochs = self.hparams["epochs"]
        epochs = self.hparams.dict["epochs"]
        pairs, labels = dataset

        length = pairs.shape[0]
        # split dataset
        split_factor = 0.8
        train_pairs, test_pairs = pairs[:int(split_factor * length)], pairs[int(split_factor * length):]
        # x_test = x_train[int(split_factor * length):]

        train_labels, test_labels = labels[:int(split_factor * length)], labels[int(split_factor * length):]
        # y_test = y_train[int(split_factor * length):]

        '''print(train_pairs.shape)
        print(test_pairs.shape)
        print(train_labels.shape)
        print(test_labels.shape)

        assert False, train_pairs.shape'''

        '''np.random.shuffle(x_train)
        np.random.shuffle(y_train)'''

        self.initialize_tensorboard(time)

        # TODO test how to add hparams

        '''self.hparams_summary_writer.add_hparams({'lr': 0.1, 'bsize': 10},
                                                {'hparam/accuracy': 100, 'hparam/loss': 0.1})'''

        net = PairNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # print(x_train.shape)

        # training
        print("Start Training")
        for epoch in range(epochs):

            # epoch_loss = 0.0
            running_loss = 0.0
            for step in range(train_labels.size):
                traj_i, traj_j = torch.from_numpy(train_pairs[step, 0, :]).float().view(1350), torch.from_numpy(
                    train_pairs[step, 1, :]).float().view(1350)
                y = torch.from_numpy(np.array(train_labels[step]))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)

                rewards = rewards.unsqueeze(0)
                y = y.unsqueeze(0)

                '''print(rewards)
                print(y)'''

                loss = criterion(rewards, y)

                # assert False, loss
                # print()

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                '''if step % 500 == 0:
                    print(step)'''
                '''if step % 500 == 499:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500))
                    running_loss = 0.0'''

            counter = 0
            for step in range(test_labels.size):
                traj_i, traj_j = torch.from_numpy(test_pairs[step, 0, :]).float().view(1350), torch.from_numpy(
                    test_pairs[step, 1, :]).float().view(1350)
                y = torch.from_numpy(np.array(test_labels[step]))

                # zero the parameter gradients
                # optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)

                if y.item() == torch.argmax(rewards):
                    counter += 1

            template = 'Epoch {}, Loss: {:10.4f}, Accuracy: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / train_labels.size,
                counter / test_labels.size
            )
            print(stats)
            self.train_summary_writer.add_scalar('train_loss', running_loss / train_labels.size, epoch)
            self.val_summary_writer.add_scalar('val_acc', counter / test_labels.size, epoch)

        if self.Save:
            from time import ctime
            import os
            path = os.path.join(self.save_path, time)
            os.mkdir(path)
            # torch.save(net.state_dict(), os.path.join(path, ctime()))
            torch.save(net.state_dict(), os.path.join(path, "model"))
            # self.model.save_weights(os.path.join(path, ctime()) + ".h5")
            self.hparams.save(os.path.join(path, "hparams.json"))

    def fit_pair_bce(self, dataset):
        # get hyperparameters and data
        # batch_size = self.hparams["batch_size"]
        batch_size = self.hparams.dict["batch_size"]
        time = self.time
        # lr = self.hparams["lr"]
        lr = self.hparams.dict["lr"]
        # epochs = self.hparams["epochs"]
        epochs = self.hparams.dict["epochs"]
        pairs, labels = dataset

        length = pairs.shape[0]
        # split dataset
        split_factor = 0.8
        train_pairs, test_pairs = pairs[:int(split_factor * length)], pairs[int(split_factor * length):]
        # x_test = x_train[int(split_factor * length):]

        train_labels, test_labels = labels[:int(split_factor * length)], labels[int(split_factor * length):]
        # y_test = y_train[int(split_factor * length):]

        '''print(train_pairs.shape)
        print(test_pairs.shape)
        print(train_labels.shape)
        print(test_labels.shape)

        assert False, train_pairs.shape'''

        '''np.random.shuffle(x_train)
        np.random.shuffle(y_train)'''

        self.initialize_tensorboard(time)

        # TODO test how to add hparams

        '''self.hparams_summary_writer.add_hparams({'lr': 0.1, 'bsize': 10},
                                                {'hparam/accuracy': 100, 'hparam/loss': 0.1})'''

        net = PairBCENet()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # print(x_train.shape)


        # training
        print("Start Training")
        for epoch in range(epochs):

            # epoch_loss = 0.0
            running_loss = 0.0
            for step in range(train_labels.size):

                traj_i, traj_j = torch.from_numpy(train_pairs[step, 0, :]).float().view(1350), torch.from_numpy(
                    train_pairs[step, 1, :]).float().view(1350)
                y = torch.from_numpy(np.array(train_labels[step])).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)

                rewards = rewards.unsqueeze(0)
                y = y.unsqueeze(0)

                '''print(rewards)
                print(y)'''

                #assert False, "bce"



                loss = criterion(rewards, y)

                # assert False, loss
                # print()

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                '''if step % 500 == 0:
                    print(step)'''
                '''if step % 500 == 499:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500))
                    running_loss = 0.0'''

            '''counter = 0
            for step in range(test_labels.size):
                traj_i, traj_j = torch.from_numpy(test_pairs[step, 0, :]).float().view(1350), torch.from_numpy(
                    test_pairs[step, 1, :]).float().view(1350)
                y = torch.from_numpy(np.array(test_labels[step]))

                # zero the parameter gradients
                # optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)

                if y.item() == torch.argmax(rewards):
                    counter += 1'''

            template = 'Epoch {}, Loss: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / train_labels.size
            )
            #print(stats)
            #self.train_summary_writer.add_scalar('train_loss', running_loss / train_labels.size, epoch)

        # needs save_path, time, net, hparams, torch, os

        if self.Save:
            from time import ctime
            import os
            path = os.path.join(self.save_path, time)
            os.mkdir(path)
            # torch.save(net.state_dict(), os.path.join(path, ctime()))
            torch.save(net.state_dict(), os.path.join(path, "model"))
            # self.model.save_weights(os.path.join(path, ctime()) + ".h5")
            self.hparams.save(os.path.join(path, "hparams.json"))

    def fit_triplet(self, dataset):
        # get hyperparameters and data
        # batch_size = self.hparams["batch_size"]
        time = self.time
        batch_size = self.hparams.dict["batch_size"]
        # lr = self.hparams["lr"]
        lr = self.hparams.dict["lr"]
        # epochs = self.hparams["epochs"]
        epochs = self.hparams.dict["epochs"]
        print(dataset.shape)
        x_train, y_train, z_train = dataset[:, 0, :, :], dataset[:, 1, :, :], dataset[:, 2, :, :]

        x_train, y_train, z_train = x_train[:2000], y_train[:2000], z_train[:2000]

        # print(x_train[0, :, :], y_train[0, :, :], z_train[0, :, :])

        print(x_train.shape)

        dim = x_train.shape[0]

        '''np.random.shuffle(x_train)
        np.random.shuffle(y_train)'''

        net_input = 1350

        # self.initialize_tensorboard()

        net = TripletNet()
        criterion = nn.TripletMarginLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        print(x_train.shape)

        trajs_h = torch.from_numpy(x_train).float().view(dim, net_input)
        trajs_i = torch.from_numpy(y_train).float().view(dim, net_input)
        trajs_j = torch.from_numpy(z_train).float().view(dim, net_input)

        print(trajs_i.shape)

        # training
        print("Start Training")
        for epoch in range(epochs):
            print("Epoch: ", str(epoch))

            running_loss = 0.0
            for step in range(dim):
                traj_h, traj_i, traj_j = trajs_h[step, :], trajs_i[step, :], trajs_j[step, :]

                '''print(traj_h, traj_i, traj_j)

                assert False, "NOO"'''

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_h, traj_i, traj_j)

                # print(rewards.shape)

                anchor, positive, negative = rewards[0].unsqueeze(0), rewards[1].unsqueeze(0), rewards[2].unsqueeze(0)

                # print(anchor, positive, negative)

                '''if step == 3:
                    assert False, "Stop in the Name of Love"'''

                anchor, positive, negative = anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0)

                loss = criterion(anchor, positive, negative)

                '''if step % 500 == 0:
                    print(loss)'''

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                '''if step % 500 == 499:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500))
                    running_loss = 0.0'''
            print('[%d] loss: %.3f' % (epoch + 1, running_loss / 500))
            running_loss = 0.0


        if self.Save:
            from time import ctime
            import os
            path = os.path.join(self.save_path, time)
            os.mkdir(path)
            # torch.save(net.state_dict(), os.path.join(path, ctime()))
            torch.save(net.state_dict(), os.path.join(path, "model"))
            # self.model.save_weights(os.path.join(path, ctime()) + ".h5")
            self.hparams.save(os.path.join(path, "hparams.json"))

    def fit_test(self, dataset):

        # get hyperparameters
        batch_size = self.hparams.dict["batch_size"]
        time = self.time
        lr = self.hparams.dict["lr"]
        epochs = self.hparams.dict["epochs"]
        self.hparams.dict["mode"] = "test"

        # get data
        pairs, labels = dataset


        # split dataset
        length = pairs.shape[0]
        split_factor = 0.8
        train_pairs, test_pairs = pairs[:int(split_factor * length)], pairs[int(split_factor * length):]
        train_labels, test_labels = labels[:int(split_factor * length)], labels[int(split_factor * length):]

        # start tensorboard
        if self.Save:
            self.initialize_tensorboard(time)

        # TODO test how to add hparams

        '''self.hparams_summary_writer.add_hparams({'lr': 0.1, 'bsize': 10},
                                                {'hparam/accuracy': 100, 'hparam/loss': 0.1})'''

        net = SingleStepPairNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # print(x_train.shape)

        running_loss = 0.0

        # training
        print("Start Training")
        for epoch in range(epochs):

            # epoch_loss = 0.0
            running_loss = 0.0
            for step in range(train_labels.size):
                traj_i, traj_j = torch.from_numpy(train_pairs[step, 0, :]).float().view(50, 27), torch.from_numpy(
                    train_pairs[step, 1, :]).float().view(50, 27)
                y = torch.from_numpy(np.array(train_labels[step]))


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)

                rewards = rewards.unsqueeze(0)
                y = y.unsqueeze(0)

                loss = criterion(rewards, y)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if step % 500 == 0:
                    print(step)


            template = 'Epoch {}, Loss: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / train_labels.size
                #counter / test_labels.size
            )
            print(stats)
            if self.Save:
                self.train_summary_writer.add_scalar('train_loss', running_loss / train_labels.size, epoch)

        if self.Save:
            self.hparams.dict["final_train_loss"] = running_loss / train_labels.size
            self.save(net=net)
