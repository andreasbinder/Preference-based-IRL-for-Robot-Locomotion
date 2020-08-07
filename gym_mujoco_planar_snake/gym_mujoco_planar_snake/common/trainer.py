import tensorflow as tf

#tf.compat.v1.enable_eager_execution()

from tensorflow import keras

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gym_mujoco_planar_snake.common.reward_nets import TripletNet, PairNet






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

        # documentation
        self.use_tensorboard = use_tensorboard
        self.Save = Save

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    # https://www.tensorflow.org/tensorboard/get_started
    # @tf.function
    def initialize_tensorboard(self):

        if self.use_tensorboard:
            from tensorboardX import SummaryWriter
            import datetime

            log_dir = 'gym_mujoco_planar_snake/log/tensorboard/PyTorch/'

            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = log_dir + current_time + '/train'
            #val_log_dir = log_dir + current_time + '/val'
            #test_log_dir = log_dir + current_time + '/test'

            self.train_summary_writer = SummaryWriter(train_log_dir)
            #self.val_summary_writer = SummaryWriter(val_log_dir)
            #self.test_summary_writer = SummaryWriter(test_log_dir)

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
        #batch_size = self.hparams["batch_size"]
        batch_size = self.hparams.dict["batch_size"]
        #lr = self.hparams["lr"]
        lr = self.hparams.dict["lr"]
        #epochs = self.hparams["epochs"]
        epochs = self.hparams.dict["epochs"]
        x_train, y_train = dataset

        # split dataset
        split_factor = 6000
        x_train = x_train[:split_factor]
        y_train = y_train[:split_factor]


        '''np.random.shuffle(x_train)
        np.random.shuffle(y_train)'''

        self.initialize_tensorboard()



        net = PairNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        print(x_train.shape)

        # training
        print("Start Training")
        for epoch in range(epochs):

            #epoch_loss = 0.0
            running_loss = 0.0
            for step in range(y_train.size):

                traj_i, traj_j = torch.from_numpy(x_train[step, 0, :]).float().view(1350), torch.from_numpy(x_train[step, 1, :]).float().view(1350)
                y = torch.from_numpy(np.array(y_train[step]))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)

                rewards = rewards.unsqueeze(0)
                y = y.unsqueeze(0)
                '''print(rewards.shape)
                print(y.shape)'''


                #rewards= nn.sigmoid(rewards)
                #print(rewards.unsqueeze(0))
                #print(abs_rewards)

                #rewards = rewards.unsqueeze(0)
                '''print(y)
                print(rewards)'''

                #assert False, "in fit_pair"

                loss = criterion(rewards, y)

                #print(loss.item())
                '''import sys
                print("basst")
                sys.exit()'''


                loss.backward()
                optimizer.step()

                '''import sys
                print("basst")
                sys.exit()'''

                # print statistics
                running_loss += loss.item()
                '''if step % 500 == 499:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500))
                    running_loss = 0.0'''
            # template = 'Epoch {}, Loss: {:10.4f}, Accuracy: {:10.4f}, Val Loss: {:10.4f}, Val Accuracy: {:10.4f}'
            template = 'Epoch {}, Loss: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / y_train.size
            )
            print(stats)
            self.train_summary_writer.add_scalar('train_loss', running_loss, epoch)



        if self.Save:
            from time import ctime
            import os
            path = os.path.join(self.save_path, ctime())
            os.mkdir(path)
            #torch.save(net.state_dict(), os.path.join(path, ctime()))
            torch.save(net.state_dict(), os.path.join(path, "model"))
            # self.model.save_weights(os.path.join(path, ctime()) + ".h5")
            self.hparams.save(os.path.join(path, "hparams.json"))


    def fit_triplet(self, dataset):
        # get hyperparameters and data
        #batch_size = self.hparams["batch_size"]
        batch_size = self.hparams.dict["batch_size"]
        #lr = self.hparams["lr"]
        lr = self.hparams.dict["lr"]
        #epochs = self.hparams["epochs"]
        epochs = self.hparams.dict["epochs"]
        print(dataset.shape)
        x_train, y_train, z_train = dataset[:, 0, :, :], dataset[:, 1, :, :], dataset[:, 2, :, :]

        dim = x_train.shape[0]

        '''np.random.shuffle(x_train)
        np.random.shuffle(y_train)'''

        net_input = 1350

        self.initialize_tensorboard()

        net = TripletNet()
        criterion = nn.TripletMarginLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        print(x_train.shape)




        trajs_h = torch.from_numpy(x_train).float().view(dim, net_input)
        trajs_i = torch.from_numpy(x_train).float().view(dim, net_input)
        trajs_j = torch.from_numpy(x_train).float().view(dim, net_input)

        print(trajs_i.shape)



        # training
        print("Start Training")
        for epoch in range(epochs):

            running_loss = 0.0
            for step in range(dim):

                traj_h, traj_i, traj_j = trajs_h[step,:], trajs_i[step,:], trajs_j[step,:]


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_h, traj_i, traj_j)

                print(rewards.shape)

                anchor, positive, negative = rewards[0].unsqueeze(0), rewards[1].unsqueeze(0), rewards[2].unsqueeze(0)

                #rewards = rewards.unsqueeze(0)

                '''print(rewards.shape)
                print(y.shape)'''


                #rewards= nn.sigmoid(rewards)
                #print(rewards.unsqueeze(0))
                #print(abs_rewards)

                #rewards = rewards.unsqueeze(0)
                print(anchor, positive, negative)

                loss = criterion(anchor, positive, negative)

                assert False, "Hier"

                #print(loss.item())
                '''import sys
                print("basst")
                sys.exit()'''


                loss.backward()
                optimizer.step()

                '''import sys
                print("basst")
                sys.exit()'''

                # print statistics
                running_loss += loss.item()
                if step % 500 == 499:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500))
                    running_loss = 0.0

            self.train_summary_writer.add_scalar('train_loss', running_loss, epoch)





