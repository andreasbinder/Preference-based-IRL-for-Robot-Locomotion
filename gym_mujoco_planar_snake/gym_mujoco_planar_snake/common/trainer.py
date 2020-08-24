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
                 use_tensorboard,
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
        # add time stamp as identifier
        self.hparams.dict["time_stamp"] = self.time

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
            log_dir = "gym_mujoco_planar_snake/results/improved_runs/tensorboard/"

            # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            current_time = time
            train_log_dir = log_dir + current_time + '/train'
            val_log_dir = log_dir + current_time + '/val'
            # test_log_dir = log_dir + current_time + '/test'

            self.train_summary_writer = SummaryWriter(train_log_dir)
            self.val_summary_writer = SummaryWriter(val_log_dir)
            # self.test_summary_writer = SummaryWriter(test_log_dir)

            # TODO hparams
            hparams_log_dir = log_dir + '/hparams'
            self.hparams_summary_writer = SummaryWriter(hparams_log_dir)

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

        '''self.hparams_summaTue Aug 18 14:33:50 2020100000ry_writer.add_hparams({'lr': 0.1, 'bsize': 10},
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
        self.hparams.dataset_size = labels.size

        # split dataset
        length = pairs.shape[0]
        split_factor = 0.8
        train_pairs, test_pairs = pairs[:int(split_factor * length)], pairs[int(split_factor * length):]
        train_labels, test_labels = labels[:int(split_factor * length)], labels[int(split_factor * length):]

        # start tensorboard
        if self.Save:
            self.initialize_tensorboard(time)


        net = SingleStepPairNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # print(x_train.shape)

        running_loss = 0.0
        running_val_loss = 0.0

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
                '''if step % 500 == 0:
                    print("Step", step, "Loss", running_loss / (step + 1))'''


            # TODO validation
            ##########
            running_val_loss = 0.0
            for step in range(test_labels.size):
                traj_i, traj_j = torch.from_numpy(test_pairs[step, 0, :]).float().view(50, 27), torch.from_numpy(
                    test_pairs[step, 1, :]).float().view(50, 27)
                y = torch.from_numpy(np.array(test_labels[step]))

                # zero the parameter gradients
                #optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)

                rewards = rewards.unsqueeze(0)
                y = y.unsqueeze(0)

                loss = criterion(rewards, y)

                #loss.backward()
                #optimizer.step()

                # print statistics
                running_val_loss += loss.item()

            ##########




            template = 'Epoch {}, Loss: {:10.4f}, Validation Loss: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / train_labels.size,
                running_val_loss / test_labels.size
                #counter / test_labels.size
            )
            print(stats)
            self.train_summary_writer.add_scalar('train_loss', running_loss / train_labels.size, epoch)
            self.val_summary_writer.add_scalar('val_loss', running_val_loss / test_labels.size, epoch)

        # TODO save hparams
        self.hparams_summary_writer.add_hparams({'lr': lr, 'bsize': batch_size, 'epochs': epochs},
                                                {'hparam/train_loss': running_loss / train_labels.size,
                                                 'hparam/val_loss': running_val_loss / test_labels.size})

        if self.Save:
            self.hparams.dict["final_train_loss"] = running_loss / train_labels.size
            self.hparams.dict["final_val_loss"] = running_val_loss / test_labels.size
            self.save(net=net)


    def fit_test_batch(self, dataset):

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





        net = SingleStepPairNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # print(x_train.shape)

        running_loss = 0.0
        running_val_loss = 0.0

        #assert False, train_pairs.shape

        # shape (num_batches, batch_size, 2, 50, 27)
        #batch = np.array([train_pairs[b:b + batch_size] for b in range(0, train_pairs.shape[0], batch_size)])
        train_pair_batch = np.array([train_pairs[b:b + batch_size] for b in range(0, train_pairs.shape[0], batch_size)])
        train_label_batch = np.array([train_labels[b:b + batch_size] for b in range(0, train_labels.shape[0], batch_size)])
        #train_batch = torch.from_numpy(batch)
        '''print(train_pair_batch[0].shape)
        print(train_pair_batch[-1].shape)

        print(train_label_batch[0].shape)
        print(train_label_batch[-1].shape)'''

        train_batch = list(zip(train_pair_batch, train_label_batch))

        #print(train_batch[0])


        #print(train_batch.shape)



        #num_batches = train_pair_batch.shape[0]

        # training
        print("Start Training")
        for epoch in range(epochs):

            # epoch_loss = 0.0
            running_loss = 0.0
            for pair_batch, label_batch in train_batch:

                print(pair_batch.shape)
                print(label_batch.shape)


                traj_i, traj_j = torch.from_numpy(pair_batch[:, 0, :, :]).float(), torch.from_numpy(
                    pair_batch[:, 1, :, :]).float()
                y = torch.from_numpy(label_batch)

                print(traj_i.shape)
                print(traj_i.shape)
                print(y.shape)





                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)



                rewards = rewards.unsqueeze(0)
                y = y.unsqueeze(0)

                print(rewards.shape)
                print(y.shape)

                loss = criterion(rewards, y)

                assert False, "TT"

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                '''if step % 500 == 0:
                    print("Step", step, "Loss", running_loss / (step + 1))'''


            # TODO validation
            ##########
            running_val_loss = 0.0
            for step in range(test_labels.size):
                traj_i, traj_j = torch.from_numpy(test_pairs[step, 0, :]).float().view(50, 27), torch.from_numpy(
                    test_pairs[step, 1, :]).float().view(50, 27)
                y = torch.from_numpy(np.array(test_labels[step]))

                # zero the parameter gradients
                #optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)

                rewards = rewards.unsqueeze(0)
                y = y.unsqueeze(0)

                loss = criterion(rewards, y)

                #loss.backward()
                #optimizer.step()

                # print statistics
                running_val_loss += loss.item()

            ##########




            template = 'Epoch {}, Loss: {:10.4f}, Validation Loss: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / train_labels.size,
                running_val_loss / test_labels.size
                #counter / test_labels.size
            )
            print(stats)
            self.train_summary_writer.add_scalar('train_loss', running_loss / train_labels.size, epoch)
            self.val_summary_writer.add_scalar('val_loss', running_val_loss / test_labels.size, epoch)

        # TODO save hparams
        self.hparams_summary_writer.add_hparams({'lr': lr, 'bsize': batch_size, 'epochs': epochs},
                                                {'hparam/train_loss': running_loss / train_labels.size,
                                                 'hparam/val_loss': running_val_loss / test_labels.size})

        if self.Save:
            self.hparams.dict["final_train_loss"] = running_loss / train_labels.size
            self.save(net=net)


    def fit_hinge(self, dataset):

        # get hyperparameters
        batch_size = self.hparams.dict["batch_size"]
        time = self.time
        lr = self.hparams.dict["lr"]
        epochs = self.hparams.dict["epochs"]
        self.hparams.dict["mode"] = "test"

        # get data
        pairs, labels = dataset


        #print(np.array(labels).shape)
        #print(labels[:10])

        # convert labels from (0,1) to (1,-1)
        labels = np.array(labels)
        labels[labels == 1] = -1
        labels[labels == 0] = 1

        #print(labels[:10])

        #assert False, "in hinge"

        # split dataset
        length = pairs.shape[0]
        split_factor = 0.8
        train_pairs, test_pairs = pairs[:int(split_factor * length)], pairs[int(split_factor * length):]
        train_labels, test_labels = labels[:int(split_factor * length)], labels[int(split_factor * length):]

        # start tensorboard
        if self.Save:
            self.initialize_tensorboard(time)


        net = SingleStepPairNet()
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.MarginRankingLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # print(x_train.shape)

        running_loss = 0.0
        running_val_loss = 0.0

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

                #rewards = rewards.unsqueeze(0)
                y = y.unsqueeze(0)
                '''print(y)
                print(rewards)'''


                loss = criterion(rewards[:1], rewards[1:], y)

                loss.backward()
                optimizer.step()

                # assert False, "in hinge"

                # print statistics
                running_loss += loss.item()
                '''if step % 500 == 0:
                    print("Step", step, "Loss", running_loss / (step + 1))'''


            # TODO validation
            ##########
            running_val_loss = 0.0
            for step in range(test_labels.size):
                traj_i, traj_j = torch.from_numpy(test_pairs[step, 0, :]).float().view(50, 27), torch.from_numpy(
                    test_pairs[step, 1, :]).float().view(50, 27)
                y = torch.from_numpy(np.array(test_labels[step]))

                # zero the parameter gradients
                #optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)

                #rewards = rewards.unsqueeze(0)
                y = y.unsqueeze(0)

                #loss = criterion(rewards, y)

                loss = criterion(rewards[:1], rewards[1:], y)

                #loss.backward()
                #optimizer.step()

                # print statistics
                running_val_loss += loss.item()

            ##########




            template = 'Epoch {}, Loss: {:10.4f}, Validation Loss: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / train_labels.size,
                running_val_loss / test_labels.size
                #counter / test_labels.size
            )
            print(stats)
            self.train_summary_writer.add_scalar('train_loss', running_loss / train_labels.size, epoch)
            self.val_summary_writer.add_scalar('val_loss', running_val_loss / test_labels.size, epoch)

        # TODO save hparams
        self.hparams_summary_writer.add_hparams({'lr': lr, 'bsize': batch_size, 'epochs': epochs},
                                                {'hparam/train_loss': running_loss / train_labels.size,
                                                 'hparam/val_loss': running_val_loss / test_labels.size})

        if self.Save:
            self.hparams.dict["final_train_loss"] = running_loss / train_labels.size
            self.save(net=net)

    def fit_triplet_single_observation(self, dataset):

        # get hyperparameters
        batch_size = self.hparams.dict["batch_size"]
        time = self.time
        lr = self.hparams.dict["lr"]
        epochs = self.hparams.dict["epochs"]
        self.hparams.dict["mode"] = "triplet_single"

        # get data
        X, Y, Z = dataset[:, 0, :, :], dataset[:, 1, :, :], dataset[:, 2, :, :]

        num_train = 4000
        num_test = 1000


        x_train, y_train, z_train = X[:num_train], Y[:num_train], Z[:num_train]

        x_test, y_test, z_test = X[num_train:], Y[num_train:], Z[num_train:]

        # print(x_train[0, :, :], y_train[0, :, :], z_train[0, :, :])






        trajs_h = torch.from_numpy(x_train).float().view(num_train, 50, 27)
        trajs_i = torch.from_numpy(y_train).float().view(num_train, 50, 27)
        trajs_j = torch.from_numpy(z_train).float().view(num_train, 50, 27)

        trajs_k = torch.from_numpy(x_test).float().view(num_test, 50, 27)
        trajs_l = torch.from_numpy(y_test).float().view(num_test, 50, 27)
        trajs_m = torch.from_numpy(z_test).float().view(num_test, 50, 27)


        # start tensorboard
        if self.Save:
            self.initialize_tensorboard(time)

        net = SingleObservationTripletNet()
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.TripletMarginLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # print(x_train.shape)

        running_loss = 0.0
        running_val_loss = 0.0

        # training
        print("Start Training")
        for epoch in range(epochs):
            print("Epoch: ", str(epoch+1))
            # epoch_loss = 0.0
            running_loss = 0.0
            for step in range(num_train):
                traj_h, traj_i, traj_j = trajs_h[step, :], trajs_i[step, :], trajs_j[step, :]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_h, traj_i, traj_j)

                # rewards = rewards.unsqueeze(0)
                anchor, positive, negative = rewards[0].unsqueeze(0), rewards[1].unsqueeze(0), rewards[2].unsqueeze(0)
                '''print(y)
                print(rewards)'''


                anchor = anchor.unsqueeze(0)
                positive = positive.unsqueeze(0)
                negative = negative.unsqueeze(0)




                loss = criterion(anchor, positive, negative)

                '''print(anchor)

                assert False, "in train"'''

                loss.backward()
                optimizer.step()

                # assert False, "in hinge"

                # print statistics
                running_loss += loss.item()
            #print('[%d] loss: %.3f' % (epoch + 1, running_loss / 500))
            #running_loss = 0.0

            # TODO validation
            ##########
            running_val_loss = 0.0
            for step in range(num_test):
                traj_h, traj_i, traj_j = trajs_k[step, :], trajs_l[step, :], trajs_m[step, :]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_h, traj_i, traj_j)

                # rewards = rewards.unsqueeze(0)
                anchor, positive, negative = rewards[0].unsqueeze(0), rewards[1].unsqueeze(0), rewards[2].unsqueeze(0)
                '''print(y)
                print(rewards)'''
                anchor = anchor.unsqueeze(0)
                positive = positive.unsqueeze(0)
                negative = negative.unsqueeze(0)


                loss = criterion(anchor, positive, negative)

                # loss = criterion(rewards, y)

                #loss = criterion(rewards[:1], rewards[1:], y)

                # loss.backward()
                # optimizer.step()

                # print statistics
                running_val_loss += loss.item()

            ##########

            template = 'Epoch {}, Loss: {:10.4f}, Validation Loss: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / num_train,
                running_val_loss / num_test
                # counter / test_labels.size
            )
            print(stats)
            self.train_summary_writer.add_scalar('train_loss', running_loss / num_train, epoch)
            self.val_summary_writer.add_scalar('val_loss', running_val_loss / num_test, epoch)

        # TODO save hparams
        self.hparams_summary_writer.add_hparams({'lr': lr, 'bsize': batch_size, 'epochs': epochs},
                                                {'hparam/train_loss': running_loss / num_train,
                                                 'hparam/val_loss': running_val_loss / num_test})

        if self.Save:
            self.hparams.dict["final_train_loss"] = running_loss / num_train
            self.save(net=net)

    def fit_bce_v2(self, dataset):

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





        net = PairBCENet()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # print(x_train.shape)

        running_loss = 0.0
        running_val_loss = 0.0

        # training
        print("Start Training")
        for epoch in range(epochs):

            # epoch_loss = 0.0
            running_loss = 0.0
            for step in range(train_labels.size):
                traj_i, traj_j = torch.from_numpy(train_pairs[step, 0, :]).float().view(50, 27), torch.from_numpy(
                    train_pairs[step, 1, :]).float().view(50, 27)
                y = torch.from_numpy(np.array(train_labels[step])).float()


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
                '''if step % 500 == 0:
                    print("Step", step, "Loss", running_loss / (step + 1))'''


            # TODO validation
            ##########
            running_val_loss = 0.0
            for step in range(test_labels.size):
                traj_i, traj_j = torch.from_numpy(test_pairs[step, 0, :]).float().view(50, 27), torch.from_numpy(
                    test_pairs[step, 1, :]).float().view(50, 27)
                y = torch.from_numpy(np.array(test_labels[step])).float()

                # zero the parameter gradients
                #optimizer.zero_grad()

                # forward + backward + optimize
                rewards, abs_rewards = net(traj_i, traj_j)

                rewards = rewards.unsqueeze(0)
                y = y.unsqueeze(0)

                loss = criterion(rewards, y)

                #loss.backward()
                #optimizer.step()

                # print statistics
                running_val_loss += loss.item()

            ##########




            template = 'Epoch {}, Loss: {:10.4f}, Validation Loss: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / train_labels.size,
                running_val_loss / test_labels.size
                #counter / test_labels.size
            )
            print(stats)
            self.train_summary_writer.add_scalar('train_loss', running_loss / train_labels.size, epoch)
            self.val_summary_writer.add_scalar('val_loss', running_val_loss / test_labels.size, epoch)

        # TODO save hparams
        self.hparams_summary_writer.add_hparams({'lr': lr, 'bsize': batch_size, 'epochs': epochs},
                                                {'hparam/train_loss': running_loss / train_labels.size,
                                                 'hparam/val_loss': running_val_loss / test_labels.size})

        if self.Save:
            self.hparams.dict["final_train_loss"] = running_loss / train_labels.size
            self.save(net=net)
