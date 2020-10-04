import torch
import torch.nn as nn
import numpy as np


from tqdm import tqdm



class TripletLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(TripletLoss, self).__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()



    def forward(self, prediction, target):


        return self.loss_fn(prediction, target)

    def prepare_data(self, raw_data, configs):
        # input shape: (num trajectories, 2)
        num_items = 3

        num_trajectories = len(list(raw_data))

        num_tuples = int(num_trajectories / num_items)

        available_labels = np.arange(num_items)

        final_dataset = []

        episode_start = lambda index: element[index, 1] % configs["episode_length"]
        calculate_distance = lambda index_a, index_b: abs(element[index_a, 1] - element[index_b, 1])

        for t in tqdm(range(num_tuples)):

            while True:

                indices = np.random.randint(num_trajectories, size=num_items)

                # check that indices are unique
                if len(set(indices)) != num_items:
                    continue

                # choose num_items items
                element = np.array([raw_data[i] for i in indices])

                # create observation tuple
                obs = tuple(element[label, 0] for label in available_labels)

                # find out the
                min_index, middle_index, max_index = element[:, 1].argsort()

                # check that the element with the lowest value occured before the others within one episode
                if episode_start(min_index) > episode_start(middle_index) or \
                        episode_start(min_index) > episode_start(max_index) or \
                        episode_start(middle_index) > episode_start(max_index):
                    continue

                # initialize labels
                label = np.ones(num_items) * -1

                # assign label for min and max value
                label[min_index], label[max_index] = 0., 1.



                min_middle = calculate_distance(min_index, middle_index)
                max_middle = calculate_distance(max_index, middle_index)

                if min_middle > max_middle:
                    label[middle_index] = 1.
                elif max_middle > min_middle:
                    label[middle_index] = 0.
                else:
                    continue

                # check that labeling was successful
                assert (label >=  0).all() and (label <= 1).all()


                item = (obs, label)
                final_dataset.append(item)

                break

        print("%i Tuples of %i Trajectories" % (len(final_dataset), num_items))

        return final_dataset

    def forward_pass(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss

class TripletMarginLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(TripletMarginLoss, self).__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()



    def forward(self, prediction, target):


        return self.loss_fn(prediction, target)

    def prepare_data(self, raw_data, configs):
        # input shape: (num trajectories, 2)
        num_items = 3

        num_trajectories = len(list(raw_data))

        num_tuples = int(num_trajectories / num_items)

        available_labels = np.arange(num_items)

        final_dataset = []

        episode_start = lambda index: element[index, 1] % configs["episode_length"]
        calculate_distance = lambda index_a, index_b: abs(element[index_a, 1] - element[index_b, 1])

        for t in tqdm(range(num_tuples)):

            while True:

                indices = np.random.randint(num_trajectories, size=num_items)

                # check that indices are unique
                if len(set(indices)) != num_items:
                    continue

                # choose num_items items
                element = np.array([raw_data[i] for i in indices])

                # create observation tuple
                obs = tuple(element[label, 0] for label in available_labels)

                # find out the
                min_index, middle_index, max_index = element[:, 1].argsort()

                # check that the element with the lowest value occured before the others within one episode
                if episode_start(min_index) > episode_start(middle_index) or \
                        episode_start(min_index) > episode_start(max_index) or \
                        episode_start(middle_index) > episode_start(max_index):
                    continue

                # initialize labels
                label = np.ones(num_items) * -1

                # assign label for min and max value
                label[min_index], label[max_index] = 0., 1.



                min_middle = calculate_distance(min_index, middle_index)
                max_middle = calculate_distance(max_index, middle_index)

                if min_middle > max_middle + configs["margin"]:
                    label[middle_index] = 1.
                elif max_middle > min_middle + configs["margin"]:
                    label[middle_index] = 0.
                else:
                    continue

                # check that labeling was successful
                assert (label >=  0).all() and (label <= 1).all()


                item = (obs, label)
                final_dataset.append(item)

                break

        print("%i Tuples of %i Trajectories" % (len(final_dataset), num_items))

        return final_dataset

    def forward_pass(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss


RANKING_DICT = {
    "Triplet" : TripletLoss(),
    "TripletMarginLoss" : TripletMarginLoss()
}