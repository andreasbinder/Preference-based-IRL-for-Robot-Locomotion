import torch
import torch.nn as nn
import numpy as np


from tqdm import tqdm


def build_test_margin_triplet_trainset(raw_data, ranking=3):
    # input shape: (num trajectories, 2)
    np.random.shuffle(raw_data)

    #data = np.array([raw_data[i:i + ranking] for i, _ in enumerate(raw_data[::ranking])])
    #print(raw_data.shape)

    num_trajectories = raw_data.shape[0]

    num_final_trajectories = 6666

    available_labels = np.arange(ranking)

    margin = 200000

    final_dataset = []
    for _ in range(num_final_trajectories):

        while True:
            indices = np.random.randint(num_trajectories, size=ranking)

            if len(set(indices)) == ranking:

                element = np.array([raw_data[i] for i in indices])

                obs = tuple(element[label, 0] for label in available_labels)

                min_index, middle_index, max_index = element[:, 1].argsort()

                label = np.array([-1.] * ranking)

                label[min_index], label[max_index] = 0., 1.

                compare_value = element[middle_index, 1]

                min_middle = abs(element[min_index, 1] - compare_value)
                max_middle = abs(element[max_index, 1] - compare_value)

                if  min_middle > max_middle + margin:
                    label[middle_index] = 1.
                elif max_middle > min_middle + margin:
                    label[middle_index] = 0.
                else:
                    # TODO continue instead
                    label[middle_index] = -1.

                if label[middle_index] != -1.:
                    item = (obs, label)
                    #final_dataset.append(item)

                    break

        #traj = np.array([raw_data[i] for i in indices])


        final_dataset.append(item)


    #data = np.array(data)






    # TODO only for pairs!!
    '''    num_elements_per_metric = data.shape[0]

    final_dataset = []

    for index in range(num_elements_per_metric):
        element = data[index,:,:]

#        for label in available_labels:

        obs = tuple(element[label,0] for label in available_labels )
        # obs = (element[0,0], element[1,0])


        # TODO do multiclass classification instead
        #label = element[:, 1].argmax()
        #label = 0 if element[0,1] > element[1,1] else 1
        min_index, middle_index, max_index = element[:,1].argsort()

        label = np.array([-1.] * ranking)

        label[min_index], label[max_index] = 0., 1.

        compare_value = element[middle_index,1]

        label[middle_index] = 1. if abs(element[min_index,1] - compare_value) > abs(element[max_index,1] - compare_value) else 0.




        item = (obs, label)
        final_dataset.append(item)'''

    #print(len(final_dataset))
    '''print(len(final_dataset))
    print(final_dataset[0][0][0].shape)
    print(final_dataset[0][1].shape)'''

    print("%i Tuples of %i Trajectories"%(len(final_dataset), ranking))



    return final_dataset

# für initial pair
def build_custom_ranking_trainset(raw_data, ranking=3):

    np.random.shuffle(raw_data)

    data = np.array([raw_data[i:i + ranking] for i, _ in enumerate(raw_data[::ranking])])

    available_labels = np.arange(ranking)

    # TODO only for pairs!!
    num_elements_per_metric = data.shape[0]

    final_dataset = []

    for index in range(num_elements_per_metric):
        element = data[index, :, :]

        #        for label in available_labels:

        obs = tuple(element[label, 0] for label in available_labels)
        # obs = (element[0,0], element[1,0])

        # TODO do multiclass classification instead
        label = element[:, 1].argsort()



        item = (obs, label)
        final_dataset.append(item)



    print("%i Tuples of %i Trajectories" % (len(final_dataset), ranking))

    return final_dataset



class InitialPairLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(InitialPairLoss, self).__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()



    def forward(self, prediction, target):


        return self.loss_fn(prediction, target)

    def prepare_data(self, raw_data, configs):
        # input shape: (num trajectories, 2)
        ranking = 2

        data = np.array([raw_data[i:i + ranking] for i, _ in enumerate(raw_data[::ranking])])

        available_labels = np.arange(ranking)

        # TODO only for pairs!!
        num_elements_per_metric = data.shape[0]

        final_dataset = []

        for index in range(num_elements_per_metric):
            element = data[index, :, :]

            #        for label in available_labels:

            obs = tuple(element[label, 0] for label in available_labels)
            # obs = (element[0,0], element[1,0])

            # TODO do multiclass classification instead
            label = element[:, 1].argsort()

            item = (obs, label)
            final_dataset.append(item)

        print("%i Tuples of %i Trajectories" % (len(final_dataset), ranking))

        return final_dataset

    def forward_pass(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        # TOOO float?
        y = torch.tensor(label).unsqueeze(0).float()

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss



class PairLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(PairLoss, self).__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()



    def forward(self, prediction, target):


        return self.loss_fn(prediction, target)

    def prepare_data(self, raw_data, configs):
        # input shape: (num trajectories, 2)
        num_items = 2

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
                min_index, max_index = element[:, 1].argsort()

                # check that the element with the lowest value occured before the others within one episode
                if episode_start(min_index) > episode_start(max_index):
                    continue

                # initialize labels
                label = np.ones(num_items) * -1

                # assign label for min and max value
                label[min_index], label[max_index] = 0., 1.


                # check that labeling was successful
                assert (label >=  0).all() and (label <= 1).all()


                item = (obs, label)
                final_dataset.append(item)

                break

        print("%i Tuples of %i Trajectories" % (len(final_dataset), num_items))

        return final_dataset

    def forward_pass(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        # TOOO float?
        y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss

class NaiveTripletLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(NaiveTripletLoss, self).__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()



    def forward(self, prediction, target):


        return self.loss_fn(prediction, target)

    def prepare_data(self, raw_data, configs):
        # input shape: (num trajectories, 2)
        ranking = 3

        data = np.array([raw_data[i:i + ranking] for i, _ in enumerate(raw_data[::ranking])])

        available_labels = np.arange(ranking)


        num_elements_per_metric = data.shape[0]

        final_dataset = []

        for index in range(num_elements_per_metric):
            element = data[index, :, :]

            #        for label in available_labels:

            obs = tuple(element[label, 0] for label in available_labels)
            # obs = (element[0,0], element[1,0])

            # TODO do multiclass classification instead
            # label = element[:, 1].argmax()
            # label = 0 if element[0,1] > element[1,1] else 1
            min_index, middle_index, max_index = element[:, 1].argsort()

            label = np.array([-1.] * ranking)

            label[min_index], label[max_index] = 0., 1.

            compare_value = element[middle_index, 1]

            label[middle_index] = 1. if abs(element[min_index, 1] - compare_value) > abs(
                element[max_index, 1] - compare_value) else 0.

            item = (obs, label)
            final_dataset.append(item)

        print("%i Tuples of %i Trajectories" % (len(final_dataset), ranking))

        return final_dataset

    def forward_pass(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss

class InitialTripletMarginLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(InitialTripletMarginLoss, self).__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):


        return self.loss_fn(prediction, target)

    def prepare_data(self, raw_data, configs):
        # input shape: (num trajectories, 2)

        ranking = 3

        # data = np.array([raw_data[i:i + ranking] for i, _ in enumerate(raw_data[::ranking])])
        # print(raw_data.shape)

        num_trajectories = raw_data.shape[0]

        num_final_trajectories = int(num_trajectories / ranking)

        available_labels = np.arange(ranking)



        final_dataset = []
        for _ in range(num_final_trajectories):

            while True:
                indices = np.random.randint(num_trajectories, size=ranking)

                if len(set(indices)) == ranking:

                    element = np.array([raw_data[i] for i in indices])

                    obs = tuple(element[label, 0] for label in available_labels)

                    min_index, middle_index, max_index = element[:, 1].argsort()

                    label = np.array([-1.] * ranking)

                    label[min_index], label[max_index] = 0., 1.

                    compare_value = element[middle_index, 1]

                    min_middle = abs(element[min_index, 1] - compare_value)
                    max_middle = abs(element[max_index, 1] - compare_value)

                    if min_middle > max_middle + configs["margin"]:
                        label[middle_index] = 1.
                    elif max_middle > min_middle + configs["margin"]:
                        label[middle_index] = 0.
                    else:
                        continue

                    item = (obs, label)
                    final_dataset.append(item)
                    break

        print("%i Tuples of %i Trajectories" % (len(final_dataset), ranking))

        return final_dataset

    def forward_pass(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss


'''class NaiveTripletLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(NaiveTripletLoss, self).__init__()

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
'''

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

class FaceNetLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(FaceNetLoss, self).__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.distance_function = nn.PairwiseDistance()

        #self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.distance_function)

        self.criterion = nn.TripletMarginLoss()

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

                '''# check that the element with the lowest value occured before the others within one episode
                if episode_start(min_index) > episode_start(middle_index) or \
                        episode_start(min_index) > episode_start(max_index) or \
                        episode_start(middle_index) > episode_start(max_index):
                    continue'''

                # labels
                label =  np.array([min_index, middle_index, max_index])
                '''# initialize labels
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
                assert (label >=  0).all() and (label <= 1).all()'''


                item = (obs, label)
                final_dataset.append(item)

                break

        print("%i Tuples of %i Trajectories" % (len(final_dataset), num_items))

        return final_dataset

    def forward_pass(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        #y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        '''print(rewards)
        print(label)'''


        min_index, middle_index, max_index  = label
        #loss = self.loss_fn(rewards, y)

        anchor, positive, negative = rewards[:,max_index].unsqueeze(0), rewards[:,middle_index].unsqueeze(0), \
                                     rewards[:,min_index].unsqueeze(0)

        loss = self.criterion(anchor, positive, negative)

        return loss

class ComplexInitialTripletLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(ComplexInitialTripletLoss, self).__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()



    def forward(self, prediction, target):


        return self.loss_fn(prediction, target)

    def prepare_data(self, raw_data, configs):
        # input shape: (num trajectories, 2)
        ranking = 3

        final_dataset = []

        calculate_distance = lambda x, y: abs(triplet[x, 1] - triplet[y, 1])

        i = 0
        label = None

        while raw_data.shape[0] > 100:

            triplet = raw_data[:3]

            min_index, middle_index, max_index = triplet[:, 1].argsort()

            min_distance = calculate_distance(min_index, middle_index)
            max_distance = calculate_distance(max_index, middle_index)
            if min_distance + configs["margin"] < max_distance:
                label = np.array([-1.] * ranking)
                label[min_index], label[middle_index], label[max_index] = 0., 0., 1.
                obs = tuple(triplet[label, 0] for label in range(ranking))
                item = (obs, label)
                final_dataset.append(item)
                raw_data = raw_data[3:]

                i += 1
                if i % 200 == 0:
                    print(label)
                    print(triplet[:, 1])

            elif max_distance + configs["margin"] < min_distance:
                label = np.array([-1.] * ranking)
                label[min_index], label[middle_index], label[max_index] = 0., 1., 1.
                obs = tuple(triplet[label, 0] for label in range(ranking))
                item = (obs, label)
                final_dataset.append(item)
                raw_data = raw_data[3:]

                i += 1
                if i % 200 == 0:
                    print(label)
                    print(triplet[:, 1])

            else:
                np.random.shuffle(raw_data)



        print("%i Tuples of %i Trajectories" % (len(final_dataset), ranking))

        return final_dataset

    def forward_pass(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss

class ComplexInitialPairLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(ComplexInitialPairLoss, self).__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()



    def forward(self, prediction, target):


        return self.loss_fn(prediction, target)

    def prepare_data(self, raw_data, configs):
        # input shape: (num trajectories, 2)
        ranking = 2

        data = np.array([raw_data[i:i + ranking] for i, _ in enumerate(raw_data[::ranking])])

        available_labels = np.arange(ranking)

        # TODO only for pairs!!
        num_elements_per_metric = data.shape[0]

        final_dataset = []

        for index in range(num_elements_per_metric):
            element = data[index, :, :]

            #        for label in available_labels:

            obs = tuple(element[label, 0] for label in available_labels)
            # obs = (element[0,0], element[1,0])

            # TODO do multiclass classification instead
            label = element[:, 1].argsort()

            item = (obs, label)
            final_dataset.append(item)

        print("%i Tuples of %i Trajectories" % (len(final_dataset), ranking))

        return final_dataset

    def forward_pass(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        # TOOO float?
        y = torch.tensor(label).unsqueeze(0).float()

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss

# TODO test
class ComplexEpisodeTripletLoss(nn.Module):

    """ Triplet Ranking Loss """

    def __init__(self):
        super(ComplexEpisodeTripletLoss, self).__init__()

        self.loss_fn = nn.BCEWithLogitsLoss()



    def forward(self, prediction, target):


        return self.loss_fn(prediction, target)

    def prepare_data(self, raw_data, configs):
        # input shape: (num trajectories, 2)
        ranking = 3

        final_dataset = []

        calculate_distance = lambda x, y: abs(triplet[x, 1] - triplet[y, 1])

        episode_start = lambda index: triplet[index, 1] % configs["episode_length"]

        i = 0
        label = None

        while raw_data.shape[0] > 100:

            triplet = raw_data[:3]

            min_index, middle_index, max_index = triplet[:, 1].argsort()

            if episode_start(min_index) > episode_start(middle_index) or \
                    episode_start(min_index) > episode_start(max_index) or \
                    episode_start(middle_index) > episode_start(max_index):
                np.random.shuffle(raw_data)
                continue

            min_distance = calculate_distance(min_index, middle_index)
            max_distance = calculate_distance(max_index, middle_index)
            if min_distance < max_distance:
                label = np.array([-1.] * ranking)
                label[min_index], label[middle_index], label[max_index] = 0., 0., 1.
                obs = tuple(triplet[label, 0] for label in range(ranking))
                item = (obs, label)
                final_dataset.append(item)
                raw_data = raw_data[3:]

                i += 1
                if i % 200 == 0:
                    print(label)
                    print(triplet[:, 1])

            elif max_distance < min_distance:
                label = np.array([-1.] * ranking)
                label[min_index], label[middle_index], label[max_index] = 0., 1., 1.
                obs = tuple(triplet[label, 0] for label in range(ranking))
                item = (obs, label)
                final_dataset.append(item)
                raw_data = raw_data[3:]

                i += 1
                if i % 200 == 0:
                    print(label)
                    print(triplet[:, 1])

            else:
                np.random.shuffle(raw_data)



        print("%i Tuples of %i Trajectories" % (len(final_dataset), ranking))

        return final_dataset

    def forward_pass(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss



RANKING_DICT = {
    "Pair" : PairLoss(),
    "Triplet" : TripletLoss(),
    "TripletMargin" : TripletMarginLoss(),
    # TODO Tests
    "NaiveTriplet" : NaiveTripletLoss(),
    "InitialTripletMargin" : InitialTripletMarginLoss(),
    #"NaiveTriplet" : NaiveTripletLoss(),
    "FaceNet" : FaceNetLoss(),
    "InitialPair" : InitialPairLoss(),
    "ComplexInitialTriplet" : ComplexInitialTripletLoss(),
    "ComplexEpisodeTriplet": ComplexEpisodeTripletLoss()
}