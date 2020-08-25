import numpy as np



def build_trainset(raw_data, ranking):

    data = np.array([raw_data[i:i + ranking] for i, _ in enumerate(raw_data[::ranking])])

    available_labels = np.arange(ranking)


    # TODO only for pairs!!
    num_elements_per_metric = data.shape[0]

    final_dataset = []

    for index in range(num_elements_per_metric):
        element = data[index,:,:]

#        for label in available_labels:

        obs = tuple(element[label,0] for label in available_labels )
        # obs = (element[0,0], element[1,0])


        # TODO do multiclass classification instead
        label = element[:, 1].argmax()
        #label = 0 if element[0,1] > element[1,1] else 1

        # TODO
        '''print(element[:, 1])
        print(label)'''

        '''import sys
        sys.exit()'''



        item = (obs, label)
        final_dataset.append(item)

    #print(len(final_dataset))

    return final_dataset

def split_dataset_for_nets(raw_data, num_nets):
    final_dataset = np.array([raw_data[i:i + num_nets] for i, _ in enumerate(raw_data[::num_nets])])

    return final_dataset


def split_into_train_val(whole_dataset, ratio):
    return whole_dataset[:int(ratio * len(whole_dataset))], \
           whole_dataset[int(ratio * len(whole_dataset)):]


