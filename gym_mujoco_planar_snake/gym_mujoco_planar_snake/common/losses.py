import numpy as np



def build_trainset(raw_data, ranking):

    data = np.array([raw_data[i:i + ranking] for i, _ in enumerate(raw_data[::ranking])])


    # TODO only for pairs!!
    num_pairs = data.shape[0]

    final_dataset = []

    for index in range(num_pairs):
        pair = data[index,:,:]
        label = 0 if pair[0,1] > pair[1,1] else 1
        obs = (pair[0,0], pair[1,0])
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


