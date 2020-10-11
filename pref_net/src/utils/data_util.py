import numpy as np

def split_into_train_val(whole_dataset, ratio):

    return whole_dataset[:int(ratio * len(whole_dataset))], \
           whole_dataset[int(ratio * len(whole_dataset)):]



def generate_dataset_from_full_episodes(all_episodes, trajectory_length, n, max_num_subtrajectories):
    """
    Divides full episodes into n snippets of k length and returns it

    Parameters
    ----------
    file_path : str
        The file location
    all_episodes
        index 0 contains list of observations
        index 1 contains the timestep
        index 2 contains list of distance


    Returns
    -------
    numpy array
        array containing the training data
    """

    training_set = []

    for index, episode in enumerate(all_episodes):
        episode_length = len(episode[0])

        # get unique starts
        while True:
            starts = np.random.randint(0, episode_length - trajectory_length, size=n)
            if len(list(np.unique(starts))) == n:
                break

        # TODO advantage for sorting?
        #starts.sort()

        training_set.append(np.array(
            [(np.array(episode[0][start:start + trajectory_length]), episode[1] + start) for start in starts]))


    training_set = np.concatenate(training_set)

    if len(list(training_set)) > max_num_subtrajectories:
        training_set = np.array([training_set[index] for index in range(max_num_subtrajectories)])

    return training_set

