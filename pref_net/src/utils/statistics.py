import numpy as np

def return_all_episode_statistics(all_episodes):
    """
    Divides full episodes into n snippets of k length and returns it

    Parameters
    ----------
    file_path : str
        The file location

    Returns
    -------
    numpy array
        array containing the training data
    """
    # print(all_episodes[0])
    print("Num Episodes")
    #assert all_episodes.shape[0] == 2, "True num episodes %s"%all_episodes.shape[0]

    print(all_episodes.shape[0])
    # print(all_episodes.shape[0])
    print("Length Episodes")
    #print(all_episodes[0].shape[0])
    mean = lambda x: sum(x) / len(x)

    #print(all_episodes[0][2])
    '''import matplotlib.pyplot as plt
    indices = range(len(list(all_episodes[0][2])))
    plt.plot(indices, all_episodes[0][2])
    plt.show()'''


    episode_distances = [sum(episode[2]) for episode in all_episodes]

    print("max")
    print(max(episode_distances))
    print("mean")
    print(mean(episode_distances))






