import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import os

# take before TREX trajectories and put results into after TREX traj
from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory
from gym_mujoco_planar_snake.common.iterator import Iterator
from gym_mujoco_planar_snake.common.trainer import Trainer
from gym_mujoco_planar_snake.common.documentation import to_excel
import tensorflow.keras as keras
#######################################################################




# load data
def get_data_from_file(path, name="Dataset"):
    return Dataset.load(path=path, name=name)

def get_all_files_from_dir():
    import os
    from itertools import chain
    path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/SubTrajectoryDataset'

    files = os.listdir(path)
    print("Source Files: ", files)

    files = [get_data_from_file(path=path, name=name) for name in files]
    result = list(chain.from_iterable(files))
    print("Total number of files", len(result))

    # assertion here
    assert all([isinstance(i, SubTrajectory) for i in result]), "Not all elements from expected source type"

    return result

def hyperparameters():
    hparams = {
        "lr": 0.0001,
        "epochs": 1,
        "batch_size": 128,
        "dataset_size": None
    }
    return hparams


def return_iterator(batch_size, max_num_traj, num_samples_per_trajectories):
    iterator = Iterator()

    # returns iterator and data
    return iterator.flow(batch_size=5,
                         max_num_traj=100,
                         num_samples_per_trajectories=5,
                         max_num_dir=10)



def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')
    parser.add_argument('--data_dir', help='subtrajectory dataset')
    parser.add_argument('--net_save_path', help='subtrajectory dataset')

    args = parser.parse_args()
    hparams = hyperparameters()
    path = args.data_dir
    net_save_path = args.net_save_path
    data = None # tuple of pairs and labels
    pairs = None # shape: (n_pairs, elements_to_compare=2, length=500, obs_space=27)
    labels = None # shape: (n_pairs,)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #########################################################################
    # TODO:                                                                 #
    #                                                                       #
    #########################################################################
    # general steps


    path= '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/SubTrajectoryDataset/'
    name = 'Dataset500'

    #data = get_data_from_file(path, name)


    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # load data
    trajectories = get_all_files_from_dir()
    #trajectories = get_data_from_file("/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/SubTrajectoryDataset/","Dataset500_Sun Jul 12 17:21:36 2020")

    hparams.update({'dataset_size': len(trajectories)})



    # create pairs of trajectories
    pairs, labels, rewards = Dataset.data_to_pairs(trajectories)
    data = pairs, labels



    trainer = Trainer(hparams)

    trainer.fit_pair_v4(data)
    #trainer.fit_pair_v2(data)

    results = trainer.results

    # document training results
    to_excel(hparams, results)


'''    # train
    model = train(data=data)

    # save reward net
    import os.path as osp
    from time import ctime
    model.save(osp.join(net_save_path, "model"+ctime()+".h5", ))'''

if __name__ == "__main__":
    main()

