from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory

# load data
def get_data_from_file(path, name="Dataset"):
    return Dataset.load(path=path, name=name)

def get_all_files_from_dir(path):
    import os
    from itertools import chain
    path = '/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/TrajectoryDataset'

    files = os.listdir(path)
    print("Source Files: ", files)

    '''import shutil
    for name in files:
        if hasattr(get_data_from_file(path=path, name=name)[0], "time_step") is False:
            shutil.move(os.path.join(path, name), "/home/andreas/Desktop/Speicher")'''


    #files = [get_data_from_file(path=path, name=name) for name in files ]


    files = [get_data_from_file(path=path, name=name) for name in files if hasattr(get_data_from_file(path=path, name=name)[0], "time_step") ]
    result = list(chain.from_iterable(files))
    print("Total number of files", len(result))

    #Dataset(result).save(path=path, name="Dataset_"+str(len(result)))



    # assertion here
    assert all([isinstance(i, SubTrajectory) for i in result]), "Not all elements from expected source type"

    return result


def get_model_dir(env_id, name):
    model_dir = osp.join(logger.get_dir(), 'models')
    os.mkdir(model_dir)
    model_dir = ModelSaverWrapper.gen_model_dir_path(model_dir, env_id, name)
    logger.log("model_dir: %s" % model_dir)
    return model_dir

def return_iterator(batch_size, max_num_traj, num_samples_per_trajectories):
    iterator = Iterator()

    # returns iterator and data
    return iterator.flow(batch_size=5,
                         max_num_traj=100,
                         num_samples_per_trajectories=5,
                         max_num_dir=10)

