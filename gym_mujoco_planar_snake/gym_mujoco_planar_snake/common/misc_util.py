from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory
import json
import yaml

class Configs():
    """
        class that stores run configurations
    """

    def __init__(self, yaml_path):
        self.update(yaml_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, yaml_path):
        """Loads parameters from json file"""
        with open(yaml_path) as f:
            self.config_file = yaml.load(yaml_path)
            #self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


# load data
def get_data_from_file(path, name="Dataset"):
    return Dataset.load(path=path, name=name)

def get_all_files_from_dir(path='gym_mujoco_planar_snake/log/TrajectoryDataset'):
    import os
    from itertools import chain


    files = os.listdir(path)
    files = [file for file in files if os.path.isfile(os.path.join(path, file))]
    print("Source Files: ", files)


    files = [get_data_from_file(path=path, name=name) for name in files if hasattr(get_data_from_file(path=path, name=name)[0], "time_step") ]
    result = list(chain.from_iterable(files))
    print("Total number of files", len(result))



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

