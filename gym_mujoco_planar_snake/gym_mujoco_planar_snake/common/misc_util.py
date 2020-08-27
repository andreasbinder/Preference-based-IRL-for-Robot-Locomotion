from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory
import json
import yaml

class Configs():
    """
        class that stores run configurations
    """

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            self.data = yaml.load(f)



        #self.update(yaml_path)

    def get_all_configs(self):
        return self.data

    def get_env_id(self):
        return self.data["ppo"]["general"]["env_id"]

    def get_seed(self):
        return self.data["ppo"]["general"]["seed"]

    def get_log_dir(self):
        return self.data["ppo"]["general"]["log_dir"]




    def get_create_initial_trajectories(self):
        return self.data["ppo"]["initial"]["create_trajectories"]

    def get_trajectory_length(self):
        return self.data["ppo"]["initial"]["trajectory_length"]

    def get_num_traj_per_episode(self):
        return self.data["ppo"]["initial"]["num_traj_per_episode"]

    def get_num_initial_agents(self):
        return self.data["ppo"]["initial"]["num_agents"]

    def get_num_initial_timesteps(self):
        return self.data["ppo"]["initial"]["num_timesteps"]

    def get_save_initial_checkpoints(self):
        return self.data["ppo"]["initial"]["save_checkpoints"]

    def get_initial_sfs(self):
        return self.data["ppo"]["initial"]["sfs"]


    def get_validate_learned_reward(self):
        return self.data["ppo"]["improved"]["validate_learned_reward"]

    def get_num_improved_agents(self):
        return self.data["ppo"]["improved"]["num_agents"]

    def get_num_improved_timesteps(self):
        return self.data["ppo"]["improved"]["num_timesteps"]

    def get_save_improved_checkpoints(self):
        return self.data["ppo"]["improved"]["save_checkpoints"]

    def get_improved_sfs(self):
        return self.data["ppo"]["improved"]["sfs"]


    def get_num_nets(self):
        return self.data["reward_learning"]["num_nets"]

    def get_ranking_approach(self):
        return self.data["reward_learning"]["hparams"]["ranking_approach"]

    def get_learning_rate(self):
        return self.data["reward_learning"]["hparams"]["lr"]

    def get_epochs(self):
        return self.data["reward_learning"]["hparams"]["epochs"]

    def get_ctrl_coeff(self):
        return self.data["ppo"]["improved"]["ctrl_coeff"]



    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, yaml_path):
        """Loads parameters from json file"""
        with open(yaml_path) as f:
            self.config_file = yaml.load(yaml_path)
            #self.__dict__.update(self.config_file)

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

