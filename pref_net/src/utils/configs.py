import yaml

class Configs():
    """
        class that stores run configurations
    """

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            self.data = yaml.load(f)

    def __getitem__(self, key):
        return self.data["key"]


    # TODO
    def get_extrapolation_configs(self):
        return self.data["extrapolation"]

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

    def get_split_ratio(self):
        return self.data["reward_learning"]["split_ratio"]

    def get_input_dim(self):
        return self.data["reward_learning"]["input_dim"]

    def get_ranking_method(self):
        return self.data["reward_learning"]["ranking_method"]

