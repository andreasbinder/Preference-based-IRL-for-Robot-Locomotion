import argparse
import os, datetime
import shutil

from gym.core import RewardWrapper
import gym



import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn


from baselines.ppo1 import mlp_policy
from baselines.common import tf_util as U
from baselines.common.running_mean_std import RunningMeanStd
from baselines.bench.monitor import Monitor

import src.common.data_util as data_util
from src.common.misc_util import Configs
from src.common.ensemble import Ensemble as RFA
from src.utils.model_saver import ModelSaverWrapper
from src.utils.agent import PPOAgent
from src.utils.seeds import set_seeds


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def cum_return(self, traj):
        # return of one traj
        sum_rewards = 0

        r = self.model(traj)

        sum_rewards += torch.sum(r)

        return sum_rewards

    def forward(self, trajs):
        cum_rs = [self.cum_return(traj).unsqueeze(0) for traj in trajs]

        return torch.cat(tuple(cum_rs), 0).unsqueeze(0)


class RewardFunctionApproximator(object):

    def __init__(self, configs, net_save_path):
        self.num_nets = configs.get_num_nets()
        self.ranking_approach = configs.get_ranking_approach()
        self.net_save_path = net_save_path

        #
        # TODO save training results
        #self.hparams = Params(args.hparams_path)
        self.lr = configs.get_learning_rate()
        self.epochs = configs.get_epochs()
        self.split_ratio = configs.get_split_ratio()
        self.input_dim = configs.get_input_dim()
        self.ranking_method = configs.get_ranking_method()


        # TODO for every mode
        self.loss_fn, self.preprocess_fn, self.forward_pass_fn = self.initialize_mode(self.ranking_method)


        self.train_summary_writer = None
        self.val_summary_writer = None
        self.initialize_tensorboard()

    def initialize_mode(self, mode):
        print("Mode %s" % mode)
        if mode == "pair":
            return (
                nn.BCEWithLogitsLoss(),
                lambda raw_data: data.build_custom_ranking_trainset(raw_data, ranking=2),
                lambda net, optimizer, inputs, label: self.forward_pass_pair_ranking(net, optimizer, inputs, label)
                    )
        if mode == "explicit":
            return (
                custom_losses.ExplicitRankingLoss(),
                lambda raw_data: data.build_custom_ranking_trainset(raw_data, ranking=self.ranking_approach),
                lambda net, optimizer, inputs, label: self.forward_pass_explicit_ranking(net, optimizer, inputs, label)
                    )
        if mode == "dcg":
            return (
                custom_losses.MultiLabelDCGLoss(),
                lambda raw_data: data.build_custom_ranking_trainset(raw_data, ranking=self.ranking_approach),
                lambda net, optimizer, inputs, label: self.forward_pass_dcg_ranking(net, optimizer, inputs, label)
                    )
        if mode == "triplet":
            return (
                nn.BCEWithLogitsLoss(),
                lambda raw_data: data.build_custom_triplet_trainset(raw_data, ranking=3),
                lambda net, optimizer, inputs, label: self.forward_pass_custom_triplet(net, optimizer, inputs, label)
                    )
        if mode == "triplet_sigmoid":
            return (
                nn.BCELoss(),
                lambda raw_data: data.build_custom_triplet_trainset(raw_data, ranking=3),
                lambda net, optimizer, inputs, label: self.forward_pass_custom_triplet_sigmoid(net, optimizer, inputs, label)
                    )

        if mode == "triplet_margin":
            return (
                nn.BCEWithLogitsLoss(),
                lambda raw_data: data.build_test_margin_triplet_trainset(raw_data, ranking=3),
                lambda net, optimizer, inputs, label: self.forward_pass_custom_triplet(net, optimizer, inputs, label)
                    )
        if mode == "triplet_better":
            return (
                nn.BCEWithLogitsLoss(),
                lambda raw_data: data.build_test_better_triplet_trainset(raw_data, ranking=3),
                lambda net, optimizer, inputs, label: self.forward_pass_custom_triplet(net, optimizer, inputs, label)
            )





    def preprocess_data(self, raw_data):
        # choose
        data = self.preprocess_fn(raw_data)

        #data = split_dataset_for_nets(data, self.num_nets)

        return data

    def fit(self, raw_data):

        whole_dataset = self.preprocess_data(raw_data)


        nets = [Net(self.input_dim)] * self.num_nets


        #nets = [Net(self.input_dim), Net(self.input_dim), Net(self.input_dim), Net(self.input_dim),Net(self.input_dim)]
        #final_dataset = [raw_data[i:i + self.num_nets] for i, _ in enumerate(raw_data[::self.num_nets])]

        #self.train(0, nets[0], whole_dataset[0])
        import numpy as np

        for index, net in enumerate(nets):
            self.initialize_tensorboard()
            self.train(index, net, whole_dataset)

            #np.random.shuffle(whole_dataset)
            whole_dataset = self.preprocess_data(raw_data)



    def train(self, index, net, dataset):

        train_set, val_set = data.split_into_train_val(dataset, self.split_ratio)


        optimizer = optim.Adam(net.parameters(), lr=self.lr)

        # training
        print("Start Learning")
        for epoch in range(self.epochs):

            running_loss = 0.0

            print("Training")
            for item in tqdm(train_set):
                inputs, label = item

                loss = self.forward_pass_fn(net, optimizer, inputs, label)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            print("Validating")
            running_val_loss = 0.0
            for item in tqdm(val_set):
                inputs, label = item

                loss = self.forward_pass_fn(net, optimizer, inputs, label)

                # print statistics
                running_val_loss += loss.item()

            template = 'Epoch {}, Loss: {:10.4f}, Validation Loss: {:10.4f}'
            stats = template.format(
                epoch + 1,
                running_loss / len(train_set),
                running_val_loss / len(val_set)
            )
            print(stats)

            self.train_summary_writer.add_scalar('train_loss', running_loss / len(train_set), epoch)
            self.val_summary_writer.add_scalar('val_loss', running_val_loss / len(val_set), epoch)

        self.save(index=index, net=net)

    def forward_pass_explicit_ranking(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss

    def forward_pass_pair_ranking(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0).float()

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss

    def forward_pass_dcg_ranking(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss

    def forward_pass_custom_triplet(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0)

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss

    def forward_pass_custom_triplet_sigmoid(self, net, optimizer, inputs, label):

        trajs = [torch.from_numpy(inp).float() for inp in inputs]

        y = torch.tensor(label).unsqueeze(0).float()

        optimizer.zero_grad()

        rewards = net(trajs)

        loss = self.loss_fn(rewards, y)

        return loss

    def save(self, index, net):

        torch.save(net.state_dict(), os.path.join(self.net_save_path, "model_" + str(index)))

    @staticmethod
    def load(log_dir, num_nets, input_dim=27):

        nets = []
        for index in range(num_nets):
            path = os.path.join(log_dir, "model_" + str(index))
            net = Net(input_dim)
            net.load_state_dict(torch.load(path))
            nets.append(net)

        return nets

    def initialize_tensorboard(self):

        from tensorboardX import SummaryWriter

        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        train_log_dir = configs["tensorboard_dir"] + current_time + '/train'
        val_log_dir = configs["tensorboard_dir"] + current_time + '/val'


        self.train_summary_writer = SummaryWriter(train_log_dir)
        self.val_summary_writer = SummaryWriter(val_log_dir)



class MyRewardWrapper(RewardWrapper):

    def __init__(self, venv, nets, ctrl_coeff):
        RewardWrapper.__init__(self, venv)

        self.venv = venv
        self.ctrl_coeff = ctrl_coeff
        self.nets = nets

        self.cliprew = 10.
        self.epsilon = 1e-8
        self.rew_rms = [RunningMeanStd(shape=()) for _ in range(len(nets))]


    def step(self, action):


        obs, rews, news, infos = self.venv.step(action)

        r_hats = 0.

        for net, rms in zip(self.nets, self.rew_rms):
            # Preference based reward
            with torch.no_grad():
                pred_rews = net.cum_return(torch.from_numpy(obs).float())

            r_hat = pred_rews.item()

            # Normalization only has influence on predicted reward
            rms.update(np.array([r_hat]))
            r_hat = np.clip(r_hat / np.sqrt(rms.var + self.epsilon), -self.cliprew, self.cliprew)

            # Sum-up each models' reward
            r_hats += r_hat

        pred = r_hats / len(self.nets) - self.ctrl_coeff * np.sum(action ** 2)

        return obs, pred, news, infos

    def reset(self, **kwargs):

        return self.venv.reset(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_configs', type=str,
                        default="/home/andreas/Documents/pbirl-bachelorthesis/pref_net/configs.yml")
    args = parser.parse_args()

    configs = Configs(args.path_to_configs)

    configs.data["reward_learning"] = configs.data["train"]

    configs_tmp = configs
    configs = configs.data["train"]

    # skip warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    set_seeds(configs["seed"])

    with open(configs["data_dir"], 'rb') as f:
        TRAIN = np.load(f, allow_pickle=True)


    train_set = data_util.generate_dataset_from_full_episodes(TRAIN, configs["subtrajectry_length"], configs["subtrajectories_per_episode"])

    SAVE_DIR = os.path.join(configs["save_dir"], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(SAVE_DIR)
    shutil.copyfile(args.path_to_configs, os.path.join(SAVE_DIR, "configs_copy.yml"))


    ensemble = RFA(configs_tmp, SAVE_DIR)

    ensemble.fit(train_set)

    net = RFA.load(SAVE_DIR, 1)

    # RL
    ENV_ID = 'Mujoco-planar-snake-cars-angle-line-v1'

    indices = [str(index) for index in range(1, configs["num_agents"] + 1)]

    for index in indices:

        with tf.variable_scope(index):

            SAVE = os.path.join(SAVE_DIR, index)

            env = gym.make(ENV_ID)

            env.seed(configs["seed"])

            env = ModelSaverWrapper(env, SAVE, configs["save_sequency"])
            env = Monitor(env, SAVE)

            env = MyRewardWrapper(env, net, configs["ctrl_coeff"])



            policy_fn = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                                ob_space=ob_space,
                                                                                ac_space=ac_space,
                                                                                hid_size=64,
                                                                                num_hid_layers=2
                                                                                )
            pi = policy_fn("pi", env.observation_space, env.action_space)

            sess = U.make_session(num_cpu=1, make_default=False)
            sess.__enter__()
            sess.run(tf.initialize_all_variables())
            with sess.as_default():

                agent = PPOAgent(env, pi, policy_fn)
                agent.learn(configs["num_timesteps"])








