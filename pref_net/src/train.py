import argparse
import os, datetime
import shutil
from tqdm import tqdm

from gym.core import RewardWrapper
import gym

import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baselines.ppo1 import mlp_policy
from baselines.common import tf_util as U
from baselines.common.running_mean_std import RunningMeanStd
from baselines.bench.monitor import Monitor


from pref_net.src.utils.configs import Configs
from pref_net.src.utils.model_saver import ModelSaverWrapper
from pref_net.src.utils.agent import PPOAgent
from pref_net.src.utils.seeds import set_seeds
import pref_net.src.utils.ranking_util as ranking_util
import pref_net.src.utils.data_util as data_util


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

    def __init__(self, net_save_path):

        # net
        self.net_save_path = net_save_path
        self.net = Net(configs["input_dim"])

        # ranking
        self.loss = ranking_util.RANKING_DICT[configs["ranking_method"]]

        # tensorboard
        self.train_summary_writer = None
        self.val_summary_writer = None
        self.initialize_tensorboard()

    def fit(self, raw_data):

        whole_dataset = self.loss.prepare_data(raw_data, configs)

        self.initialize_tensorboard()

        return self.train(self.net, whole_dataset)

    def train(self, net, dataset):

        train_set, val_set = data_util.split_into_train_val(dataset, configs["split_ratio"])

        optimizer = optim.Adam(net.parameters(), lr=configs["lr"])

        # training
        print("Start Learning")
        for epoch in range(configs["epochs"]):

            running_loss = 0.0

            print("Training")
            for item in tqdm(train_set):
                inputs, label = item

                loss = self.loss.forward_pass(net, optimizer, inputs, label)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            print("Validating")
            running_val_loss = 0.0
            for item in tqdm(val_set):
                inputs, label = item

                loss = self.loss.forward_pass(net, optimizer, inputs, label)

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

        return net

    def initialize_tensorboard(self):

        from tensorboardX import SummaryWriter

        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + mode
        train_log_dir = configs["tensorboard_dir"] + current_time + '/train'
        val_log_dir = configs["tensorboard_dir"] + current_time + '/val'

        self.train_summary_writer = SummaryWriter(train_log_dir)
        self.val_summary_writer = SummaryWriter(val_log_dir)


class MyRewardWrapper(RewardWrapper):

    def __init__(self, venv, nets, ctrl_coeff):
        RewardWrapper.__init__(self, venv)
        self.venv = venv
        self.ctrl_coeff = ctrl_coeff

        # TODO change for one net
        self.nets = [nets]

        self.cliprew = 10.
        self.epsilon = 1e-8
        self.rew_rms = [RunningMeanStd(shape=()) for _ in range(len(self.nets))]

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

    configs_file = Configs(args.path_to_configs)
    configs = configs_file["train"]

    # skip warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print("#" * 15)
    print("Reward Learning")


    set_seeds(configs["seed"])
    print("Seed: %s"%(configs["seed"]))


    mode = configs["ranking_method"]
    print("Method Approach: %s"%(mode))

    SAVE_DIR = os.path.join(configs["save_dir"], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" +  mode)
    os.makedirs(SAVE_DIR)
    shutil.copyfile(args.path_to_configs, os.path.join(SAVE_DIR, "configs_copy.yml"))

    # data
    with open(configs["data_dir"], 'rb') as f:
        train_set = np.load(f, allow_pickle=True)

    '''np.random.shuffle(TRAIN)

    train_set = data_util.generate_dataset_from_full_episodes(TRAIN, configs["subtrajectry_length"],
                                                              configs["subtrajectories_per_episode"],
                                                              configs["max_num_subtrajectories"])'''

    # TODO validate trainset
    '''test_file = "/home/andreas/Documents/pbirl-bachelorthesis/log/original_datasets/inp.npy"
    #test_file = "/home/andreas/Desktop/subtrajectories_15k.npy"
    with open(test_file, 'rb') as f:
        train_set = np.load(f, allow_pickle=True)'''
    # TODO
    #torch.manual_seed(2)

    # reward learning
    rfa = RewardFunctionApproximator(SAVE_DIR)

    net = rfa.fit(train_set)

    torch.save(net.state_dict(), os.path.join(SAVE_DIR, "model"))

    # rl with learnt reward function R_hat
    if configs["run_rl"]:

        print("#" * 15)
        print("Run RL")


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



                sess = U.make_session(num_cpu=1, make_default=False)
                sess.__enter__()
                sess.run(tf.initialize_all_variables())
                with sess.as_default():

                    policy_fn = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                                      ob_space=ob_space,
                                                                                      ac_space=ac_space,
                                                                                      hid_size=64,
                                                                                      num_hid_layers=2
                                                                                      )
                    pi = policy_fn("pi", env.observation_space, env.action_space)


                    agent = PPOAgent(env, pi, policy_fn)
                    agent.learn(configs["num_timesteps"])
