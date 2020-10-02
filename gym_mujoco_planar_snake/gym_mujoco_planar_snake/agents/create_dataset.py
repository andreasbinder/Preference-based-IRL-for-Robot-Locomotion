import numpy as np

import os.path as osp
import tensorflow as tf

from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy
#from baselines.ppo1.pposgd_simple import learn
from baselines import logger

import gym, logging
from gym.core import ObservationWrapper
import os

from gym_mujoco_planar_snake.common.env_wrapper import prepare_env
from gym_mujoco_planar_snake.common import my_tf_util
from gym_mujoco_planar_snake.common.misc_util import Configs
from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

from baselines.common.mpi_running_mean_std import RunningMeanStd

from baselines.common.distributions import make_pdtype

from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque


def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]



class PPOAgent(object):

    def __init__(self, env, pi):

        self.env = env
        self.pi = pi

    def learn(self, num_timesteps):

        self._learn(self.env, self.pi,
                    max_timesteps=num_timesteps,
                    timesteps_per_actorbatch=2048,
                    clip_param=0.2,
                    entcoeff=0.0,
                    optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                    gamma=0.99, lam=0.95,
                    schedule='linear',
                    )

    # also for benchmark
    # run untill done
    def run_environment_episode(self, env, pi, seed, model_file, max_timesteps, render, stochastic):
        number_of_timestep = 0
        done = False

        # load model
        my_tf_util.load_state(model_file)

        # set seed
        # set_global_seeds(seed)
        env.seed(seed)

        info_collector = InfoCollector(env)

        obs = env.reset()

        cum_reward = []

        observations = []

        # max_timesteps is set to 1000
        while (not done) and number_of_timestep < max_timesteps:

            action, _ = pi.act(stochastic, obs)

            obs, reward, done, info = env.step(action)

            observations.append(obs)

            cum_reward.append(info["distance_delta"])

            info['seed'] = seed
            info['env'] = env.spec.id

            info["obs"] = obs

            # add info
            info_collector.add_info(info)

            # render
            if render:
                env.render()

            number_of_timestep += 1

        return observations, cum_reward
        # return info_collector


    def _learn(self, env, pi, *,
              timesteps_per_actorbatch,  # timesteps per actor per update
              clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
              optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
              gamma, lam,  # advantage estimation
              max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
              callback=None,  # you can do anything in the callback, since it takes locals(), globals()
              adam_epsilon=1e-5,
              schedule='constant'  # annealing for stepsize parameters (epsilon and adam)
              ):
        # Setup losses and stuff
        # ----------------------------------------
        ob_space = env.observation_space
        ac_space = env.action_space
        # pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
        oldpi = policy_fn("oldpi", ob_space, ac_space)  # Network for old policy
        atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                                shape=[])  # learning rate multiplier, updated with schedule
        clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

        ob = U.get_placeholder_cached(name="ob")
        ac = pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-entcoeff) * meanent

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
        surr1 = ratio * atarg  # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
        total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        var_list = pi.get_trainable_variables()
        lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
        adam = MpiAdam(var_list, epsilon=adam_epsilon)

        assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in
                                                        zipsame(oldpi.get_variables(), pi.get_variables())])
        compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

        U.initialize()
        adam.sync()

        # Prepare for rollouts
        # ----------------------------------------
        seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

        assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                    max_seconds > 0]) == 1, "Only one time constraint permitted"

        while True:
            if callback: callback(locals(), globals())
            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif max_episodes and episodes_so_far >= max_episodes:
                break
            elif max_iters and iters_so_far >= max_iters:
                break
            elif max_seconds and time.time() - tstart >= max_seconds:
                break

            if schedule == 'constant':
                cur_lrmult = 1.0
            elif schedule == 'linear':
                cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            else:
                raise NotImplementedError

            logger.log("********** Iteration %i ************" % iters_so_far)

            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

            assign_old_eq_new()  # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = []  # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(newlosses)
            meanlosses, _, _ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names):
                logger.record_tabular("loss_" + name, lossval)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
            lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.dump_tabular()


class ModelSaverWrapper(ObservationWrapper):

    def __init__(self, env, model_dir, save_frequency_steps):
        ObservationWrapper.__init__(self, env=env)


        self.save_frequency_steps = save_frequency_steps
        self.total_steps = 0
        self.total_steps_save_counter = 0
        self.total_episodes = 0


        self.model_dir = model_dir

    def reset(self, **kwargs):
        self.total_episodes += 1

        # todo start saving after 100k timesteps
        if self.total_steps_save_counter == self.save_frequency_steps or self.total_steps == 1:
            buffer = 9

            len_total_steps = len(str(self.total_steps))

            zeros = (buffer - len_total_steps) * "0"

            file_name = osp.join(self.model_dir, zeros + str(self.total_steps))

            my_tf_util.save_state(file_name)

            logger.log('Saved model to: ' + file_name)

            self.total_steps_save_counter = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps += 1
        self.total_steps_save_counter += 1

        return self.env.step(action)

    def observation(self, observation):
        return observation






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_configs', type=str,
                        default="gym_mujoco_planar_snake/agents/configurations/configs.yml")
    args = parser.parse_args()

    configs = Configs(args.path_to_configs)
    configs = configs.data["create_dataset"]

    # hyperparameters
    # TODO migrate to configs
    TRAJECTORY_LENGTH = 50 #50 100
    EPISODE_MAX_LENGTH = 1000
    RENDER = False
    sfs = 5000
    ENV_ID = 'Mujoco-planar-snake-cars-angle-line-v1'
    NUM_TIMESTEPS = 1500000
    SAVE_DIR = "/tmp/test_create0/"
    MAX_EPISODE_STEPS = 1000
    FULL_EPISODES = []
    EXTRAPOLATE_NAME = "create_test.npy"
    SAVE_PATH = "/tmp/test_create_result/"
    percentage = 0.5


    # seeds
    set_global_seeds(configs["seed"])

    # skip warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    with tf.variable_scope(str(configs["variable_scope"])):

        env = gym.make(ENV_ID)
        env = ModelSaverWrapper(env, SAVE_DIR, sfs)

        info_dict_collector = InfoDictCollector(env)


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

            agent = PPOAgent(env, pi)
            agent.learn(NUM_TIMESTEPS)

            list = [x[:-len(".index")] for x in os.listdir(SAVE_DIR) if x.endswith(".index")]
            list.sort(key=str.lower)
            model_files = [osp.join(SAVE_DIR, ele) for ele in list]
            num_models = len(model_files)


            print('available models: ', len(model_files))

            for model_file in model_files:

                logger.log("load model_file: %s" % model_file)

                # run one episode
                time_step = int(model_file[-9:])
                # TODO specify target velocity
                # only takes effect in angle envs
                env.unwrapped.metadata['target_v'] = 0.1

                # observations, cum_reward
                observations, cum_reward = agent.run_environment_episode(env,
                                                                         pi,
                                                                         configs["seed"],
                                                                         model_file,
                                                                         MAX_EPISODE_STEPS,
                                                                         render=RENDER,
                                                                         stochastic=False
                                                                         )


                trajectories = np.array([(observations, time_step, cum_reward)])


                FULL_EPISODES.append(np.array(trajectories))

            #FULL_EPISODES = np.concatenate(FULL_EPISODES)

            # Split in train and extrapolation data

            FULL_EPISODES = np.concatenate(FULL_EPISODES)
            FACTOR = int(num_models * percentage)
            TRAIN = FULL_EPISODES[:FACTOR]
            EXTRAPOLATE = FULL_EPISODES[FACTOR:]

            TRAIN_NAME = "train.npy"
            EXTRAPOLATE_NAME = "extrapolate.npy"

            with open(os.path.join(SAVE_PATH, TRAIN_NAME), 'wb') as f:
                np.save(f, np.array(TRAIN))

            with open(os.path.join(SAVE_PATH, EXTRAPOLATE_NAME), 'wb') as f:
                np.save(f, np.array(EXTRAPOLATE))

            from gym_mujoco_planar_snake.benchmark.plot_results import return_all_episode_statistics

            return_all_episode_statistics(TRAIN)

            from gym_mujoco_planar_snake.common.data_util import generate_dataset_from_full_episodes

            train_set = generate_dataset_from_full_episodes(TRAIN, 50, 100)

            name_for_default = "subtrajectories.npy"

            with open(os.path.join(SAVE_PATH, name_for_default), 'wb') as f:
                np.save(f, np.array(train_set))


    # TODO
    # run env als Teil von agent




