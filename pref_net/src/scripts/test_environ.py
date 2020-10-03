import gym


from baselines.common import  tf_util as U


import src

import tensorflow as tf




from baselines.ppo1 import mlp_policy
from baselines.ppo1.pposgd_simple import learn

policy_fn = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                            ob_space=ob_space,
                                                                            ac_space=ac_space,
                                                                            hid_size=64,
                                                                            num_hid_layers=2
                                                                            )

ENV_ID = 'Mujoco-planar-snake-cars-angle-line-v1'

env = gym.make(ENV_ID)

sess = U.make_session(num_cpu=1, make_default=False)
sess.__enter__()
sess.run(tf.initialize_all_variables())
with sess.as_default():
    learn(env, policy_fn,
                       max_timesteps=2000,
                       timesteps_per_actorbatch=2048,
                       clip_param=0.2,
                       entcoeff=0.0,
                       optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                       gamma=0.99, lam=0.95,
                       schedule='linear',
                       )


