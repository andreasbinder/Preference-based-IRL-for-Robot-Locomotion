
import os.path as osp

from gym.core import ObservationWrapper

from pref_net.src.utils import my_tf_util


from baselines import logger


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

