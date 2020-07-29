# https://github.com/openai/gym/blob/master/gym/core.py
from gym.core import ActionWrapper
import numpy as np

class AcWrapper(ActionWrapper):

    def __init__(self, env, clip_value):
        ActionWrapper.__init__(self, env)

        self.env = env

        self.clip_value = np.abs(clip_value)

        self.neg_clip_value = np.negative(self.clip_value)


    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):

        fixed = [4,5,6,7]

        for joint in fixed:
            action[joint] = np.clip(action[joint], self.neg_clip_value, self.clip_value)

        #action = np.clip(action, self.neg_clip_value, self.clip_value)

        return action

    def reverse_action(self, action):
        raise NotImplementedError