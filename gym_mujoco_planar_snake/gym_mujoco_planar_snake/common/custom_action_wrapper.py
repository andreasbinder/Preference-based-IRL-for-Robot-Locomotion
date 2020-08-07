# https://github.com/openai/gym/blob/master/gym/core.py
from gym.core import ActionWrapper
import numpy as np

class ClipActionWrapper(ActionWrapper):

    def __init__(self, env, clip_value, joints):
        ActionWrapper.__init__(self, env)

        self.env = env

        self.joints = joints

        self.clip_value = np.abs(clip_value)

        self.neg_clip_value = np.negative(self.clip_value)


    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):

        for joint in self.joints:
            action[joint] = np.clip(action[joint], self.neg_clip_value, self.clip_value)

        return action

    def reverse_action(self, action):
        raise NotImplementedError