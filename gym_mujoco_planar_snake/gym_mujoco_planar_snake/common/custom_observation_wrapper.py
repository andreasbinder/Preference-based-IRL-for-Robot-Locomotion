from gym.core import ObservationWrapper
import numpy as np
import os
from time import ctime

# 3000 traj in one run
# 1000 epi, 1000000 ts -> 200 traj per
# if timestep > 200000 then choose 200 random numbers
#

class GenTrajWrapper(ObservationWrapper):

    def __init__(self, env, path, max_timesteps, trajectory_length, num_traj_per_epoch):
        ObservationWrapper.__init__(self, env=env)

        self.max_timesteps = max_timesteps
        #self.sfs = sfs
        self.last_e_steps = 0
        self.episodes = 1
        self.observations_list = []
        self.rewards_list = []
        self.trajectories = []
        self.trajectory_length = trajectory_length

        self.path = path
        self.num_traj_per_epoch = num_traj_per_epoch
        self.name = "trajectories.npy"

        self.saved = False


    def store(self, observation, reward):
        self.observations_list.append(observation)
        self.rewards_list.append(reward)

        return observation, reward

    def step(self, action):
        self.last_e_steps += 1
        observation, reward, done, info = self.env.step(action)
        #print(done)
        self.store(observation, reward)
        return observation, reward, done, info

    def reset(self, **kwargs):

        if self.last_e_steps > 0:

            #assert False, self.last_e_steps

            if self.trajectory_length == 50:
                starts = np.random.randint(0, 950, size=self.num_traj_per_epoch)
                starts.sort()
            else:
                starts = [0]

            #print(starts)create trajectories
            for start in starts:
                # tuple of form ([50,27], [50])
                trajectory = np.array(self.observations_list)[start:start+self.trajectory_length], \
                             np.sum(self.rewards_list[start:start+self.trajectory_length])

                self.trajectories.append(trajectory)

            self.observations_list = []
            self.rewards_list = []

            #print(self.last_e_steps)

            if self.last_e_steps >= self.max_timesteps and not self.saved:
                # print("in save")
                path = self.path

                with open(os.path.join(path, self.name), 'wb') as f:
                    np.save(f, np.array(self.trajectories))
                    self.trajectories = []

                self.saved = True


        return self.env.reset(**kwargs)