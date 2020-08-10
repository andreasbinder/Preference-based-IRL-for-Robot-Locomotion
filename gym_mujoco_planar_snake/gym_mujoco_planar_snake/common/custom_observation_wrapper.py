from gym.core import ObservationWrapper
import numpy as np
import os

# 3000 traj in one run
# 1000 epi, 1000000 ts -> 200 traj per
# if timestep > 200000 then choose 200 random numbers
#

class CustomObservationWrapper(ObservationWrapper):

    def __init__(self, env, sfs):
        ObservationWrapper.__init__(self, env=env)


        self.sfs = sfs
        self.last_e_steps = 0
        self.episodes = 1
        self.observations_list = []
        self.rewards_list = []
        self.trajectories = []



    def store(self, observation, reward):
        self.observations_list.append(observation)
        self.rewards_list.append(reward)

        return observation, reward

    def step(self, action):
        self.last_e_steps += 1
        observation, reward, done, info = self.env.step(action)
        self.store(observation, reward)
        return observation, reward, done, info

    def reset(self, **kwargs):
        #res = self._observation(observation)

        if self.last_e_steps > 0:
            starts = np.random.randint(0, 950, size=200)
            starts.sort()
            for start in starts:
                trajectory = np.array([np.array(self.observations_list)[start:start+50], np.sum(self.rewards_list[start:start+50])])
                print(np.array(self.observations_list)[start:start+50].shape)
                print(len(self.observations_list[start:start+50]))
                print(self.observations_list[start])

                #print(trajectory.shape)

                self.trajectories.append(trajectory)

            self.observations_list = []
            self.rewards_list = []



        if self.last_e_steps % self.sfs == 0 and self.last_e_steps != 0:

            path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/trajectories_on_the_fly'

            with open(os.path.join(path, str(self.last_e_steps)+".npy"), 'wb') as f:
                np.save(f, np.array(self.trajectories))


            #print("Total Steps: ", str(self.last_e_steps))




        return self.env.reset(**kwargs)