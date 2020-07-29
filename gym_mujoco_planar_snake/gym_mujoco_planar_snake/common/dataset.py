# original Methods: gen_trag, prebuilt, sample
import numpy as np

class TrajectoryBase(object):
    def __init__(self, cum_reward, observations, time_step=0):

        self.observations = observations
        self.cum_reward = cum_reward
        self.time_step = time_step



class SubTrajectory(TrajectoryBase):
    def __init__(self, cum_reward, observations, time_step):
        super(SubTrajectory, self).__init__(cum_reward=cum_reward, observations=observations, time_step=time_step)


class Trajectory(TrajectoryBase):
    def __init__(self, path, pure_data_file, time_step, fixed_joints = None):
        super(Trajectory, self).__init__(cum_reward=0, observations=[], time_step=time_step)
        #self.elements = elements
        self.pure_data_file = pure_data_file
        self.path = path
        self.fixed_joints = [] if fixed_joints is None else fixed_joints
        self.rewards = []
        self.observations = []
        self.cum_reward = sum(self.rewards)
        self.time_step = time_step

        # not sure if needed
        # self.actions = []


    def toString(self):
        template = ' Trajectoryinfo(time_step={time_step}, cum_reward={cum_reward}) '

        return template.format(
            time_step=self.time_step,
            cum_reward=self.cum_reward)

class Dataset(object):
    def __init__(self, list_trajectories, loss="pair"):
        self.list_trajectories = list_trajectories
        self.size = len(self.list_trajectories)
        self.losses = {
            "pair" : 2,
            "triplet" : 3,
            "list" : 100 # tbd
        }
        self.loss_step = self.losses[loss]
        self.horizon_length = None

    def __getitem__(self, index):

        return self.list_trajectories[index]

    def __iter__(self):
        return iter(self.list_trajectories)

    def __next__(self):
        return  next(self.list_trajectories)

    def __len__(self):
        return len(self.list_trajectories)

    def remove(self, item):
        self.list_trajectories.remove(item)

    def copy(self):
        return Dataset(self.list_trajectories.copy())

    def toString(self):
        trajectory_infos = [traj.toString() for traj in self.list_trajectories]
        print("".join(trajectory_infos))

    def shuffle(self):
        from random import shuffle
        return shuffle(self.list_trajectories)

    def sample(self, num_samples=1):
        from random import sample
        from random import choices
        return sample(self.list_trajectories, k=num_samples)
        #return self.list_trajectories[randint(0,self.size - 1)]

    # https://stackoverflow.com/questions/51461051/how-to-load-only-selected-objects-from-a-pickle-file
    # maybe give possibility to save each item at a time
    def save(self, path, name="Dataset"):
        import os.path as osp
        import pickle
        output = osp.join(path, name)

        with open(output, 'wb') as f:
            pickle.dump(self.list_trajectories, f)

    @staticmethod
    def load(path, name="Dataset"):
        import os.path as osp
        import pickle
        inp = osp.join(path, name)

        with open(inp, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_pairs(data):
        from random import shuffle
        shuffle(data)

        return [data[i:i + 2] for i, _ in enumerate(data[::2])]

    @staticmethod
    def get_pairs_v2(data):
        from random import shuffle
        shuffle(data)

        return [data[i:i + 2] for i, _ in enumerate(data[::2])]

    @staticmethod
    def get_triplets(data):
        pass
        # (anchor, positive, negative)



    @staticmethod
    def data_to_pairs(data):
        #data = self.list_trajectories

        # TODO include time_step information

        # TODO dont compare the same trajectories


        pairs, labels, rewards = [], [], []
        demos = Dataset.get_pairs(data)

        for demo_pair in demos:
            obs1, obs2 = demo_pair[0].observations, demo_pair[1].observations
            if isinstance(obs1, list):
                obs1 = np.array(obs1)
            if isinstance(obs2, list):
                obs2 = np.array(obs2)

            pairs.append(np.array([obs1, obs2]))
            rew1, rew2 = demo_pair[0].cum_reward, demo_pair[1].cum_reward
            # meaning: index of superior trajecotory
            label = 0 if rew1 > rew2 else 1
            rewards.append(np.array([rew1, rew2]))

            #labels.append(np.array([label]))
            labels.append(label)

        #pairs = [pair for pair in pairs if pair.shape == (2, 50, 27)]

        # arr = np.array(pairs)

        return np.array(pairs), np.array(labels), np.array(rewards)

# should not be called directory
if __name__ == "__main__":
    print("Should not be called directly")







