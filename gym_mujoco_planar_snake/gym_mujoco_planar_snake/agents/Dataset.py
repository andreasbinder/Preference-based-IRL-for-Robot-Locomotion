# path
# reward action observation
# joint to be
# original Methods: gen_trag, prebuilt, sample

class TrajectoryBase(object):
    def __init__(self, rewards=[], actions=[], observations=[]):
        self.rewards = rewards
        self.actions = actions
        self.observations = observations
        self.cum_reward = sum(self.rewards)

class SubTrajectory(TrajectoryBase):
    def __init__(self, rewards=[], actions=[], observations=[]):
        super(SubTrajectory, self).__init__(rewards, actions, observations)


class Trajectory(TrajectoryBase):
    def __init__(self, path, elements, time_step, fixed_joints = None):
        super(Trajectory, self).__init__()
        self.elements = elements
        self.pure_data_file = None
        self.path = path
        self.fixed_joints = [] if fixed_joints is None else fixed_joints
        self.rewards = []
        self.actions = []
        self.observations = []
        self.cum_reward = sum(self.rewards)
        self.time_step = time_step

    def toString(self):
        template = ' Trajectoryinfo(time_step={time_step}, cum_reward={cum_reward}) '

        return template.format(
            time_step=self.time_step,
            cum_reward=self.cum_reward)

class Dataset(object):
    def __init__(self, list_trajectories):
        self.list_trajectories = list_trajectories
        self.size = len(self.list_trajectories)

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

    def save(self, path, name="Dataset"):
        import os.path as osp
        import pickle
        output = osp.join(path, name)

        with open(output, 'wb') as f:
            pickle.dump(self.list_trajectories, f)

    def load(self, path, name="Dataset"):
        import os.path as osp
        import pickle
        inp = osp.join(path, name)

        with open(inp, 'rb') as f:
            return pickle.load(f)






