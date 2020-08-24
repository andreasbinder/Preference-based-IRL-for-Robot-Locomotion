import numpy as np
import matplotlib.pyplot as plt

class DataFrame(object):

    def __init__(self,
                 true_rew,
                 predictions=None,
                 file_path=None,
                 env = "Mujoco-planar-snake-cars-angle-line-v1",
                 initial = True):

        #print(true_rew)



        self.env = env
        self.true_rew = None
        self.predictions = None

        if file_path != None:
            with open(file_path, 'rb') as f:
                data = np.load(f, allow_pickle=True)


        if initial:
            #self.true_rew = true_rew.reshape((true_rew.shape[1], true_rew.shape[0])).sum(axis=1)
            self.true_rew = true_rew.sum(axis=0) / true_rew.shape[0]



            '''# TODO standardize length
            episode_length = 1000 if self.env == "Mujoco-planar-snake-cars-angle-line-v1" else None
            self.num_episodes = int(data[:, 0].size / episode_length)

            self.true_rew = data[:, 0].reshape((self.num_episodes, episode_length)).sum(axis=1)
            self.predictions = data[:, 1].reshape((self.num_episodes, episode_length)).sum(axis=1)'''
        else:
            self.true_rew = true_rew.sum(axis=0) / true_rew.shape[0]
            self.predictions = predictions.sum(axis=0) / predictions.shape[0]

            print(self.true_rew.shape)
            print(self.predictions.shape)






    def get_mean_true_rew(self):
        print("Mean True Reward: ", self.true_rew.mean())
        return self.true_rew.mean()

    def get_max_true_rew(self):
        print("Max True Reward: ", self.true_rew.max())
        return self.true_rew.max()

    def plot_true_vs_pred(self):

        indices = np.arange(self.true_rew.size)

        plt.plot(indices, self.true_rew, color='b')
        # äplt.plot(indices, pred, color='r')

        plt.show()


def shot_results():
    pass

if __name__ == "__main__":
    path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/improved_runs/crossentropy/ensemble_v1/ppo_Sun Aug 23 12:34:39 2020/Sun Aug 23 13:45:24 20201000000.npy"

    df = DataFrame(path)

    df.plot_true_vs_pred()



