import numpy as np
import matplotlib.pyplot as plt

class DataFrame(object):

    def __init__(self,
                 true_rew,
                 preds=None,
                 initial = True):

        # TODO episode length
        episode_length = 1000

        if initial:
            # true rew has form (num_runs, collected summed returns per run)
            # eg (2,20) means two agents with 2 checkpoints)
            # TODO divide by num agents
            self.true_rew = true_rew.sum(axis=0) / true_rew.shape[0] #

            print(true_rew.shape)
            print(self.true_rew.shape)


            self.preds = None
        else:
            # reshape
            '''print(true_rew.shape)
            print(preds.shape)'''



            self.true_rew = true_rew.sum(axis=0) / true_rew.shape[0]
            self.preds = preds.sum(axis=0) / preds.shape[0]

            '''print(true_rew.shape)
            print(preds.shape)'''

            num_episodes = int(self.true_rew.size / episode_length)

            #print(num_episodes)

            # TODO replace 20 with / (length_episde / length_trajectory)
            self.true_rew = self.true_rew.reshape(num_episodes,episode_length).sum(axis=1) / 20
            self.preds = self.preds.reshape(num_episodes,episode_length).sum(axis=1) / 20





            # TODO reshape and sum over steps inside one episode!!


    def get_mean_true_rew(self):
        #print("Mean True Reward: ", self.true_rew.mean())
        return self.true_rew.mean()

    def get_max_true_rew(self):
        #print("Max True Reward: ", self.true_rew.max())
        return self.true_rew.max()

    def get_mean_preds(self):
        #print("Mean True Reward: ", self.true_rew.mean())
        return self.preds.mean()

    def get_max_preds(self):
        #print("Max True Reward: ", self.true_rew.max())
        return self.preds.max()

    # TODO
    def plot_true_vs_pred(self):

        indices = np.arange(self.true_rew.size)

        plt.plot(indices, self.true_rew, color='b')
        # äplt.plot(indices, pred, color='r')

        plt.show()


def show_results(initial_df, improved_df):

    print("Mean:")
    print("Initial: %f , Improved %f"%(initial_df.get_mean_true_rew(), improved_df.get_mean_true_rew()))

    print("Max:")
    print("Initial: %f , Improved %f" % (initial_df.get_max_true_rew(), improved_df.get_max_true_rew()))







