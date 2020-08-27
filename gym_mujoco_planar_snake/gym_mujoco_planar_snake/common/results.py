import numpy as np
import matplotlib.pyplot as plt

class DataFrame(object):

    def __init__(self,
                 true_rew,
                 preds=None,
                 initial = True):


        if initial:
            # true rew has form (num_runs, collected summed returns per run)
            # eg (2,20) means two agents with 2 checkpoints)
            self.true_rew = true_rew.sum(axis=0) / true_rew.shape[0]
            self.preds = None
        else:
            # reshape


            self.true_rew = true_rew.sum(axis=0) / true_rew.shape[0]
            self.preds = preds.sum(axis=0) / preds.shape[0]



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







