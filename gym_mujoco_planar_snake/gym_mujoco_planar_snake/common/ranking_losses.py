# source of inspiration:
# https://www.kaggle.com/davidgasquez/ndcg-scorer

import torch
import numpy as np

class DCG_Loss(object):

    def __init__(self):
        pass

    def __call__(self, y_true, y_score, k=5):
        """Discounted cumulative gain (DCG) at rank K.

        Parameters
        ----------
        y_true : array, shape = [n_samples]
            Ground truth (true relevance labels).
        y_score : array, shape = [n_samples, n_classes]
            Predicted scores.
        k : int
            Rank.

        Returns
        -------
        score : float
        """
        '''order = torch.argsort(y_score)#[::-1]
        y_true = torch.take(y_true, order[:k])

        gain = 2 ** y_true - 1

        len_yt_true = y_true.shape[0]
        discounts = torch.log2(torch.arange(len_yt_true).float() + 2.)'''

        gain = y_score
        i = torch.arange(2,7).float()
        discounts = torch.log2(i)

        return torch.sum(gain / discounts)


true_relevance = torch.tensor([[3, 2, 1, 0, 0]])

# Releveance scores in output order

relevance_score = torch.tensor([[3, 2, 0, 0, 1]])

dcg = DCG_Loss()

loss = dcg(true_relevance, true_relevance)

print(loss)
