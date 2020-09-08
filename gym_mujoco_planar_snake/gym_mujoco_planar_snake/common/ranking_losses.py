# source of inspiration:
# https://www.kaggle.com/davidgasquez/ndcg-scorer

import torch
import torch.nn as nn
import numpy as np

from scipy.spatial.distance import cdist

# TEMPLATE: https://discuss.pytorch.org/t/custom-rank-loss-function/81885
class MultiLabelDCGLoss(nn.Module):

    """ Mean Reciprocal Rank Loss """

    def __init__(self):
        super(MultiLabelDCGLoss, self).__init__()

    def forward(self, prediction, target):
        target = target[0]
        prediction = prediction[0]

        gain = prediction[target]

        # TODO normalize


        i = torch.arange(2, 2 + target.shape[0]).float()
        discounts = torch.log2(i)

        '''print(gain)
        print(discounts)'''

        #assert False, "Success"

        sigmoid = nn.Sigmoid()

        #gain = sigmoid(gain)

        dcg_sore = torch.sum(gain / discounts)



        dcg_sore = sigmoid(dcg_sore)

        return -dcg_sore



class ExplicitRankingLoss(object):
    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def __call__(self, prediction, target):



        target = target[0]
        prediction = prediction[0]

        target_values = prediction[target]

        #print(target_values)


        #loss_unnormlized = torch.square(torch.pow(target_values - prediction, 2))

        #print(loss_unnormlized)

        # TODO normalize
        #mean = loss_unnormlized.mean()
        #var = loss_unnormlized.var()
        #loss = (loss_unnormlized - mean) / var


        #loss = torch.sum(loss_unnormlized)

        loss = self.loss_fn(prediction, target_values)

        return loss











