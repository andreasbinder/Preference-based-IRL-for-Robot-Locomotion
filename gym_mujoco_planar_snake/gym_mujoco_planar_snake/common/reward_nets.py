import torch
import torch.nn as nn

class PairNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(50*27, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #x = traj.permute(0, 3, 1, 2)  # get into NCHW format
        # compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        r = self.model(traj)
        #print(r)
        sum_rewards += torch.sum(r)
        #print(sum_rewards)
        sum_abs_rewards += torch.sum(torch.abs(r))
        #print(sum_abs_rewards)
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''

        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j

class PairBCENet(nn.Module):
    def __init__(self, input_dim=27):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.sigmoid = nn.Sigmoid()


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #x = traj.permute(0, 3, 1, 2)  # get into NCHW format
        # compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        r = self.model(traj)
        #print(r)
        sum_rewards += torch.sum(r)
        #print(sum_rewards)
        sum_abs_rewards += torch.sum(torch.abs(r))
        #print(sum_abs_rewards)
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''

        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)

        diff = cum_r_i - cum_r_j

        x = self.sigmoid(diff)



        return x, abs_r_i + abs_r_j


class TripletNet(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO bestes Modell raussuchen
        self.model = nn.Sequential(
            nn.Linear(50*27, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #x = traj.permute(0, 3, 1, 2)  # get into NCHW format
        # compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        r = self.model(traj)
        #print(r)
        sum_rewards += torch.sum(r)
        #print(sum_rewards)
        sum_abs_rewards += torch.sum(torch.abs(r))
        #print(sum_abs_rewards)
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_h, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''

        cum_r_h, abs_r_h = self.cum_return(traj_h)
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_h.unsqueeze(0), cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_h + abs_r_i + abs_r_j
        #return torch.cat((cum_r_h.unsqueeze(0), cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_h + abs_r_i + abs_r_j


class SingleStepPairNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(27, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        '''self.model = nn.Sequential(
            nn.Linear(27, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )'''




    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #x = traj.permute(0, 3, 1, 2)  # get into NCHW format
        # compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        r = self.model(traj)
        #print(r)
        sum_rewards += torch.sum(r)
        #print(sum_rewards)
        #assert False, sum_rewards
        sum_abs_rewards += torch.sum(torch.abs(r))
        #print(sum_abs_rewards)
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''

        '''print(traj_i.shape)
        print(traj_j.shape)'''

        # TODO
        '''traj_i  = traj_i.view(50,27)
        traj_j = traj_j.view(50, 27)'''

        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j


class SingleObservationTripletNet(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO bestes Modell raussuchen
        self.model = nn.Sequential(
            nn.Linear(27, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #x = traj.permute(0, 3, 1, 2)  # get into NCHW format
        # compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        r = self.model(traj)
        #print(r)
        sum_rewards += torch.sum(r)
        #print(sum_rewards)
        sum_abs_rewards += torch.sum(torch.abs(r))
        #print(sum_abs_rewards)
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_h, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''

        cum_r_h, abs_r_h = self.cum_return(traj_h)
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_h.unsqueeze(0), cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_h + abs_r_i + abs_r_j
        #return torch.cat((cum_r_h.unsqueeze(0), cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_h + abs_r_i + abs_r_j


