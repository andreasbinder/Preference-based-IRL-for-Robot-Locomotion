import numpy as np

path1 = "/home/andreas/Desktop/2020-10-03_01-12-01/train.npy"

with open(path1, 'rb') as f:
    d1 = np.load(f, allow_pickle=True)


path1 = "/home/andreas/Desktop/2020-10-03_01-12-01/extrapolate.npy"

with open(path1, 'rb') as f:
    d2 = np.load(f, allow_pickle=True)

pass