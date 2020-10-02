import numpy as np

path1 = "/tmp/test_create_result/train.npy"

with open(path1, 'rb') as f:
    d1 = np.load(f, allow_pickle=True)


path1 = "/tmp/test_create_result/extrapolate.npy"

with open(path1, 'rb') as f:
    d2 = np.load(f, allow_pickle=True)

pass