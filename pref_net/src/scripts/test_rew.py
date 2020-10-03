import numpy as np

path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/Sat Aug 22 18:33:08 2020/trajectories.npy"

paths = [
    "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/Sat Aug 22 18:33:08 2020/trajectories.npy",
    "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/Sat Aug 22 18:46:40 2020/trajectories.npy",
    "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/Sat Aug 22 19:00:16 2020/trajectories.npy",
    "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/Sat Aug 22 19:13:53 2020/trajectories.npy",
    "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/Sat Aug 22 19:27:14 2020/trajectories.npy"

]

cum =  np.zeros((300,))

for path in paths:

    with open(path, 'rb') as f:
        d = np.load(f, allow_pickle=True)

    rew = np.array([r for _, r in d]).reshape((300, 10)).sum(axis=1)
    rew.sort()
    cum += rew
    print(rew.max(), rew.mean())

cum /= len(paths)
print(cum.max(), cum.mean())



path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/improved_runs/crossentropy/Sun Aug 23 13:45:24 20201000000.npy"

with open(path, 'rb') as f:
    d = np.load(f, allow_pickle=True)



num_episodes = int(d[:,0].size / 1000)

rew = d[:,0].reshape((num_episodes,1000)).sum(axis=1)
pred = d[:,1].reshape((num_episodes,1000)).sum(axis=1)

'''import matplotlib.pyplot as plt

rew = np.array([r for _, r in d]).reshape((300, 10)).sum(axis=1)

rew.sort()

print(rew.max())
print(rew.mean())

indices = np.arange(300)

plt.plot(indices, rew)

plt.show()'''
