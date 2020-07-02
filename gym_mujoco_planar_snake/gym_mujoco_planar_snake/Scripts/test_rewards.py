import numpy as np 


with open('/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/\
gym_mujoco_planar_snake/Scripts/rewards_0722.npy','rb') as f:

    a = np.load(f)

with open('/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/\
gym_mujoco_planar_snake/Scripts/rewards_0246.npy','rb') as f:

    b = np.load(f)

with open('/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/\
gym_mujoco_planar_snake/Scripts/rewards_without_injury.npy','rb') as f:

    c = np.load(f)

with open('/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/\
gym_mujoco_planar_snake/Scripts/rewards_000657.npy','rb') as f:

    d = np.load(f)

print(a.size)

import matplotlib.pyplot as plt

x = [i for i in range(100)]

plt.plot(x, a,'r', x, b, 'b', x, c, 'g', x, d, 'y' )

#plt.plot(x[20:], a[20:])
plt.ylabel('some numbers')
plt.show()