import os

path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs'

print(os.listdir(path))

for item in os.listdir(path):

    if len(os.listdir(os.path.join(path,item))) == 0:
        os.rmdir(os.path.join(path,item))
    else:
        for item2 in os.listdir(os.path.join(path,item)):
            pass
