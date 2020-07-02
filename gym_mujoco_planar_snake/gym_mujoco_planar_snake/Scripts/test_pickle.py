import pickle

path = '/home/andreas/LRZ_Sync+Share/Thesis/ICML2019-TREX/mujoco/learner/demo_models/halfcheetah/checkpoints/00001'

with open(path + '.env_stat.pkl', 'rb') as f:
    s = pickle.load(f)

    print(s)
