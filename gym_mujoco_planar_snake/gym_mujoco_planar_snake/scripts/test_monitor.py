from baselines.bench.monitor import Monitor, load_results, get_monitor_files

path = "/home/andreas/Desktop"

df = load_results(path)
#m = df.as_matrix(columns="r")
print(df["r"])
print(df["r"].to_numpy().mean())

path_2 = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/ppo_ensemble_Aug_29_22:13:20/agent_0"
n_f = get_monitor_files(path_2)
print(n_f)
