import subprocess


template = 'python3 gym_mujoco_planar_snake/agents/run_ppo.py --num-timesteps 300000 --seed {seed}'

for seed in range(5):
    cmd = template.format(
        seed=int(seed)

    )

    subprocess.run(cmd, cwd=".", shell=True)
