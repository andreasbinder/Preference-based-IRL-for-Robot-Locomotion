import subprocess

#joints = [0,1,2,3,4,5,6,7] * 2
joints = [7]

template = 'python3 gym_mujoco_planar_snake//agents//run_between_v2_and_PPO.py --run1 {joint}'

for joint in joints:
    cmd = template.format(
        joint = str(joint)
    )
    subprocess.run(cmd, cwd=".", shell=True)



#process2 = subprocess.run(['python3 gym_mujoco_planar_snake/agents/run_between_v2_and_PPO.py', '--run1 6'], cwd=".", shell=True)

#process3 = subprocess.run(['ls'], cwd="gym_mujoco_planar_snake/Scripts/")