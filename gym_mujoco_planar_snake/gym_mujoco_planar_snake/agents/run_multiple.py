import subprocess
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--num_samples_per_trajectories', type=int, default=int(10))
parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')
parser.add_argument('--log_dir', help='log directory', default='gym_mujoco_planar_snake/log/clip_test/')
parser.add_argument('--save_path', help='save directory for prepared subtrajectories',
                    default='gym_mujoco_planar_snake/log/SubTrajectoryDataset')

args = parser.parse_args()
#joints = [0,1,2,3,4,5,6,7] * 2
joints = [7]

template = 'python3 gym_mujoco_planar_snake//agents//run_PPO.py --run1 {joint}'

for joint in joints:
    cmd = template.format(
        joint = str(joint)
    )
    subprocess.run(cmd, cwd=".", shell=True)



#process2 = subprocess.run(['python3 gym_mujoco_planar_snake/agents/run_between_v2_and_PPO.py', '--run1 6'], cwd=".", shell=True)

#process3 = subprocess.run(['ls'], cwd="gym_mujoco_planar_snake/Scripts/")