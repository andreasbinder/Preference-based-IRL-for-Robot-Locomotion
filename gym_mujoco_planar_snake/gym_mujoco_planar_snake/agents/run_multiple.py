import subprocess
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--injured', type=bool, default=False)
parser.add_argument('--run1', nargs='*', type=int, default=None)

args = parser.parse_args()

joints = args.run1

save_frequency_steps = 1000

#joints = [0,1,2,3,4,5,6,7] * 2
joints = [0]

template = 'python3 gym_mujoco_planar_snake//agents//run_PPO.py --run1 {joint} --run{sfs}'

for joint in joints:
    cmd = template.format(
        joint = str(joint),
        sfs = save_frequency_steps
    )
    subprocess.run(cmd, cwd=".", shell=True)
