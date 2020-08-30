from gym_mujoco_planar_snake.common.env_wrapper import MyMonitor
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ppo_dir', type=str)
parser.add_argument('--vf_dir', type=str)
args = parser.parse_args()



MyMonitor.compare_initial_and_improved_reward(args.ppo_dir, args.vf_dir)