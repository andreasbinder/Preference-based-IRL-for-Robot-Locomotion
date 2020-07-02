# Usage

There are four steps to be performed

1. First you have to start by running PPO with some joints fixed
```bash
python3 run_PPO.py --run1 0 1
```
The most important argument are the ones going from --run1 to --run5. You can define the joints you want to fix with indexes ranging from 0 to 7. In case you want a run without impairment you have to set the run parameter to -1. The results are stored in the ```log/ ``` directory. The runs are categorized by run number und the indexes of the fixed joints, so eg. run2_56 indicates to be the third run with joint 5 and 6 as fixed ones.

2. Those trajectories 

## Changes made
* added fixed_joints as argument to pposgd_simple_di
