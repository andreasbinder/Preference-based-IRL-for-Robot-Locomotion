from gym.envs.registration import register

# used mujoco model
register(
    id='Mujoco-planar-snake-cars-angle-line-v1',
    entry_point='pref_net.src.envs.mujoco_15:MujocoPlanarSnakeCarsAngleLineEnv',
    max_episode_steps=1000, #TODO parameter responsible for episode length
    reward_threshold=6000.0,
)
