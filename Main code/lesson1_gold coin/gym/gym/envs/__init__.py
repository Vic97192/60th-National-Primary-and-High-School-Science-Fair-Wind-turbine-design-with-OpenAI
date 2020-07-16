from gym.envs.classic_control.grid_mdp import GridEnv

gym.envs.register (
    id='GridWorld-v0',
    entry_point='envs.classic_control:GridEnv',
    max_episode_steps=200,
    reward_threshold=100.0,
)