from gymnasium.envs.registration import register
import gymnasium as gym

register(
    id='vertisim',
    entry_point='src.environments.vertisim_env:VertiSimEnvWrapper',
    max_episode_steps=1000000
)
