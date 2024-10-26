import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from typing import Callable


# def make_env(env_id: str, rl_model, rank: int, seed: int = 0) -> Callable:
#     """
#     Utility function for multiprocessed env.

#     :param env_id: (str) the environment ID
#     :param num_env: (int) the number of environment you wish to have in subprocesses
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     :return: (Callable)
#     """

#     def _init() -> gym.Env:
#         env = gym.make(env_id, rl_model)
#         env.reset(seed=seed + rank)
#         return env

#     set_random_seed(seed)
#     return _init


def make_env(env_id, rank, rl_model="PPO"):
    def _init():
        from gymnasium.envs.registration import register
        register(
            id=env_id,
            entry_point='environments.vertisim_env:VertiSimEnvWrapper',
            max_episode_steps=1000000
        )
        env = gym.make(env_id, rl_model=rl_model)
        env.reset()
        return env
    return _init
