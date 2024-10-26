import os
import sys
# Add the RL-UAM folder to the current path
current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(current_path + "/RL-UAM")

from stable_baselines3.common.env_checker import check_env
from src.environments.vertisim_env import VertiSimEnvWrapper
import gymnasium as gym
from src.utils.helpers import read_config_file


env_config = read_config_file("./configs/config_rl.json")

env = gym.make('vertisim', rl_model="MaskablePPO", env_config=env_config)
# It will check your custom environment and output additional warnings if needed
check_env(env=env, warn=True)