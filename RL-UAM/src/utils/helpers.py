import time
import datetime
import gymnasium as gym
import numpy as np
from torch import nn
import pandas as pd
from typing import Dict, List
import json
import yaml
import string
import random
import os



def get_random_id():
    # Set simulation id
    random.seed(round(time.time()))
    return ''.join(random.choices(string.ascii_uppercase, k=8))

def extract_dict_values(d):
    """
    Extracts all scalar values from a nested dictionary with any depth.
    """
    values_list = []

    for key, value in d.items():
        if isinstance(value, dict):
            values_list.extend(extract_dict_values(value))
        elif isinstance(value, list):
            # Flatten the list
            values_list.extend(flatten_list(value))
        else:
            values_list.append(value)

    return values_list

def flatten_list(lst):
    """
    Recursively flattens a nested list into a flat list.
    """
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def set_seed(seed=None, how='instant'):
    """
    Sets the seed for random number generation.
    :param seed: an optional seed value (int)
    :return: None
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    elif how == 'instant':
        seed= int(time.time() % 1 *1000000)
        random.seed(seed)
        np.random.seed(seed)
        random_int = random.randint(0, 1000)
        random.seed(seed+random_int)
        np.random.seed(seed+random_int)
    else:
        random.seed(42)
        np.random.seed(42)

def ymd_hms():
    """
    Returns the current time in year, month, day, hours, minutes, and seconds.
    """
    return convert_gmt_to_pacific(time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))


def convert_gmt_to_pacific(gmt_time):
    """
    Converts a GMT time string to Pacific time.
    """
    gmt_time = datetime.datetime.strptime(gmt_time, "%Y-%m-%d_%H-%M-%S")
    pacific_time = gmt_time - datetime.timedelta(hours=7)
    return pacific_time.strftime("%Y-%m-%d_%H-%M-%S")

def seconds_to_hms(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%02d:%02d:%02d" % (hours, minutes, seconds)


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Mask function that masks out the actions that are not available in the current state.
    :param env: (gym.Env)
    :return: (np.ndarray)
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        return env.action_mask()

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

def make_env(rl_model, env_config):
    def _init():
        env = gym.make('vertisim', rl_model=rl_model, env_config=env_config)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _init



def read_yaml_file(yaml_path: str) -> dict:
    """
    Reads a yaml file and returns the config as a dictionary.
    :param yaml_path: (str) The path to the yaml file
    :return: (dict) The config
    """
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def read_config_file(file_path: str) -> bool:
    if file_path.endswith(".yaml"):
        rl_config = read_yaml_file(file_path)
    elif file_path.endswith(".json"):
        rl_config = read_json_file(file_path)
    else:
        raise ValueError(f"config path must be either yaml or json file. Received {file_path}")
    return rl_config


def read_json_file(json_path: str) -> Dict:
    """
    Reads a json file and returns the config as a dictionary.
    :param json_path: (str) The path to the json file
    :return: (dict) The config
    """
    with open(json_path, "r") as f:
        content = f.read()
        if content == "":
            raise ValueError(f"The json file {json_path} is empty. You should first start VertiSim container.")
        return json.loads(content)

def convert_to_str(d: Dict):
    """
    Converts all values in a dictionary to strings if they are not instance of int, float, str, bool.
    :param d: (dict) The config
    :return: (dict) The config with all values converted to strings
    """
    for key, value in d.items():
        if not isinstance(value, (int, float, str, bool)):
            d[key] = str(value) 
    return d

from .learning_rate_schedule import linear_schedule

def get_learning_schedule(learning_rate, learning_scheduler):
    if learning_scheduler == "linear":
        return linear_schedule(learning_rate)
    elif learning_scheduler in ["constant", "CosineAnnealingWarmRestarts"]:
        return learning_rate
    else:
        raise ValueError(f"Learning rate schedule ({learning_scheduler}) not implemented")
    


str_to_activation = {
    "ReLU": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "identity": nn.Identity,
    "Mish": nn.Mish
}

def get_vertiport_ids_from_config(config: Dict) -> List[str]:
    """
    Extracts the vertiport ids from the config dictionary.
    :param config: (dict) The config
    :return: (list) The vertiport ids
    """
    return list(config["network_and_demand_params"]["vertiports"].keys())

def get_vertiport_distances(cache_key):
    import os
    # Read the vertiport distances from the csv file
    # Get the current working directory
    cwd = os.getcwd()
    # Set input location
    input_location = os.path.join(cwd, f'cache/{cache_key}_distances.csv')
    vertiport_distances = pd.read_csv(input_location)
    return vertiport_distances
