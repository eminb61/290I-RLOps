import os
import sys
import glob
# Add the RL-UAM folder to the current path
current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(current_path + "/RL-UAM")
sys.path.append(current_path + "/RL-UAM/src/rl_models/sb3")

from src.environments.vertisim_env import VertiSimEnvWrapper
from sb3_contrib import MaskablePPO
from src.rl_models.sb3.sb3_contrib_local.ppo_mask_recurrent.ppo_mask_recurrent import MaskableRecurrentPPO
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.utils.helpers import read_config_file, make_env, mask_fn
import time
import argparse

def load_model(model_path, model_type):
    if model_type == "MaskablePPO":
        return MaskablePPO.load(model_path)
    elif model_type == "MaskableRecurrentPPO":
        return MaskableRecurrentPPO.load(model_path)
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported")


class CustomDummyVecEnv(DummyVecEnv):
    def action_mask(self):
        # Retrieve action masks from all environments
        return [env.action_mask() for env in self.envs]

def predict(env_config, model_path, norm_path, model_type):


    # # Create the environment
    env = gym.make('vertisim', rl_model=model_type, env_config=env_config)
    env = ActionMasker(env, mask_fn)    
    env = Monitor(env)
    env = CustomDummyVecEnv([lambda: env])

    # Load normalization statistics
    if norm_path:
        norm_env = VecNormalize.load(norm_path, env)
        norm_env.training = False
        norm_env.norm_obs = True
        norm_env.norm_reward = True
        env = norm_env

    # Load the model
    model = load_model(model_path, model_type)

    print("Making prediction")
    obs = env.reset()
    terminated_count = 0
    while terminated_count < 1:
        # Retrieve current action mask
        action_masks = env.action_mask()
        action, _states = model.predict(obs, action_masks=action_masks)
        new_state, reward, done, _ = env.step(action)
        terminated_count += int(done)
        if done:
            env.close()


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", "-ec", type=str, help="The environment config file path")
    parser.add_argument("--model_path", "-m", type=str, help="Trained RL model to use")
    parser.add_argument("--model_type", "-t", type=str, help="Type of the RL model (e.g., MaskablePPO, MaskableRecurrentPPO)")
    parser.add_argument("--norm_path", "-n", type=str, help="Normalization statistics path")
    parser.add_argument("--batch_run", "-br", action="store_true", help="Run the simulation in batch")
    parser.add_argument("--passenger_schedule_folder", "-paxp", type=str, help="The folder containing the passenger schedules", default=None)
    args = parser.parse_args()

    env_config = read_config_file(args.env_config)
    if not args.model_type:
        args.model_type = env_config['sim_params']['algorithm']

    if args.batch_run:
        # Folder containing the CSV files
        paxp = args.passenger_schedule_folder

        passenger_schedule_folder = "test_data"

        # Get all CSV files in the folder
        csv_files = glob.glob(os.path.join(passenger_schedule_folder, "*.csv"))

        # Iterate over each CSV file and update the config for each run
        for csv_file in csv_files:
            print(f"Running simulation for: {csv_file}")

            file_name = os.path.basename(csv_file)
            new_csv_path = os.path.join(paxp, file_name)
            
            # Update the passenger_schedule_file_path in the env_config
            env_config['network_and_demand_params']['passenger_schedule_file_path'] = new_csv_path
            env_config['sim_params']['algorithm'] = args.model_type
            env_config['sim_params']['logging'] = False
            
            # Run the simulation
            predict(env_config=env_config, model_path=args.model_path, model_type=args.model_type, norm_path=args.norm_path)

    else:
        # Run the simulation
        env_config['sim_params']['algorithm'] = args.model_type
        predict(env_config=env_config, model_path=args.model_path, model_type=args.model_type, norm_path=args.norm_path)

