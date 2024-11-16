import time
import os
import sys
import multiprocessing
import concurrent.futures
import numpy as np
from tqdm import tqdm

import random

 
# sys.path.append(os.path.join("/home/eminburakonat", 'RL-UAM-Framework', 'RL-UAM'))
# sys.path.append(os.path.join("/home/eminburakonat", 'RL-UAM-Framework'))

# Add the RL-UAM folder to the current path
current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(current_path + "/RL-UAM")
sys.path.append(current_path + "/RL-UAM/src/rl_models/sb3")


from src.utils.helpers import seconds_to_hms, ymd_hms, get_random_id

import argparse
from typing import Dict
from src.utils.helpers import read_config_file
import optuna
from optuna.samplers import NSGAIISampler
import optuna_distributed
import itertools
from functools import partial
from src.rl_models.sb3.sb3_runners.maskable_ppo import maskable_ppo


# ------------------ SETUP ------------------ START
def create_log_directories(base_dir: str, unique_id: str):
    timestamp = ymd_hms()
    log_dir = os.path.join(f"logs/sb3/optuna/{base_dir}", base_dir + "_" +timestamp + "_" + unique_id)
    tensorboard_log_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    return log_dir, tensorboard_log_dir

def run_rl_model(rl_model: str, log_dir: str, tensorboard_log_dir: str, env_config: Dict, rl_config: Dict, progress_bar: bool, save_model: bool = False, exp_id: str = None):
    if rl_model == "MaskablePPO":
        return maskable_ppo(log_dir, tensorboard_log_dir, env_config, rl_config, progress_bar=progress_bar, save_model=save_model, exp_id=exp_id)
    elif rl_model == "MaskableRecurrentPPO":
        return maskable_recurrent_ppo(log_dir, tensorboard_log_dir, env_config, rl_config, progress_bar=progress_bar, save_model=save_model, exp_id=exp_id)
    elif rl_model == "MaskableGATPPO":
        return maskable_gat_ppo(log_dir, tensorboard_log_dir, env_config, rl_config, progress_bar=progress_bar, save_model=save_model, exp_id=exp_id)
    else:
        raise NotImplementedError(f"RL model {rl_model} not implemented")
# ------------------ SETUP ------------------ END

# ------------------ SB3 RUNNER ------------------ START
def sb3_runner(env_config: Dict, rl_config: Dict, log_dir: str, tensorboard_log_dir: str, progress_bar: bool = False, save_model: bool = True):
    # if env_config.get("sim_mode", {}).get("client_server", False):
        # time.sleep(3)

    start_time = time.time()

    # Create experiment id
    exp_id = f"{rl_config['rl_model']}_{ymd_hms()}"

    run_rl_model(rl_model=rl_config["rl_model"], 
                 log_dir=log_dir, 
                 tensorboard_log_dir=tensorboard_log_dir, 
                 env_config=env_config, 
                 rl_config=rl_config, 
                 progress_bar=progress_bar,
                 save_model=save_model,
                 exp_id=exp_id)

    elapsed_time = time.time() - start_time
    print(f"{rl_config['rl_model']} took {seconds_to_hms(elapsed_time)} to complete.")
    return None
# ------------------ SB3 RUNNER ------------------ END

# ------------------ OPTUNA ------------------ START
def suggest_hyperparameters(trial, config):
    hyperparameters = {}
    for param, param_config in config.items():
        if 'min' in param_config and 'max' in param_config:
            # Continuous parameter
            hyperparameters[param] = trial.suggest_float(
                param, param_config['min'], param_config['max']
            )
        elif 'values' in param_config:
            # Categorical parameter
            hyperparameters[param] = trial.suggest_categorical(
                param, tuple(param_config['values'])
            )
        else:
            raise ValueError(f"Invalid configuration for parameter: {param}")
    return hyperparameters


def objective(trial, env_config, rl_config, experiment_name="Optuna", save_model=False):
    trial_id = f"trial_{trial.number}"
    exp_id = f"{rl_config['rl_model']}_{trial_id}_{ymd_hms()}"
    trial_hyperparameters = suggest_hyperparameters(trial, rl_config['parameters'])
    trial_rl_config = {**rl_config, **trial_hyperparameters}
    log_dir, tensorboard_log_dir = create_log_directories(base_dir=rl_config['description'], 
                                                          unique_id=trial_id)

    mean_reward = run_rl_model(rl_model=rl_config["rl_model"], 
                        log_dir=log_dir, 
                        tensorboard_log_dir=tensorboard_log_dir, 
                        env_config=env_config, 
                        rl_config=trial_rl_config, 
                        progress_bar=False, 
                        save_model=save_model,
                        exp_id=exp_id)
    return mean_reward if mean_reward is not None else -np.inf

# Define a wrapper function that calls 'objective' with the required arguments
def objective_wrapper(trial, env_config, rl_config, experiment_name, save_model):
    return objective(trial, env_config, rl_config, experiment_name, save_model)


def run_optuna_study(env_config, rl_config, n_trials=2, n_jobs=1, experiment_name="Optuna", save_model=False):
    db_url = env_config['external_optimization_params']['optuna_db_url']
    if db_url is not None:
        sampler = NSGAIISampler()
        storage = optuna.storages.RDBStorage(url=db_url)
        study = optuna.create_study(direction='maximize', 
                                    study_name=experiment_name, 
                                    storage=storage, 
                                    load_if_exists=True, 
                                    sampler=sampler)
        study = optuna_distributed.from_study(study)
    else:
        study = optuna.create_study(direction='maximize', 
                                    study_name=experiment_name)

    # Use functools.partial to create a new function that has some parameters pre-filled
    wrapped_objective = partial(objective_wrapper, env_config=env_config, rl_config=rl_config, experiment_name=experiment_name, save_model=save_model)

    # Create a progress bar
    pbar = tqdm(total=n_trials, desc="Optuna Trials")

    # Define a callback to update the progress bar
    def tqdm_callback(study, trial):
        pbar.update(1)

    study.optimize(wrapped_objective, 
                   n_trials=n_trials, 
                   n_jobs=n_jobs, 
                   gc_after_trial=True, 
                   callbacks=[tqdm_callback])
    
    # Close the progress bar
    pbar.close()

    # Output the best parameters
    print("Best trial:")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")

# ------------------ OPTUNA ------------------ END
        
# ------------------ MULTIPROCESSING ------------------ START
def generate_hyperparameter_combinations(rl_config):
    param_lists = {k: v['values'] for k, v in rl_config['parameters'].items()}
    for combination in itertools.product(*param_lists.values()):
        yield dict(zip(param_lists.keys(), combination))


def train_with_hyperparameters(env_config, rl_config, combination, combination_id, save_model):
    rl_config_updated = rl_config.copy()
    rl_config_updated.update(combination)
    log_dir, tensorboard_log_dir = create_log_directories(".", combination_id)

    sb3_runner(env_config, rl_config_updated, progress_bar=False, save_model=save_model, log_dir=log_dir, tensorboard_log_dir=tensorboard_log_dir)

def parallel_training(env_config, rl_config, num_cores, save_model=False):
    combinations = list(generate_hyperparameter_combinations(rl_config))
    print(f"Running {len(combinations)} combinations of hyperparameters")
    random.shuffle(combinations)  # Shuffle the combinations    
    pool = multiprocessing.Pool(num_cores)
    for idx, combination in enumerate(combinations):
        combination_id = f"combination_{idx}"
        pool.apply_async(train_with_hyperparameters, args=(env_config, rl_config, combination, combination_id, save_model))
    pool.close()
    pool.join()

# ------------------ MULTIPROCESSING ------------------ END

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", "-ec", type=str, default="../../configs/config_rl.json", help="The environment config file path", required=True)
    parser.add_argument("--rl_config", "-rlc", type=str, help="The RL config file path. Should be yaml file", required=True)
    parser.add_argument("--num_cores", "-n", type=int, default=1, help="Number of cores to use for parallelization")  
    parser.add_argument("--parallel", "-p", action="store_true", help="Enable parallel execution")
    parser.add_argument("--optimize", "-o", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--save_model", "-s", action="store_true", help="Save model")
    parser.add_argument("--experiment_name", "-exp", type=str, default="Optuna", help="Name of the experiment")
    args = parser.parse_args()

    rl_config = read_config_file(args.rl_config)
    env_config = read_config_file(args.env_config)

    if args.optimize:
        # Run the optuna study
        print("Running Optuna study")
        n_trials = max(40, args.num_cores)
        run_optuna_study(env_config, rl_config, n_trials=n_trials, n_jobs=args.num_cores, experiment_name=args.experiment_name, save_model=args.save_model)
    elif args.parallel:
        # Run the training loop in parallel
        print(f"Running {args.num_cores} parallel instances of SB3 runner")
        parallel_training(env_config, rl_config, args.num_cores, save_model=args.save_model)
    elif args.num_cores > 1:
        raise ValueError("Number of cores cannot be greater than 1 to run single instance of SB3 runner")
    elif not args.parallel and not args.optimize:
        # Run the training loop
        print(f"Running SB3 runner on {args.num_cores} core")
        log_dir, tensorboard_log_dir = create_log_directories(rl_config['description'], rl_config['rl_model'])
        sb3_runner(env_config=env_config, rl_config=rl_config, progress_bar=True, save_model=args.save_model, log_dir=log_dir, tensorboard_log_dir=tensorboard_log_dir)
    else:
        # Raise CLI error
        raise ValueError("Either --parallel or --optimize should be enabled or none.")

if __name__ == "__main__":
    # import multiprocessing as mp
    # mp.set_start_method('spawn')
    main()