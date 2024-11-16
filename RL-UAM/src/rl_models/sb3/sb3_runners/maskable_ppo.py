from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, ProgressBarCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from src.environments.vertisim_env import VertiSimEnvWrapper
import gymnasium as gym
from torch import nn
from src.utils.learning_rate_schedule import linear_schedule
from src.utils.helpers import mask_fn, str_to_activation, get_learning_schedule
from src.utils.callbacks import HParamCallback, EarlyStoppingCallback
from stable_baselines3.common.logger import configure
import time
import os


def make_env(rl_model, env_config, rank, seed=0):
    def _init():
        env = gym.make('vertisim', rl_model=rl_model, env_config=env_config)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        # Set the seed
        env.seed(seed+rank)
        return env
    return _init


def maskable_ppo(log_dir, tensorboard_log_dir, env_config, rl_config, progress_bar=False, save_model=False, exp_id=None):

    policy_kwargs = dict(activation_fn=str_to_activation[rl_config["activation_fn"]],
                        net_arch=[rl_config["hidden_layers"]]*rl_config["n_hidden_layers"])  
                            
    # Number of environments
    n_envs = rl_config.get("num_workers", 1)

    # Create the vectorized environment
    # envs = SubprocVecEnv([make_env("MaskablePPO", env_config, rank=i, seed=rl_config["seed"]) for i in range(n_envs)])

    envs = DummyVecEnv([make_env("MaskablePPO", env_config, rank=0, seed=rl_config["seed"])])    
    # Wrap the environment with VecNormalize for observation and reward normalization
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, gamma=rl_config["gamma"], clip_obs=30, clip_reward=500)

    # Create the model
    model = MaskablePPO(policy=MaskableActorCriticPolicy, 
                        env=envs, 
                        verbose=0, 
                        batch_size=rl_config["batch_size"],
                        ent_coef=rl_config["ent_coef"],
                        n_steps=rl_config["n_steps"],
                        learning_rate=get_learning_schedule(learning_rate=rl_config["learning_rate"], 
                                                            learning_scheduler=rl_config["learning_scheduler"]), 
                        gamma=rl_config["gamma"],
                        clip_range=rl_config["clip_range"],     
                        clip_range_vf=rl_config["clip_range_vf"],       
                        policy_kwargs=policy_kwargs,
                        tensorboard_log=tensorboard_log_dir,
                        stats_window_size=10) 
    # Print the model
    print(f"Using environment: {envs} with {MaskablePPO} policy. It will run for {rl_config['total_steps']} steps")  

    callbacks = []    
    callbacks.append(HParamCallback(params=rl_config))

    # Save a checkpoint every X steps
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=log_dir,
                                             name_prefix="checkpoint", save_vecnormalize=True)
    # callbacks.append(checkpoint_callback)
    

    early_stopping_callback = EarlyStoppingCallback(check_freq=20000,
                                                    performance_threshold=0,
                                                    min_timesteps=100000) 
    callbacks.append(early_stopping_callback)   
    
    if rl_config["learning_scheduler"] == "CosineAnnealingWarmRestarts":
        from src.utils.callbacks import CosineAnnealingSchedulerCallback
        callbacks.append(CosineAnnealingSchedulerCallback())


    rl_config['hidden_layers'] = str(rl_config['hidden_layers'])

    model.learn(total_timesteps=rl_config['total_steps'], 
                callback=callbacks,
                log_interval=1, progress_bar=True)
    
    print("Training finished: Exp id: ", exp_id)

    if save_model:
        # Save the model
        model.save(f"./model/{exp_id}")
        envs.save(f"./model/{exp_id}_vecnormalize.pkl")
        print(f"Model saved (exp_id: {exp_id})")

    # Evaluate the trained agent
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)
    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")