from typing import Any, Dict
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch

# class EarlyStoppingCallback(BaseCallback):
#     def __init__(self, check_freq, performance_threshold, verbose=1):
#         super(EarlyStoppingCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.performance_threshold = performance_threshold
#         self.best_mean_reward = -np.inf

#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:
#             # Retrieve performance
#             x, y = self.model.ep_info_buffer.xs, self.model.ep_info_buffer.ys
#             if len(y) > 0:
#                 mean_reward = np.mean(y[-100:])  # Last 100 episodes
#                 if mean_reward > self.best_mean_reward:
#                     self.best_mean_reward = mean_reward

#                 if self.n_calls >= 100000 and mean_reward < self.performance_threshold:
#                     print(f"Early stopping: Step {self.n_calls}, Mean reward {mean_reward} below threshold")
#                     return False  # Stop training
#         return True
    

class EarlyStoppingCallback(BaseCallback):
    """
    Custom callback for early stopping the training based on performance threshold.

    :param verbose: Verbosity level
    :param check_freq: Frequency to perform the check
    :param performance_threshold: Threshold for performance
    :param min_timesteps: Minimum number of timesteps before considering stopping
    """
    def __init__(self, check_freq, performance_threshold, min_timesteps=100000, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.performance_threshold = performance_threshold
        self.min_timesteps = min_timesteps

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0 and self.num_timesteps > self.min_timesteps:
            # Assuming that 'self.model.ep_info_buffer' contains episode information
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            if mean_reward < self.performance_threshold:
                print(f"Early stopping at timestep {self.num_timesteps} due to low performance.")
                return False  # Stop training
        return True
    

from stable_baselines3.common.logger import HParam

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params


    def _on_training_start(self) -> None:
        # hparam_dict = self.params
        # Ensure all values are of a compatible type
        hparam_dict = {k: v if isinstance(v, (int, float, str, bool)) else str(v) 
                       for k, v in self.params.items()}


        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True



class CosineAnnealingSchedulerCallback(BaseCallback):
    """
    Custom callback to update the learning rate using CosineAnnealingWarmRestarts scheduler.

    :param T_0: Number of iterations for the first restart.
    :param T_mult: A factor increases T_i after a restart. Default: 1.
    :param eta_min: Minimum learning rate. Default: 0.
    :param verbose: Verbosity level.
    """
    def __init__(self, T_0=10, T_mult=1, eta_min=0, verbose=0):
        super(CosineAnnealingSchedulerCallback, self).__init__(verbose)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.scheduler = None

    def _on_training_start(self) -> None:
        # Initialize the scheduler at the start of training
        optimizer = self.model.policy.optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min
        )

    def _on_step(self) -> bool:
        # Update the scheduler at each call
        # Assuming one call corresponds to one optimizer step
        self.scheduler.step()
        return True
