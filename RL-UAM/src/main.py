import time
from utils.helpers import seconds_to_hms

def main():
    time.sleep(8)

    start_time = time.time()
    # Set the log directories
    log_dir = "./logs"
    tensorboard_log_dir = f"{log_dir}/tensorboard"

    # Set the RL model
    rl_model = "MaskablePPO"

    if rl_model == "PPO":
        from rl_models.sb3.ppo import ppo
        ppo(log_dir, tensorboard_log_dir)
    elif rl_model == "MaskablePPO":
        from rl_models.sb3.maskable_ppo import maskable_ppo
        maskable_ppo(log_dir, tensorboard_log_dir)
    elif rl_model == "DQN":
        from rl_models.sb3.dqn import dqn
        dqn(log_dir, tensorboard_log_dir)
    else:
        raise NotImplementedError(f"RL model {rl_model} not implemented")

    # End timing after the function completes
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"{rl_model} took {seconds_to_hms(elapsed_time)} to complete.")

if __name__ == "__main__":
    main()