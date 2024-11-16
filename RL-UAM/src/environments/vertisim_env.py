from typing import Any, SupportsFloat, Dict, Tuple, List
import gymnasium as gym
import numpy as np
from src.utils.helpers import extract_dict_values
from src.utils.read_config import get_simulation_params
import requests
import itertools
import torch
import time
import json
import os
from src.utils.config_watcher import ConfigWatcher
import sys
from requests.exceptions import RequestException

SERVICE_ORCHESTRATOR_API_URL = "http://service_orchestrator:6000"
VERTISIM_API_URL = "http://vertisim_service:5001"

class VertiSimEnvWrapper(gym.Env):
    def __init__(self, rl_model: str, env_config: Dict) -> None:
        super().__init__()
        self.rl_model = rl_model
        self.env_config = env_config
        self.config_path = os.getenv('CONFIG_PATH', '/app/config_rl_training.json')

        # # Initialize config watcher
        # self.config_watcher = ConfigWatcher(self.config_path)
        # self.config_watcher.start(self._handle_config_update)

        self.client_server_mode = bool(self.env_config['sim_mode']['client_server'])
        self.instance_id = None
        self.params = self._fetch_params()

        if not self.client_server_mode:
            from vertisim.vertisim.instance_manager import InstanceManager
            self.instance_manager = InstanceManager(config=self.env_config)
                    
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        self.action_space = self.get_action_space(self.params['n_actions'], 
                                                self.params['n_aircraft'])
        self.observation_space = self.get_observation_space(
            self.params['n_vertiports'],
            self.params['n_aircraft'],
            self.params['n_vertiport_state_variables'],
            self.params['n_aircraft_state_variables'],
            self.params['n_environmental_state_variables'],
            self.params['n_additional_state_variables']
        )
        self.mask = np.zeros((self.params['n_aircraft'], self.params['n_actions']), dtype=np.int64)
        self.mask[:, -1] = 1

    def _load_config(self) -> Dict:
        """Load config from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file: {str(e)}")

    # def _handle_config_update(self, new_config: Dict):
    #     """Handle config file updates"""
    #     try:
    #         self.env_config = new_config
            
    #         # Update parameters based on new config
    #         self.params = get_simulation_params(self.env_config)
            
    #         # Reset spaces with new parameters
    #         self._setup_spaces()
            
    #         # Force a reset on next step
    #         self.needs_reset = True

    #         print(f"Updated environment with new config")
            
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to update environment with new config: {str(e)}")

    def _fetch_params(self):
        # if self.client_server_mode:
        #     return self._fetch_params_client_server()
        # else:
        #     return self._fetch_params_local()
        return self._fetch_params_local()
        
    def _fetch_params_local(self):
        return get_simulation_params(self.env_config)

    def _fetch_params_client_server(self):
        if not self.instance_id:
            raise RuntimeError("No instance ID available")
            
        response = requests.get(f"{VERTISIM_API_URL}/instance/{self.instance_id}/space_params", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            raise ConnectionError(f"Failed to fetch parameters from VertiSim. Status code: {response.status_code}, Response text: {response.text}")

    def get_action_space(self, n_actions, n_aircraft):
        if self.rl_model in ["DQN"]:
            return gym.spaces.Discrete(n_actions**n_aircraft)
        elif self.rl_model in ["PPO", "MaskablePPO", "RecurrentPPO", "MaskableRecurrentPPO", 
                               "MaskableGATPPO", "MaskableRecurrentGATPPO", "RandomPolicy"]:
            return gym.spaces.MultiDiscrete([n_actions] * n_aircraft)
        else:
            raise ValueError(f"Unsupported RL model: {self.rl_model}")

    def get_observation_space(self, 
                              n_vertiports, 
                              n_aircraft, 
                              n_vertiport_state_variables, 
                              n_aircraft_state_variables, 
                              n_environmental_state_variables,
                              n_additional_state_variables):

        total_state_variables = (
            n_vertiports * n_vertiport_state_variables +
            n_aircraft * n_aircraft_state_variables +
            n_vertiports * (n_environmental_state_variables+1) +
            n_additional_state_variables # For sim_time
        )

        return gym.spaces.Box(low=0, high=np.inf, shape=(total_state_variables,), dtype=np.float64)
    
    def step(self, action) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        # Convert actions
        action: list = self._convert_actions(action)
        # print(f"Action: {action}")        

        if self.client_server_mode:
            return self._step_client_server(action)
        else:
            return self._step(action)

    def _step(self, action):
        response = self.instance_manager.step(action)
        return self.process_step_response(response)

    def _step_client_server(self, action):
        MAX_RETRIES: int = 6
        BACKOFF_FACTOR: int = 2
        INITIAL_DELAY: int = 1

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{VERTISIM_API_URL}/instance/{self.instance_id}/step",
                    json={"actions": action},
                    timeout=120
                )
                if response.status_code == 200:
                    return self.process_step_response(response)
                else:
                    raise ConnectionError(f"Failed to fetch step from VertiSim. Status code: {response.status_code}, Response text: {response.text}")
            except RequestException as e:
                if attempt >= MAX_RETRIES - 1:
                    raise ConnectionError(f"Failed to fetch step from VertiSim after {MAX_RETRIES} tries. Error: {str(e)}") from e
                delay = INITIAL_DELAY * BACKOFF_FACTOR ** attempt
                print(f"Failed to fetch step from VertiSim. Error: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)

    def process_step_response(self, response):
        # Process the successful response
        new_state, reward, terminated, truncated, self.mask = self._extract_step_response(response=response)
        # Convert action mask to the correct format
        action_mask = self.action_mask()

        if self.action_space.__class__.__name__ == "Discrete":
            action_mask = self.multidiscrete_to_discrete_mask(multi_discrete_mask=action_mask, dimensions=self.params['n_aircraft'], n_actions=self.params['n_actions'])
        # action_mask = self.multidiscrete_to_discrete_mask(multi_discrete_mask=action_mask, dimensions=self.params['n_aircraft'], n_actions=self.params['n_actions'])

        info = {'action_mask': action_mask}
    
        return new_state, reward, terminated, truncated, info
        # else:
        #     dones = terminated or truncated
        #     return new_state, reward, dones, info

    def reset(self, seed=None, options=None):
        """Reset environment and reload config"""
        super().reset(seed=seed)
        
        # Reload config file
        self.env_config = self._load_config()
        self.params = self._fetch_params()
        self._setup_spaces()

        if self.client_server_mode:
            return self._reset_client_server()
        else:
            return self._reset()
        
    def _reset(self):
        self.instance_manager.reset()
        # Wait until VertiSim is ready
        self.wait_for_vertisim()
        # Get the initial state from VertiSim
        response = self.instance_manager.get_initial_state()
        # Extract the initial state from the response
        initial_state, action_mask = response['initial_state'], response['action_mask']

        return self.convert_state_and_action_mask_to_tensor(initial_state, action_mask)

    def _reset_client_server(self):
        # Send a reset request to VertiSim and receive the initial state
        response = requests.post(f"{SERVICE_ORCHESTRATOR_API_URL}/reset_instance", timeout=120)
        if response.status_code == 200:
            try:
                response_data = response.json()
                self.instance_id = response_data["instance_id"]
                initial_state = response_data['initial_state']
                action_mask = response_data['action_mask']
                
                return self.convert_state_and_action_mask_to_tensor(initial_state, action_mask)
            except:
                raise ValueError("Failed to extract initial state from reset response.")
        else:
            raise ConnectionError(f"Failed to fetch reset from Service Orchestrator. Status code: {response.status_code}, Response text: {response.text}")
    
    def convert_state_and_action_mask_to_tensor(self, initial_state, action_mask):
        # Convert action mask to the correct format
        action_mask = self.convert_action_mask_to_numpy(action_mask)
        if self.action_space.__class__.__name__ == "Discrete":
            action_mask = self.multidiscrete_to_discrete_mask(multi_discrete_mask=action_mask, 
                                                              dimensions=self.params['n_aircraft'], 
                                                              n_actions=self.params['n_actions'])
        # action_mask = self._convert_mask(action_mask)
        obs_tensor = np.array(extract_dict_values(initial_state)).reshape(-1)
        info = {'action_mask': action_mask}   
        return obs_tensor, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        if self.client_server_mode and self.instance_id:
            try:
                requests.delete(f"{VERTISIM_API_URL}/instance/{self.instance_id}", timeout=30)
            except:
                pass
            self.instance_id = None
        elif hasattr(self, 'instance_manager') and self.instance_manager:
            self.instance_manager.close()
            self.instance_manager = None
        if hasattr(self, 'config_watcher'):
            self.config_watcher.stop()
        super().close()
        
    def wait_for_vertisim(self, timeout: int = 120):
        """
        Checks whether VertiSim is ready to accept the next request.
        """
        start_time = time.time()

        while True:
            try:
                status = self.instance_manager.status
                if status == True:
                    break
            except RequestException as e:
                print(f"Waiting for VertiSim to be ready. Error: {str(e)}")
                time.sleep(1)
            # Check if timeout has been reached
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out while waiting for Vertisim to be ready.")    

    def action_mask(self):
        return self.convert_action_mask_to_numpy(self.mask)
    
    def convert_action_mask_to_numpy(self, action_mask):
        return np.array(action_mask, dtype=np.float64).reshape(self.params['n_aircraft'], self.params['n_actions'])

    def _convert_actions(self, action) -> list:
        if self.action_space.__class__.__name__ == "Discrete":
            action = self._convert_dqn_action(action)
        else:
            action = self._convert_ppo_action(action)
        return action

    def _convert_dqn_action(self, action) -> list:
        action = VertiSimEnvWrapper.discrete_to_multidiscrete(action, dimensions=self.params['n_aircraft'], n_actions=self.params['n_actions'])
        return [self._convert_single_action(a) for a in action]

    def _convert_ppo_action(self, action) -> list:
        return action.tolist()

    def _convert_single_action(self, action):
        if isinstance(action, np.int64) or torch.is_tensor(action):
            return int(action.item())
        return int(action)

    def _convert_mask(self, mask: np.ndarray) -> list:
        if self.action_space.__class__.__name__ == "Discrete":
            actions = []
            for array in mask:
                actions.append(VertiSimEnvWrapper.multidiscrete_to_discrete(array, n_actions=self.params['n_actions']))
            action_mask = np.zeros((self.params['n_actions']**self.params['n_aircraft']))
            action_mask[actions] = 1
            return np.array(action_mask)
        else:
            return mask

    def multidiscrete_to_discrete_mask(self, multi_discrete_mask, dimensions, n_actions):
        """
        Converts a multi-discrete action mask to a discrete action mask.
        
        :param multi_discrete_mask: List or 1D numpy array representing the multi-discrete action mask.
        :param dimensions: Number of dimensions in the multi-discrete action space.
        :param n_actions: Number of actions in each dimension of the multi-discrete action space.
        :return: A 1D numpy array representing the discrete action mask.
        """
        # print(f"Multi-discrete mask: {multi_discrete_mask}")
        # Ensure the mask is a numpy array for easy manipulation
        multi_discrete_mask = np.array(multi_discrete_mask).reshape((-1, n_actions))

        # The total number of discrete actions
        total_discrete_actions = n_actions ** dimensions

        # Initialize the discrete mask with zeros (assume all actions are invalid initially)
        discrete_mask = np.zeros(total_discrete_actions, dtype=int)

        # Iterate over all possible combinations of actions in the multi-discrete space
        for action_combination in itertools.product(*[range(n_actions) for _ in range(dimensions)]):
            # Check if the current combination is valid (not masked)
            if all(multi_discrete_mask[dim, action] == 1 for dim, action in enumerate(action_combination)):
                # Convert the valid multi-discrete action to a discrete action
                discrete_action = sum(action * (n_actions ** i) for i, action in enumerate(reversed(action_combination)))
                # Mark the corresponding discrete action as valid
                discrete_mask[discrete_action] = 1

        # print(f"Discrete mask: {discrete_mask}")
        return discrete_mask
        
    def _extract_step_response(self, response):
        # Extract new state, reward, done, and info from response data
        if self.client_server_mode:
            data = response.json()
        else:
            data = response
                
        new_state = np.array(extract_dict_values(data['new_state']))

        reward = data['reward']
        terminated = data['terminated']
        truncated = data['truncated']
        action_mask = data['action_mask']
        return new_state, reward, terminated, truncated, action_mask

    def seed(self, seed=None):
        # self.enseed(seed)
        # return super(VertiSimEnvWrapper, self).seed(seed)
        pass

    @staticmethod
    def multidiscrete_to_discrete(action, n_actions=3) -> int:
        """Converts a MultiDiscrete action to a Discrete action."""
        discrete_action = 0
        for i, a in enumerate(reversed(action)):
            discrete_action += a * (n_actions ** i)
        return int(discrete_action)

    @staticmethod
    def discrete_to_multidiscrete(action, dimensions=4, n_actions=3) -> List[int]:
        """Converts a Discrete action back to a MultiDiscrete action."""
        multidiscrete_action = []
        for _ in range(dimensions):
            multidiscrete_action.append(action % n_actions)
            action = action // n_actions
        return list(reversed(multidiscrete_action))  
    
    def get_wrapper_attr(self, attr_name):
        """
        Method to get attributes from the wrapper.
        :param attr_name: Name of the attribute to retrieve.
        :return: The value of the requested attribute.
        """
        try:
            return getattr(self, attr_name)
        except AttributeError:
            raise AttributeError(f"Attribute {attr_name} not found in VertiSimEnvWrapper.")
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the instance_manager to prevent pickling non-picklable objects
        state['instance_manager'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the instance_manager
        self.instance_manager = None