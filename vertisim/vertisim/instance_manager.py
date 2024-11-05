from vertisim.vertisim import VertiSim
import simpy
import numpy as np
import time

class InstanceManager:
    def __init__(self, config):
        self.config = config
        self.sim_instance = None
        self.status = False
        self._setup_sim_instance()
            
    def _setup_sim_instance(self, reset=False):
        """Setup simulation instance with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.sim_instance is None:
                    self.sim_instance = VertiSim(
                        env=simpy.Environment(),
                        config=self.config,
                        reset=reset
                    )
                    self.status = True
                    return True
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                self.status = False
                raise RuntimeError(f"Failed to initialize simulation after {max_retries} attempts")
        return False

    def reset(self):
        """Reset the simulation instance"""
        try:
            # Close existing instance if it exists
            if self.sim_instance is not None:
                self.sim_instance.close()
                self.sim_instance = None
            
            # Create new instance
            self._setup_sim_instance(reset=True)
            
            # Get initial state to ensure everything is properly initialized
            initial_state = self.get_initial_state()
            return {"status": "Success", "message": "Reset successful"}
        except Exception as e:
            self.status = False
            raise RuntimeError(f"Reset failed: {str(e)}")
    
    # def get_initial_state(self):
    #     self._setup_sim_instance()
    #     initial_state = self.sim_instance.get_initial_state()
    #     action_mask = self.sim_instance.action_mask(initial_state=True)
    #     return {"initial_state": initial_state, "action_mask": action_mask}
    
    def get_initial_state(self):
        """Get initial state ensuring sim_instance exists"""
        if self.sim_instance is None:
            self._setup_sim_instance()
        
        if self.sim_instance is None:
            raise RuntimeError("Failed to initialize simulation instance")
            
        initial_state = self.sim_instance.get_initial_state()
        action_mask = self.sim_instance.action_mask(initial_state=True)
        # print(f"Initial state: {initial_state}")
        return {"initial_state": initial_state, "action_mask": action_mask}
        

    def step(self, actions):
        """Take a step in the simulation"""
        # if self.sim_instance is None:
        #     self._setup_sim_instance()  # Try to reinitialize if needed
        #     if self.sim_instance is None:
        #         raise RuntimeError("Simulation instance is not initialized")
        if self.sim_instance is None:
            raise RuntimeError("Simulation instance is not initialized")
        
        if self.config["sim_mode"]["client_server"]:
            try:
                response = self.sim_instance.step(actions)
                # print(f"Actions: {actions}")
                # print(f"Step response: {response}")
                return response
            except Exception as e:
                self.status = False
                raise RuntimeError(f"Step failed: {str(e)}")
        else:
            response = self.sim_instance.step(actions)
            return {
                "new_state": response[0],
                "reward": response[1],
                "terminated": response[2],
                "truncated": response[3],
                "action_mask": response[4]
            }
    
    def close(self):
        """
        Close the VertiSim instance and release all resources.
        """
        if hasattr(self, 'sim_instance') and self.sim_instance is not None:
            self.sim_instance.close()
            self.sim_instance = None  # Optional: Help garbage collection   
        self.status = False 
    
    def get_performance_metrics(self):
        return self.sim_instance.get_performance_metrics()

    def get_vertiport_ids_distances(self):
        return self.sim_instance.sim_setup.vertiport_ids, self.sim_instance.sim_setup.vertiport_distances
    
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Remove sim_instance to prevent pickling non-picklable objects
    #     state['sim_instance'] = None
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # Reinitialize sim_instance
    #     self.sim_instance = None