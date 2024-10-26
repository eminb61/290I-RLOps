from vertisim.vertisim import VertiSim
import simpy

class InstanceManager:
    def __init__(self, config):
        self.config = config
        self.sim_instance = None  # Delay initialization
        self.status = True
    
    def _setup_sim_instance(self, reset=False):
        if self.sim_instance is None:
            self.sim_instance = VertiSim(
                env=simpy.Environment(),
                config=self.config,
                reset=reset
            )

    def reset(self):
        self._setup_sim_instance(reset=True)
        # self.sim_instance.close()
        self.status = False
        self.sim_instance = VertiSim(
            env=simpy.Environment(),
            config=self.config
        )
        self.status = self.sim_instance.status
    
    def get_initial_state(self):
        self._setup_sim_instance()
        initial_state = self.sim_instance.get_initial_state()
        action_mask = self.sim_instance.action_mask(initial_state=True)
        return {"initial_state": initial_state, "action_mask": action_mask}
        
    def step(self, actions):
        # self._setup_sim_instance()
        if self.config["sim_mode"]["client_server"]:
            return self.sim_instance.step(actions)  
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
    
    def get_performance_metrics(self):
        return self.sim_instance.get_performance_metrics()

    def get_vertiport_ids_distances(self):
        return self.sim_instance.sim_setup.vertiport_ids, self.sim_instance.sim_setup.vertiport_distances
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove sim_instance to prevent pickling non-picklable objects
        state['sim_instance'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize sim_instance
        self.sim_instance = None