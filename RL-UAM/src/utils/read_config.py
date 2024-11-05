from typing import Dict

def get_simulation_params(config: Dict):
    return {
        "n_actions": get_action_count(config),
        "n_aircraft": get_aircraft_count(config), 
        "n_vertiports": get_vertiport_count(config),
        "n_vertiport_state_variables": get_vertiport_state_variable_count(config),
        "n_aircraft_state_variables": get_aircraft_state_variable_count(config),
        "n_environmental_state_variables": get_environmental_state_variable_count(config),
        "n_additional_state_variables": get_additional_state_variable_count(config)
    }

def get_action_count(config):
    return get_vertiport_count(config) + 2

def get_aircraft_count(config) -> int:
    return sum(
        vertiport['aircraft_arrival_process']['num_initial_aircraft_at_vertiport']
        for _, vertiport in config['network_and_demand_params']['vertiports'].items()
    )

def get_vertiport_count(config) -> int:
    return len(config['network_and_demand_params']['vertiports']) 

def get_vertiport_state_variable_count(config):
    count = 0
    for state in config['sim_params']['simulation_states']['vertiport_states']:
        if state in config['sim_params']['per_destination_states']:
            count += get_vertiport_count(config) - 1
        else:
            if state == "waiting_time_bins":
                # HARD CODED for 290I
                num_waiting_time_bins = config['sim_params']['max_passenger_waiting_time'] // config['external_optimization_params']['periodic_time_step']
                count += num_waiting_time_bins * (get_vertiport_count(config) - 1)
            else:
                count += 1
    return count

def get_aircraft_state_variable_count(config):
    return len(config['sim_params']['simulation_states']['aircraft_states'])

def get_environmental_state_variable_count(config):
    num_env_vars = len(config['sim_params']['simulation_states']['environmental_states'])
    if num_env_vars > 0:
        return num_env_vars
    else:
        return -1

def get_additional_state_variable_count(config):
    return len(config['sim_params']['simulation_states']['additional_states'])
