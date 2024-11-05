import simpy
from typing import Any, Dict, List
from .sim_setup import SimSetup
from .aircraft.aircraft import AircraftStatus
from .utils.helpers import get_stopping_events, get_truncation_events, miliseconds_to_hms, \
    create_action_enum, convert_action_values_to_enum, create_action_dict, reverse_dict, \
        current_and_lookahead_pax_count, get_total_waiting_passengers_at_vertiport, \
        get_inflow_demand_to_vertiport, extract_dict_values
import time
import numpy as np
import gymnasium as gym
from collections import defaultdict
from pprint import pformat
from enum import Enum
from .utils.get_state_variables import get_simulator_states
from .utils import rl_utils
from .rl_methods.reward_function import RewardFunction
from .rl_methods.action_mask import ActionMask
from .utils.fetch_data_from_db import fetch_latest_data_from_db
from .initiate_flow_entities import initiate_flow_entities
from .utils.units import sec_to_ms, ms_to_hr, sec_to_min, ms_to_min
from .run_step_by_step_simulation import run_step_by_step_simulation, run_steps_until_specific_events


class VertiSim:
    def __init__(self, 
                 env: simpy.Environment, 
                 config: Dict,
                 reset: bool = False):
        self.env = env
        self.config = config
        self.terminal_event, self.truncation_event, self.truncation_events, self.stopping_events = self.set_simulation_events(env=self.env, config=self.config)

        self.sim_start_time = time.time()
        self.sim_setup = SimSetup(env=self.env,
                                  sim_params=config['sim_params'],
                                  sim_mode=config['sim_mode'],
                                  external_optimization_params=config['external_optimization_params'],
                                  network_and_demand_params=config['network_and_demand_params'],
                                  airspace_params=config['airspace_params'],
                                  passenger_params=config['passenger_params'],
                                  aircraft_params=config['aircraft_params'],
                                  output_params=config['output_params'],
                                  stopping_events=self.stopping_events,
                                  truncation_events=self.truncation_events,
                                  truncation_event=self.truncation_event,
                                  reset=reset)  

        # Set the max simulation time
        self.max_sim_time = sec_to_ms(self.sim_setup.sim_params['max_sim_time'])
        # The status of the simulation provides information whether the simulation running or not
        # Remove the old logs.
        # self.sim_setup.logger.remove_logs(seconds=10)

        self.num_aircraft = self.get_aircraft_count()

        if self.config['sim_mode']['rl']:
            self.setup_for_rl()

        self.status = True
        self.last_decision_time = -1
        self.decision_making_interval = sec_to_ms(config['external_optimization_params']['periodic_time_step'])     

    def setup_for_rl(self):
        # self.actions = create_action_enum(self.sim_setup.vertiport_ids)        
        self.action_dict = create_action_dict(self.sim_setup.vertiport_ids)
        self.reverse_action_dict = reverse_dict(self.action_dict)
        self.reward_function = RewardFunction(config=self.config, reverse_action_dict=self.reverse_action_dict, sim_setup=self.sim_setup, logger=self.sim_setup.logger)
        self.action_mask_fn = ActionMask(sim_setup=self.sim_setup, config=self.config, num_aircraft=self.num_aircraft)

    def set_simulation_events(self, env, config):
        """
        Set the simulation events.
        """
        terminal_event = env.event()
        truncation_event = env.event()
        truncation_events = get_truncation_events(env=env,
                                                  truncation_events=config["external_optimization_params"]["truncation_events"],
                                                  aircraft_count=self.get_aircraft_count())
        stopping_events = get_stopping_events(env=env, 
                                              stopping_events=config["external_optimization_params"]["stopping_events"],
                                              aircraft_count=self.get_aircraft_count(),
                                              vertiport_ids=self.get_vertiport_ids(),
                                              pax_count=self.get_passenger_count())
        return terminal_event, truncation_event, truncation_events, stopping_events

    def run(self):
        # This only runs uninterrupted simulation. No distinction for online optimization
        self.run_uninterrupted_simulation()          

        if self.config['output_params']['only_return_brief_metrics']:
            return self.finalize_simulation()
        
        self.finalize_simulation()
        # performance_metrics = self.sim_setup.calculate_performance_metrics()
        # self.sim_setup.save_results(performance_metrics)
        # self.sim_setup.print_passenger_trip_time_stats()

    def finalize_simulation(self):
        if self.config['sim_mode']['rl'] and \
            self.config['sim_params']['algorithm'] != 'RandomPolicy' and \
            self.config['sim_params']['print_brief_metrics']:
            self.sim_setup.log_brief_metrics(print_metrics=self.config['sim_params']['print_brief_metrics'])
                    
        if (not self.sim_setup.sim_mode['rl'] or self.sim_setup.output_params['save_output']):
            self.print_total_time_to_run()
            if self.env.now > 10:
                performance_metrics = self.sim_setup.calculate_performance_metrics()
                self.sim_setup.log_brief_metrics(print_metrics=self.config['sim_params']['print_brief_metrics'])
                if self.config['output_params']['only_return_brief_metrics']:
                    return self.sim_setup.save_results(performance_metrics)
                self.sim_setup.save_results(performance_metrics)

    def run_uninterrupted_simulation(self):
        """
        Run the simulation until the max simulation time or there is no event left in the queue.
        """
        if self.sim_setup.sim_mode['offline_optimization']:
            self.env.run()
        else:
            if self.config['external_optimization_params']['periodic_time_step']:
                self.advance_periodically_ondemand()
            else:
                while self._should_continue_simulation():
                    self.env.step()
            # self.env.run(until=self.max_sim_time)

    def reset(self):
        """
        Reset the simulation.
        """
        # if not self.terminal_event.triggered:
        self.sim_setup.logger.finalize_logging()
        return self.terminal_event.succeed()
        
    def close(self):
        """
        Close the VertiSim simulation and release all resources.
        """
        # Close any SimPy environment resources if necessary
        if hasattr(self, 'env') and self.env is not None:
            # Perform any necessary cleanup on the SimPy environment
            self.env = None  # SimPy environments don't require explicit closure
        
        # Close any loggers or files
        if hasattr(self, 'sim_setup') and self.sim_setup is not None:
            self.sim_setup = None   
    
    def step(self, actions):
        """
        Run the simulation for one timestep.
        """
        # Get the number of triggers for this step
        # total_triggers = sum(action != 0 for action in actions)
        # Apply the actions and rewards at the same time to keep the Markov property
        self.env.process(self.apply_action_reward(actions=actions))
        # Advance to the next stopping event. The returned state and reward pair should be
        # the state and reward after all the actions are applied.

        if self.config['external_optimization_params']['periodic_and_event_driven']:
            return self.advance_to_next_stopping_event_and_stop_periodically()
        elif self.config['external_optimization_params']['periodic_time_step']:
            return self.advance_periodically()
        else:
            return self.advance_to_next_stopping_event()
        # if self.config['external_optimization_params']['periodic_time_step']:
        #     return self.advance_periodically()
        # else:
        #     return self.advance_to_next_stopping_event()

    def apply_action_reward(self, actions: Enum):
        """
        Apply action to the simulation and compute the reward before execution.
        We change the state of the aircrafts based on the action before we step the simulation
        to be able to calculate the reward at the same time for the given action list.

        Actions:
        0, 1, ..., N-1: Idle to fly to vertiport n
        N: Idle to charge
        N+1: Do nothing
        """
        action_dict = {i: self.action_dict[action] for i, action in enumerate(actions)}
        self.sim_setup.logger.info(f"Action: {action_dict}")
        self.sim_setup.system_manager.actions = actions

        # If a passenger has been waiting for more than X minutes, remove them from the waiting room.
        if self.sim_setup.system_manager.external_optimization_params['spill_optimization'] and self.env.now > 0:
            self.sim_setup.system_manager.remove_spilled_passengers(actions=actions)
            
        process_list = []
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            # Charge the aircraft
            if actions[aircraft_id] == self.reverse_action_dict['CHARGE']:
                aircraft.status = AircraftStatus.CHARGE
                aircraft.is_process_completed = False
            # Do nothing
            elif actions[aircraft_id] == self.reverse_action_dict['DO_NOTHING']:
                aircraft.is_process_completed = True
            # Fly to the vertiport
            elif actions[aircraft_id] < self.sim_setup.num_vertiports():
                aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()
                aircraft.origin_vertiport_id = aircraft.current_vertiport_id
                aircraft.destination_vertiport_id = self.sim_setup.vertiport_index_to_id_map[actions[aircraft_id]]

                assert aircraft.current_vertiport_id != aircraft.destination_vertiport_id, f"Invalid action!: Origin vertiport assigned as destination."

                aircraft.status = AircraftStatus.FLY  
                aircraft.is_process_completed = False
                aircraft.flight_direction = f"{aircraft.origin_vertiport_id}_{aircraft.destination_vertiport_id}"
                self.sim_setup.logger.debug(f"Aircraft {aircraft_id} will fly from {aircraft.origin_vertiport_id} to {aircraft.destination_vertiport_id}")
            else:
                raise ValueError(f"Invalid action: {actions[aircraft_id]}")
                
        # Compute the reward
        if self.env.now == 0:
            self.reward_function.reward = 0
        else:
            self.reward_function.compute_reward(actions=actions)

        # Apply the actions
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] == self.reverse_action_dict['CHARGE']:
                process_list.append(self.env.process(aircraft.charge_aircraft(parking_space=aircraft.parking_space)))
            elif actions[aircraft_id] < self.sim_setup.num_vertiports():
                destination_vertiport_id = self.sim_setup.vertiport_index_to_id_map[actions[aircraft_id]]
                # Get the departing passengers
                departing_passengers = self.sim_setup.scheduler.collect_departing_passengers_by_od_vertiport(
                    origin_vertiport_id=aircraft.current_vertiport_id, 
                    destination_vertiport_id=destination_vertiport_id)   
                process_list.append(self.env.process(self.sim_setup.system_manager.reserve_aircraft(aircraft=aircraft, 
                                                                                                    destination_vertiport_id=destination_vertiport_id,
                                                                                                    departing_passengers=departing_passengers)))
        yield self.env.all_of(process_list)   

    def action_mask(self, initial_state=False, final_state=False):
        """
        Create the action mask for the current state of the simulation.
        [Fly: list, Charge: int, Do nothing: int]
        
        Returns:
            List[int]: A flattened list representing the mask where 1 allows the action and 0 masks it.
        """
        return self.action_mask_fn.get_action_mask(initial_state=initial_state, final_state=final_state)

    def schedule_periodic_stop_event(self):
        """
        Schedule a stopping event at a fixed interval.
        """
        # Calculate time until the next multiple of interval
        time_to_next_multiple = self.decision_making_interval - (self.env.now % self.decision_making_interval)
        if time_to_next_multiple == 0:
            time_to_next_multiple = self.decision_making_interval  # If it's exactly a multiple, schedule for the next interval

        stop_event = self.env.event()
        self.env.process(self.trigger_stop_event_after_interval(stop_event, time_to_next_multiple))
        return stop_event

    def trigger_stop_event_after_interval(self, event, interval):
        """
        Process to trigger a given event after a specified interval.
        """
        yield self.env.timeout(interval)
        event.succeed()

    def trigger_stop_event_after_interval_ondemand(self, event, interval):
        """
        Process to trigger a given event after a specified interval.
        """
        # Calculate time until the next multiple of interval
        time_to_next_multiple = interval - (self.env.now % interval)
        if time_to_next_multiple == 0:
            time_to_next_multiple = interval  # If it's exactly a multiple, schedule for the next interval

        yield self.env.timeout(time_to_next_multiple)
        event.succeed()

    def advance_periodically_ondemand(self):
        """
        Advance the simulation for fixed intervals and make decisions.
        """        
        while self._should_continue_simulation_ondemand():
            # Schedule the next stop event
            stop_event = self.env.event()
            self.env.process(self.trigger_stop_event_after_interval_ondemand(stop_event, self.decision_making_interval))
            
            # Advance the simulation until the stop event is triggered
            while not stop_event.triggered and self._should_continue_simulation_ondemand():
                self.env.step()
            
            if self._should_continue_simulation_ondemand():
                self.sim_setup.logger.info(f"Periodic time step: {miliseconds_to_hms(self.env.now)}")
                # Make on-demand decision
                self.env.process(self.sim_setup.system_manager.make_ondemand_decision()) 

    def _should_continue_simulation_ondemand(self):
        return (self.env.now < self.max_sim_time and 
                (not self.terminal_event.triggered and 
                not self.truncation_event.triggered and
                not self._check_demand_satisfied()))

    def advance_periodically(self):
        """
        Advance the simulation for a fixed interval and return the state and reward.
        """
        # Schedule the simulation to advance for the interval
        stop_event = self.schedule_periodic_stop_event()
        # Advance the simulation until the stop event is triggered
        while self._should_continue_simulation():
            # Step the simulation
            self.env.step()

            if stop_event.triggered:
                # Return the states
                return self.return_state_reward_mask()
            
        self.finalize_simulation()
        return self.get_current_state(), round(self.reward_function.reward, 2), self.is_terminated(), self.is_truncated(), self.action_mask(final_state=True)    

    def advance_to_next_stopping_event(self):
        """
        Runs the simulation until the next defined stopping event.
        Rewards the agent right after the each action is completed.
        Rewards the agent separately for each action.
        """

        while self._should_continue_simulation():
            # Step the simulation
            self.env.step()
          
            # If there are any triggered stopping event queue, pop it out, reset it and return the states
            if len(self.sim_setup.system_manager.triggered_stopping_event_queue):
                # Pop the event on top of the queue
                event_name = self.sim_setup.system_manager.triggered_stopping_event_queue.pop()
                self.sim_setup.logger.info(f"Triggered event: {event_name}")
                # Return the states
                return self.return_state_reward_mask()

        self.finalize_simulation()
        return self.get_current_state(), round(self.reward_function.reward, 2), self.is_terminated(), self.is_truncated(), self.action_mask(final_state=True)    

    def advance_to_next_stopping_event_and_stop_periodically(self):
        """
        Runs the simulation until the next defined stopping event.
        Rewards the agent right after the each action is completed.
        Rewards the agent separately for each action.
        """
        periodic_stop_event = self.schedule_periodic_stop_event()

        while self._should_continue_simulation():
            # Step the simulation
            self.env.step()
            
            # Check for other stopping events. Triggering events are added to the queue with event name and occurence time (event_name, event_time)
            if len(self.sim_setup.system_manager.triggered_stopping_event_queue):
                # Pop the event on top of the queue
                event_name, event_time = self.sim_setup.system_manager.triggered_stopping_event_queue.pop()  

                # Check if we already made a decision at this time
                if event_time == self.last_decision_time:
                    continue  # Skip this event as a decision was already made for this time 

                self.sim_setup.logger.info(f"Triggered event: {event_name}")

                # Store the current simulation time when making a decision
                self.last_decision_time = event_time            

                # Return the states
                return self.return_state_reward_mask()
            
            # Check for periodic stop event
            if periodic_stop_event.triggered:
                # Check if we already made a decision at this time
                if self.env.now == self.last_decision_time:
                    continue  # Skip the periodic stop event if a decision was already made
                
                self.sim_setup.logger.info(f"Periodic time step: {miliseconds_to_hms(self.env.now)}")
                
                # Store the current simulation time when making a decision
                self.last_decision_time = self.env.now

                return self.return_state_reward_mask()       

        self.finalize_simulation()
        return self.get_current_state(), round(self.reward_function.reward, 3), self.is_terminated(), self.is_truncated(), self.action_mask(final_state=True)

    def _should_continue_simulation(self):
        if self.env.peek() >= self.max_sim_time:
            self.terminal_event.succeed()
        return (self.env.peek() < self.max_sim_time and 
                (not self.terminal_event.triggered and 
                not self.truncation_event.triggered and
                not self._check_demand_satisfied()))
    
    def return_state_reward_mask(self):
        """
        Return the state, reward, and action mask.
        """
        if self.config['external_optimization_params']['periodic_time_step'] != 60:
            # Need to calculate spill separeately
            self.reward_function.add_spill_cost()

        reward = round(self.reward_function.reward, 3)
        current_state = self.get_current_state()
        action_mask = self.action_mask()
        self.reward_function.reset_rewards()

        self.update_reward_tracker(reward)

        # Convert the state to human readable format: Convert aircraft status to aircraft state ENUMs, convert aircraft od_pair to vertiport ids and flight directions
        self.sim_setup.logger.info(f"Reward: {reward}")
        self.sim_setup.logger.info(f"Current state: {pformat(current_state, width=150)}")
        masks = np.array(action_mask).reshape(self.num_aircraft, -1)
        masks_log = {i: masks[i].tolist() for i in range(self.num_aircraft)}
        self.sim_setup.logger.debug(f"Action mask: {masks_log}")    
        return current_state, reward, self.is_terminated(), self.is_truncated(), action_mask    
    
    def _check_demand_satisfied(self):
        # If system_manager.passenger_arrival_complete == True and if there is no passenger left in the waiting rooms of the vertiports, succeed the terminal event
        # self.sim_setup.logger.warning(f"Is passenger arrival complete: {self.sim_setup.system_manager.passenger_arrival_complete}")
        # self.sim_setup.logger.warning(f"Is all waiting rooms empty: {self.sim_setup.system_manager.check_all_waiting_rooms_empty()}")
        # self.sim_setup.logger.warning(f"Is all passenger travelled: {self.sim_setup.system_manager.is_all_passenger_travelled()}")
        # self.sim_setup.logger.warning(f"Total travelled passengers: {self.sim_setup.system_manager.trip_counter_tracker}")
        if self.config['external_optimization_params']['spill_optimization']:
            # Check if last passenger has arrived and all waiting rooms are empty
            if bool(self.sim_setup.system_manager.passenger_arrival_complete 
                    and self.sim_setup.system_manager.check_all_waiting_rooms_empty()):
                self.sim_setup.logger.info(self.sim_setup.log_brief_metrics())
                self.terminal_event.succeed()                
                return True
        else:
            if bool(
                self.sim_setup.system_manager.passenger_arrival_complete
                and self.sim_setup.system_manager.check_all_waiting_rooms_empty()
                and self.sim_setup.system_manager.is_all_passenger_travelled()):
                # self.sim_setup.logger.warning("Demand satisfied")
                self.sim_setup.logger.info(self.sim_setup.log_brief_metrics())
                self.terminal_event.succeed()
                return True
        return False
    
    def _reset_event(self, event_name):
        """
        Reset the event by creating a new event.
        """
        self.stopping_events[event_name] = self.env.event()

    def _handle_end_of_simulation(self):
        if self.env.peek() < self.max_sim_time and not self.terminal_event.triggered:
            self.terminal_event.succeed()

    def is_terminated(self):
        """
        Check if the simulation has terminated.
        """
        return self.terminal_event.triggered
    
    def is_truncated(self):
        """
        Check if the simulation has truncated.
        """
        return self.truncation_event.triggered            

    def get_current_state(self):
        """
        Get the current state of the simulation.
        """
        return get_simulator_states(vertiports=self.sim_setup.vertiports,
                                    aircraft_agents=self.sim_setup.system_manager.aircraft_agents,
                                    num_initial_aircraft=self.sim_setup.system_manager.num_initial_aircraft,
                                    simulation_states=self.config['sim_params']['simulation_states'],
                                    reward_function_parameters=self.config['external_optimization_params']['reward_function_parameters'])
    
    def get_initial_state(self):
        """
        Reset the simulation.
        """
        # print("Gettting the instance initial state")
        while self.sim_setup.system_manager.get_available_aircraft_count() < self.sim_setup.system_manager.num_initial_aircraft:          
            self.env.step()
            while len(self.sim_setup.system_manager.triggered_stopping_event_queue)>0:   
                # Pop them out from the triggered event queue
                self.sim_setup.system_manager.triggered_stopping_event_queue.pop() 
        # Set the total demand after initialization
        self.total_demand = self.sim_setup.system_manager.total_demand
        return self.get_current_state()
    
    def update_reward_tracker(self, reward):
        """
        Update the reward tracker.
        """
        # Increase the step by one
        self.sim_setup.event_saver.rl_reward_tracker['step'] += 1
        # Add the reward to the cumulative sum
        self.sim_setup.event_saver.rl_reward_tracker['total_reward'] += reward
        # Update the mean reward
        self.sim_setup.event_saver.rl_reward_tracker['mean_reward'] = \
            self.sim_setup.event_saver.rl_reward_tracker['total_reward'] / self.sim_setup.event_saver.rl_reward_tracker['step']

    @staticmethod
    def multidiscrete_to_discrete(action, n_actions=3):
        """Converts a MultiDiscrete action to a Discrete action."""
        discrete_action = 0
        for i, a in enumerate(reversed(action)):
            discrete_action += a * (n_actions ** i)
        return discrete_action

    @staticmethod
    def discrete_to_multidiscrete(action, dimensions=4, n_actions=3):
        """Converts a Discrete action back to a MultiDiscrete action."""
        multidiscrete_action = []
        for _ in range(dimensions):
            multidiscrete_action.append(action % n_actions)
            action = action // n_actions
        return list(reversed(multidiscrete_action))   
    
    def get_aircraft_count(self) -> int:
        return sum(
            vertiport['aircraft_arrival_process']['num_initial_aircraft_at_vertiport']
            for _, vertiport in self.config['network_and_demand_params']['vertiports'].items()
        )

    def get_vertiport_count(self) -> int:
        return len(self.sim_setup.vertiport_ids)
    
    def get_vertiport_ids(self) -> list:
        return list(self.config['network_and_demand_params']['vertiports'].keys())
    
    def get_action_count(self):
        return self.get_vertiport_count() + 2

    def get_vertiport_state_variable_count(self):
        count = 0
        for state in self.config['sim_params']['simulation_states']['vertiport_states']:
            if state in self.config['sim_params']['per_destination_states']:
                count += self.get_vertiport_count() - 1
            else:
                if state == "waiting_time_bins":
                    # HARD CODED for 290I
                    num_waiting_time_bins = self.config['sim_params']['max_passenger_waiting_time'] // self.config['external_optimization_params']['periodic_time_step']
                    count += num_waiting_time_bins * (self.get_vertiport_count() - 1)
                else:
                    count += 1
        return count

    def get_aircraft_state_variable_count(self):
        return len(self.config['sim_params']['simulation_states']['aircraft_states'])
    
    def get_environmental_state_variable_count(self):
        num_env_vars = len(self.config['sim_params']['simulation_states']['environmental_states'])
        if num_env_vars > 0:
            return num_env_vars
        else:
            return -1
    
    def get_additional_state_variable_count(self):
        return len(self.config['sim_params']['simulation_states']['additional_states'])
    
    def get_passenger_count(self):
        return sum(
            vertiport['passenger_arrival_process']['num_passengers']
            for _, vertiport in self.config['network_and_demand_params']['vertiports'].items()
        )
    
    def get_num_waiting_passengers_per_vertiport(self, vertiport_id):
        """
        Check the number of waiting passengers at the given vertiport.
        """
        return self.sim_setup.vertiports[vertiport_id].get_waiting_passenger_count()

    def print_total_time_to_run(self):
        end_time = time.time()
        time_taken = end_time - self.sim_start_time
        print(f'Simulation Completed. Total time to run: {round(time_taken, 2)} seconds\n')   
    
    def get_performance_metrics(self):
        return fetch_latest_data_from_db(db_path="sqlite/db/vertisimDatabase.sqlite")
    
    def get_aircraft_count(self) -> int:
        return sum(
            vertiport['aircraft_arrival_process']['num_initial_aircraft_at_vertiport']
            for _, vertiport in self.config['network_and_demand_params']['vertiports'].items()
        )
    
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Remove sim_instance to prevent pickling non-picklable objects
    #     state['env'] = None
    #     state['sim_setup'] = None
    #     return state
    
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # Reinitialize sim_instance
    #     self.env = simpy.Environment()
    #     self.terminal_event, self.truncation_event, self.truncation_events, self.stopping_events = self.set_simulation_events(env=self.env, config=self.config)

    #     self.sim_start_time = time.time()
    #     self.sim_setup = SimSetup(env=self.env,
    #                               sim_params=self.config['sim_params'],
    #                               sim_mode=self.config['sim_mode'],
    #                               external_optimization_params=self.config['external_optimization_params'],
    #                               network_and_demand_params=self.config['network_and_demand_params'],
    #                               airspace_params=self.config['airspace_params'],
    #                               passenger_params=self.config['passenger_params'],
    #                               aircraft_params=self.config['aircraft_params'],
    #                               output_params=self.config['output_params'],
    #                               stopping_events=self.stopping_events,
    #                               truncation_events=self.truncation_events,
    #                               truncation_event=self.truncation_event)        
    #     # Set the max simulation time
    #     self.max_sim_time = sec_to_ms(self.sim_setup.sim_params['max_sim_time'])

    #     self.num_aircraft = self.get_aircraft_count()

    #     if self.config['sim_mode']['rl']:
    #         self.setup_for_rl()

    #     self.status = True
    #     self.last_decision_time = -1
    #     self.decision_making_interval = sec_to_ms(self.config['external_optimization_params']['periodic_time_step'])     