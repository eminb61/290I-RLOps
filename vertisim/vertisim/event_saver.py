from collections import defaultdict
from .utils.helpers import timestamp_to_datetime
from .utils.units import ms_to_sec, hr_to_ms
from typing import Dict, Any, Union


class EventSaver:
    """
    eventSaver saves events both for analysis and visualization purposes.
    """

    def __init__(self, env, vertiports, sim_start_timestamp, sim_params, battery_capacity, node_locations, flight_directions):
        self.env = env
        self.vertiports = vertiports
        self.sim_start_timestamp = sim_start_timestamp
        self.sim_params = sim_params
        self.battery_capacity = battery_capacity
        self.node_locations = node_locations
        self.aircraft_agent_trajectory = defaultdict(dict)
        self.passenger_agent_trajectory = defaultdict(dict)
        self.flight_schedule = defaultdict(dict)
        self.passenger_consolidation_time = defaultdict(lambda: defaultdict(dict))
        self.aircraft_allocation_time = defaultdict(dict)
        self.aircraft_arrival_queue_count = defaultdict(dict)
        self.passenger_departure_queue_count = defaultdict(dict)
        self.aircraft_departure_queue_count = defaultdict(dict)
        self.passenger_waiting_time = defaultdict(dict)
        self.passenger_transfer_time = defaultdict(dict)
        self.passenger_departure_queue_waiting_time = defaultdict(dict)
        self.aircraft_holding_time = defaultdict(dict)
        self.aircraft_idle_time_tracker = defaultdict(dict)
        self.aircraft_charging_times = defaultdict(dict)
        self.total_charge_time_counter = 0
        self.aircraft_taxi_times = defaultdict(dict)
        self.fato_usage_tracker = defaultdict(dict)
        self.flight_and_time_tracker = defaultdict(dict)
        self.flight_count_tracker = defaultdict(int)
        self.flight_duration_tracker = defaultdict(dict)
        self.load_factor_tracker = defaultdict(list)
        self.passenger_count_tracker = defaultdict(int)
        self.passenger_time_tracker = defaultdict(dict)
        self.repositioning_counter = defaultdict(dict)
        self.aircraft_energy_consumption_tracker = defaultdict(dict)
        self.flight_phase_energy_consumption_tracker = defaultdict(lambda: defaultdict(dict))
        self.ground_aircraft_counter = defaultdict(dict)
        self.flying_aircraft_counter = defaultdict(dict)
        self.passenger_trip_time_tracker = defaultdict(dict)
        self.aircraft_process_time_tracker = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.average_energy_consumption_per_od_pair = {flight_direction: {'sum': 0, 'count': 0, 'average': 15} for flight_direction in flight_directions}
        self.spilled_passenger_tracker = {flight_direction: 0 for flight_direction in flight_directions}
        self.rl_reward_tracker = {'step': 0, 'mean_reward': 0, 'total_reward': 0}
        self.ondemand_reward_tracker = {'total_reward': 0}


        for vertiport_id, _ in self.vertiports.items():
            self.aircraft_arrival_queue_count[vertiport_id][self.sim_start_timestamp] = 0
            self.passenger_departure_queue_count[vertiport_id][self.sim_start_timestamp] = 0
            self.aircraft_departure_queue_count[vertiport_id][self.sim_start_timestamp] = 0
            self.fato_usage_tracker[vertiport_id][self.sim_start_timestamp] = 0
            self.flight_count_tracker[vertiport_id] = 0
            self.passenger_count_tracker[vertiport_id] = 0
            self.passenger_time_tracker[vertiport_id][self.sim_start_timestamp] = 0
            self.repositioning_counter[vertiport_id] = 0
        self.flying_aircraft_counter[self.sim_start_timestamp] = 0
    
    def performance_metrics(self):
        return {
            'aircraft_agent_trajectory': self.aircraft_agent_trajectory,
            'passenger_agent_trajectory': self.passenger_agent_trajectory,
            'flight_schedule': self.flight_schedule,
            'passenger_consolidation_time': self.passenger_consolidation_time,
            'aircraft_allocation_time': self.aircraft_allocation_time,
            'aircraft_arrival_queue_count': self.aircraft_arrival_queue_count,
            'passenger_departure_queue_count': self.passenger_departure_queue_count,
            'aircraft_departure_queue_count': self.aircraft_departure_queue_count,
            'passenger_waiting_time': self.passenger_waiting_time,
            'passenger_transfer_time': self.passenger_transfer_time,
            'passenger_departure_queue_waiting_time': self.passenger_departure_queue_waiting_time,
            'aircraft_holding_time': self.aircraft_holding_time,
            'aircraft_idle_time_tracker': self.aircraft_idle_time_tracker,
            'aircraft_charging_times': self.aircraft_charging_times,
            'aircraft_taxi_times': self.aircraft_taxi_times,
            'fato_usage_tracker': self.fato_usage_tracker,
            'spilled_passenger_tracker': self.spilled_passenger_tracker,
            'flight_and_time_tracker': self.flight_and_time_tracker,
            'flight_count_tracker': self.flight_count_tracker,
            'flight_duration_tracker': self.flight_duration_tracker,
            'load_factor_tracker': self.load_factor_tracker,
            'passenger_count_tracker': self.passenger_count_tracker,
            'passenger_time_tracker': self.passenger_time_tracker,
            'aircraft_energy_consumption_tracker': self.aircraft_energy_consumption_tracker,
            'ground_aircraft_counter': self.ground_aircraft_counter,
            'repositioning_counter': self.repositioning_counter,
            'flying_aircraft_counter': self.flying_aircraft_counter,
            'total_charge_time_counter': self.total_charge_time_counter,
            'passenger_trip_time_tracker': self.passenger_trip_time_tracker,
            'aircraft_process_time_tracker': self.aircraft_process_time_tracker,
            'flight_phase_energy_consumption_tracker': self.flight_phase_energy_consumption_tracker,
            'rl_reward_tracker': self.rl_reward_tracker,
            'ondemand_reward_tracker': self.ondemand_reward_tracker
        }

    def update_flying_aircraft_counter(self, flight_update: int):
        if self.sim_params['light_simulation']:
            return
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                     sim_time=self.env.now)
        time = self.is_time_overlapping(time=time, agent_type='tracker', tracker=self.flying_aircraft_counter)
        last_value = list(self.flying_aircraft_counter.values())[-1]
        self.flying_aircraft_counter[time] = last_value + flight_update 

    def update_ground_aircraft_counter(self, aircraft_count: Dict):
        if self.sim_params['light_simulation']:
            return
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                     sim_time=self.env.now)
        time = self.is_time_overlapping(time=time, agent_type='tracker', tracker=self.ground_aircraft_counter)
        self.ground_aircraft_counter[time] = aircraft_count

    def update_repositioning_counter(self, vertiport_id, repositioning_count: int):
        self.repositioning_counter[vertiport_id] += repositioning_count

    def update_aircraft_energy_consumption_tracker(self, flight_direction, energy_consumption):
        """
        Saves and updates the energy consumption (SoC %) of the aircraft.
        """
        if self.sim_params['light_simulation'] and len(self.aircraft_energy_consumption_tracker[flight_direction]) > 20:
            return
        
        # Convert the energy consumption from kWh to SoC
        energy_consumption_soc = energy_consumption * self.battery_capacity / 100
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                     sim_time=self.env.now)
        time = self.is_time_overlapping(time=time, agent_type='tracker', tracker=self.aircraft_energy_consumption_tracker[flight_direction])
        self.aircraft_energy_consumption_tracker[flight_direction][time] = energy_consumption_soc

        self.update_average_energy_consumption(flight_direction, energy_consumption_soc)

    def update_average_energy_consumption(self, flight_direction: str, new_energy_consumption: float, alpha: float = 0.1):
        """
        Update the average energy consumption using an exponentially weighted average (EWA).
        
        Args:
        - flight_direction: The direction of the flight (e.g., OD pair).
        - new_energy_consumption: The energy consumption for the latest flight in SoC%.
        - alpha: The weighting factor (0 < alpha <= 1). Default is 0.1.
        """
        # Check if the flight direction is in the dictionary
        if flight_direction in self.average_energy_consumption_per_od_pair:
            # Get the current average energy consumption
            previous_average = self.average_energy_consumption_per_od_pair[flight_direction]['average']
            
            # Calculate the new exponentially weighted average
            new_average_energy_consumption = round((1 - alpha) * new_energy_consumption + alpha * previous_average, 2)
            
            # Update the average energy consumption with the new value
            self.average_energy_consumption_per_od_pair[flight_direction]['average'] = new_average_energy_consumption

    def save_flight_phase_energy(self, flight_direction, flight_id, flight_phase, energy):
        if self.sim_params['light_simulation'] or len(self.flight_phase_energy_consumption_tracker[flight_direction]) > 20:
            return
        # If the value exists, add the new value to the existing value
        if flight_phase in self.flight_phase_energy_consumption_tracker[flight_direction][flight_id]:
            self.flight_phase_energy_consumption_tracker[flight_direction][flight_id][flight_phase] += energy
        else:
            self.flight_phase_energy_consumption_tracker[flight_direction][flight_id][flight_phase] = energy
    
    def update_flight_duration_tracker(self, flight_direction, flight_duration):
        if self.sim_params['light_simulation'] and len(self.flight_duration_tracker[flight_direction]) > 20:
            return
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                     sim_time=self.env.now)
        self.flight_duration_tracker[flight_direction][time] = flight_duration

    def update_flight_tracker(self, origin, destination):
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                     sim_time=self.env.now)
        time = self.is_time_overlapping(time=time, agent_type='aircraft')
        self.flight_and_time_tracker[origin][time] = 1
        self.flight_count_tracker[f"{origin}_{destination}"] += 1

    def update_load_factor_tracker(self, vertiport_id, load_factor):
        if not self.sim_params['light_simulation']:
            time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                        sim_time=self.env.now)
            time = self.is_time_overlapping(time=time, agent_type='aircraft')
            self.passenger_time_tracker[vertiport_id][time] = load_factor
            if load_factor > 0:
                self.load_factor_tracker[vertiport_id].append(load_factor)
                self.passenger_count_tracker[vertiport_id] += load_factor
    
    def update_fato_usage_tracker(self, vertiport_id, fato_usage):
        if not self.sim_params['light_simulation']:
            time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                        sim_time=self.env.now)
            time = self.is_time_overlapping(time=time, agent_type='passenger')
            # self.fato_usage_tracker[time]['fato_id'] = fato_id
            # Get the last value from the dictionary and add the queue_update to it
            last_value = list(self.fato_usage_tracker[vertiport_id].values())[-1]
            self.fato_usage_tracker[vertiport_id][time] = last_value + fato_usage

    def save_passenger_group_consolidation_time(self, vertiport_id, passenger_ids, wr_number, consolidation_time):
        if not self.sim_params['light_simulation']:
            time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                        sim_time=self.env.now)
            time = self.is_time_overlapping(time=time, agent_type='passenger')
            self.passenger_consolidation_time[vertiport_id][time]['passenger_ids'] = passenger_ids
            self.passenger_consolidation_time[vertiport_id][time]['wr_number'] = wr_number
            self.passenger_consolidation_time[vertiport_id][time]['consolidation_time'] = consolidation_time
    
    def save_aircraft_allocation_time(self, vertiport_id, tail_number, allocation_time):
        if not self.sim_params['light_simulation']:
            self.aircraft_allocation_time[vertiport_id][tail_number] = allocation_time

    def save_passenger_waiting_time(self, vertiport_id: Any, passenger_id: Any, waiting_time: float) -> None:
        self.passenger_waiting_time[vertiport_id][passenger_id] = waiting_time

    def save_passenger_transfer_time(self, vertiport_id, passenger_id, transfer_time):
        if not self.sim_params['light_simulation']:
            self.passenger_transfer_time[vertiport_id][passenger_id] = transfer_time

    def save_passenger_trip_time(self, agent_id, flight_direction, trip_time):
        self.passenger_trip_time_tracker[flight_direction][agent_id] = trip_time        

    def save_passenger_departure_queue_waiting_time(self, vertiport_id, passenger_id, boarding_time):
        if not self.sim_params['light_simulation']:
            self.passenger_departure_queue_waiting_time[vertiport_id][passenger_id] = self.env.now - boarding_time

    def update_passenger_departure_queue_counter(self, vertiport_id, queue_update: int) -> None:
        if not self.sim_params['light_simulation']:
            time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                        sim_time=self.env.now)
            time = self.is_time_overlapping(time=time, agent_type='passenger')
            # Get the last value from the dictionary and add the queue_update to it
            last_value = list(self.passenger_departure_queue_count[vertiport_id].values())[-1]
            self.passenger_departure_queue_count[vertiport_id][time] = last_value + queue_update

    def update_spilled_passenger_counter(self, flight_dir) -> None:
        self.spilled_passenger_tracker[flight_dir] += 1

    def update_aircraft_arrival_queue_counter(self, vertiport_id: Any, queue_update: int) -> None:
        if self.sim_params['light_simulation']:
            return
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                     sim_time=self.env.now)
        time = self.is_time_overlapping(time=time, agent_type='aircraft')
        # Get the last value from the dictionary and add the queue_update to it
        last_value = list(self.aircraft_arrival_queue_count[vertiport_id].values())[-1]
        self.aircraft_arrival_queue_count[vertiport_id][time] = last_value + queue_update

    def update_aircraft_departure_queue_counter(self, 
                                                vertiport_id: str, 
                                                queue_update: int) -> None:
        if self.sim_params['light_simulation']:
            return
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                     sim_time=self.env.now)
        time = self.is_time_overlapping(time=time, agent_type='aircraft')
        # Get the last value from the dictionary and add the queue_update to it
        last_value = list(self.aircraft_departure_queue_count[vertiport_id].values())[-1]
        self.aircraft_departure_queue_count[vertiport_id][time] = last_value + queue_update

    def save_aircraft_holding_time(self, 
                                   vertiport_id: str, 
                                   waiting_time: float) -> None:
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                     sim_time=self.env.now)
        self.aircraft_holding_time[vertiport_id][time] =  waiting_time

    def save_aircraft_charging_time(self,
                                    vertiport_id: str,
                                    charging_time: float) -> None:
        if self.sim_params['light_simulation']:
            return
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                     sim_time=self.env.now)        
        self.aircraft_charging_times[vertiport_id][time] = charging_time
        self.total_charge_time_counter += charging_time

    def save_aircraft_taxi_time(self, 
                                vertiport_id: str, 
                                taxi_time: float) -> None:
        if self.sim_params['light_simulation']:
            return
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                     sim_time=self.env.now)        
        self.aircraft_taxi_times[vertiport_id][time] = taxi_time
    
    def save_aircraft_idle_time(self, 
                                vertiport_id: Any, 
                                idle_time: int) -> None:
        if self.sim_params['light_simulation']:
            return
        time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                    sim_time=self.env.now)
        time = self.is_time_overlapping(time=time, agent_type='aircraft')
        self.aircraft_idle_time_tracker[vertiport_id][time] = idle_time
    
    def save_aircraft_process_times(self, 
                                    agent_id: str, 
                                    flight_direction: str, 
                                    flight_id: str, 
                                    event: str, 
                                    process_time: float) -> None:
        if not self.sim_params['light_simulation']:
            # If there is a value for the event, then add the process time to it
            if event in self.aircraft_process_time_tracker[agent_id][flight_direction][flight_id]:
                self.aircraft_process_time_tracker[agent_id][flight_direction][flight_id][event] += process_time
            else:
                self.aircraft_process_time_tracker[agent_id][flight_direction][flight_id][event] = process_time

    def save_agent_state(self,
                         agent: object,
                         agent_type: str,
                         event: str = None) -> None:
        """
        Saves the agent state
        :param agent:
        :param agent_type:
        :param event:
        """          
        if self.sim_params['save_trajectories']:
            time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                        sim_time=self.env.now)
            time = round(time)
            time = self.is_time_overlapping(time=time, agent_type=agent_type)
            if agent_type == 'aircraft':
                self.aircraft_agent_trajectory[time]['tail_number'] = f'{agent_type}{agent.tail_number}'
                self.aircraft_agent_trajectory[time]['location'] = agent.location
                self.aircraft_agent_trajectory[time]['speed'] = agent.forward_velocity
                self.aircraft_agent_trajectory[time]['soc'] = agent._soc
                self.aircraft_agent_trajectory[time]['flight_direction'] = agent.flight_direction
                self.aircraft_agent_trajectory[time]['event'] = event
                self.aircraft_agent_trajectory[time]['latitude'] = self.node_locations[agent.location][0]
                self.aircraft_agent_trajectory[time]['longitude'] = self.node_locations[agent.location][1]  
                self.aircraft_agent_trajectory[time]['altitude'] = self.node_locations[agent.location][2]                         
            elif agent_type == 'passenger':
                self.passenger_agent_trajectory[time]['passenger_id'] = f'{agent_type}{agent.passenger_id}'
                self.passenger_agent_trajectory[time]['latitude'] = self.node_locations[agent.location][0]
                self.passenger_agent_trajectory[time]['longitude'] = self.node_locations[agent.location][1]
                self.passenger_agent_trajectory[time]['altitude'] = self.node_locations[agent.location][2]
                self.passenger_agent_trajectory[time]['location'] = agent.location
                self.aircraft_agent_trajectory[time]['flight_direction'] = f'{agent.origin_vertiport_id}_{agent.destination_vertiport_id}'
                self.passenger_agent_trajectory[time]['event'] = event
            else:
                raise ValueError(f'Agent type {agent_type} is not supported')


    def save_flight_info(self, 
                         origin_vertiport_id: Any, 
                         destination_vertiport_id: Any,
                         tail_number: int, 
                         flight_id: str,
                         aircraft_pushback_time: float,
                         passengers: list, 
                         aircraft_model: str, 
                         event_type: str):
        """
        Saves the flight information
        :param tail_number:
        :param passengers:
        :param aircraft_model:
        :param event_type:
        :return:
        """
        if not self.sim_params['light_simulation']:
            time = timestamp_to_datetime(sim_start_time=self.sim_start_timestamp,
                                        sim_time=self.env.now)
            time = self.is_time_overlapping(time=time, agent_type='aircraft')
            self.flight_schedule[time]['tail_number'] = f'aircraft{tail_number}'
            self.flight_schedule[time]['flight_id'] = flight_id
            self.flight_schedule[time]['origin_vertiport_id'] = origin_vertiport_id
            self.flight_schedule[time]['destination_vertiport_id'] = destination_vertiport_id
            self.flight_schedule[time]['aircraft_pushback_time'] = ms_to_sec(aircraft_pushback_time)
            self.flight_schedule[time]['passengers'] = passengers
            self.flight_schedule[time]['event'] = event_type
            self.flight_schedule[time]['aircraft_model'] = aircraft_model

    def is_time_overlapping(self, time: int, agent_type: str, tracker: Dict = None) -> int:
        """
        eventSaver saves each event based on the time. Events might happening in parallel. So, the eventSaver might be
        overwriting events on top of each other. To prevent this, is_time_overlapping adds 1 milisecond to the
        event's time if there are any events that happened at the same time.
        """
        # TODO: Improve this function by checking the last few elements of the trajectory dictionary
        if agent_type == 'aircraft':
            if len(self.aircraft_agent_trajectory) == 0:
                return time
            if time not in list(self.aircraft_agent_trajectory.keys()):
                return time
            time += 1  # add 1 milisecond
            return self.is_time_overlapping(time=time, agent_type=agent_type)
        elif agent_type == 'passenger':
            if len(self.passenger_agent_trajectory) == 0:
                return time
            if time not in list(self.passenger_agent_trajectory.keys()):
                return time
            time += 1
            return self.is_time_overlapping(time=time, agent_type=agent_type)
        elif agent_type == 'tracker':
            if len(tracker) == 0:
                return time
            if time not in list(tracker.keys()):
                return time
            time += 1
            return self.is_time_overlapping(time=time, agent_type=agent_type, tracker=tracker)
        else:
            raise ValueError('agent_type must be either aircraft, passenger or tracker')
