import simpy
import random
from typing import Dict, Any, Union, List, Optional
from collections import defaultdict
from .utils.helpers import flatten, extract_dict_values
from .aircraft.aircraft import AircraftStatus
import numpy as np
from enum import Enum
from .utils.units import ms_to_min, ms_to_sec, ms_to_hr, min_to_ms, sec_to_ms, sec_to_min
from .utils.read_files import read_input_file
from .utils.helpers import miliseconds_to_hms, expected_pax_arrivals
from pydantic import BaseModel

class EntitySubState(BaseModel):
    num_jobs_in_service: int
    num_jobs_in_queue: int

class VertiportStates(BaseModel):
    num_waiting_passengers: Dict[str, int]
    num_spilled_passengers: Dict[str, int]
    # fato: Optional[EntitySubState] = None
    # park: EntitySubState = None
    # taxi: EntitySubState = None
    # arrival_fix: EntitySubState
    # departure_fix: EntitySubState
    # num_aircraft: int
    # num_flight_requests: int
    # num_available_fato: int
    # num_available_parking_pad: int
    # num_available_charger: int
    # total_pax_waiting_time_per_od: Dict[str, float]
    expected_pax_arr_per_od: Dict[str, float]
    # aircraft_status: Dict[int, int]
    # num_holding_aircraft: int
    # avg_pax_waiting_time: Dict[str, float]
    # std_pax_waiting_time_per_od: Dict[str, float]
    # time_to_next_available_charger: float
    waiting_time_bins: Dict[str, List[int]]

class VertiportConfig:

    def __init__(self,
                 env: simpy.Environment,
                 vertiport_id: Any,
                 vertiport_data: dict,
                 vertiport_layout: dict,
                 vertiport_ids: list,
                 network_and_demand: dict,
                 aircraft_capacity: int,
                 pax_waiting_time_threshold: int = 600,
                 num_waiting_time_bins: int = 10):
        self.env = env
        self.vertiport_id = vertiport_id
        self.vertiport_data = vertiport_data
        self.vertiport_layout = vertiport_layout
        self.parking_pad_ids = vertiport_layout.parking_pad_ids
        self.vertiport_ids = vertiport_ids
        self.network_and_demand = network_and_demand
        self.aircraft_capacity = aircraft_capacity
        self.pax_waiting_time_threshold = pax_waiting_time_threshold
        self.num_waiting_time_bins = num_waiting_time_bins
        self.charger_max_charge_rate = self.network_and_demand['vertiports'][self.vertiport_id]['charger_max_charge_rate']
        self.charger_efficiency = self.network_and_demand['vertiports'][self.vertiport_id]['charger_efficiency']
        self.demand_type = self.get_demand_type()         
        self.pax_arrival_df = None
        self.vertiport_index = self.map_vertiport_id_to_index(vertiport_id)        
        self.shared_charger_sets = self.vertiport_data['shared_charger_sets']
        self.fato_store = self.add_fato_ids_to_store()
        self.parking_space_store = self.check_charger_resource_type()
        self.taxiway_node_store = self.add_taxiway_nodes_to_store()
        self.edge_store = self.add_edges_to_store()
        self.charger_resource = simpy.Resource(self.env, capacity=self.vertiport_data['num_chargers'])
        self.waiting_room_stores = self.create_waiting_room_stores()
        self.flight_request_stores = self.create_flight_request_stores()
        self.available_aircraft_store = simpy.FilterStore(self.env)
        self.taxi_resource = simpy.Resource(self.env, capacity=1)
        self.arrival_fix_resource = simpy.Resource(self.env, capacity=self.vertiport_data['holding_unit_capacity'])
        self.departure_fix_resource = simpy.Resource(self.env, capacity=1)
        self.holding_times = defaultdict(defaultdict)
        self.active_charging_processes = [] # min-heap for active charging sessions at the vertiport
        self.spilled_passengers_dict = {dest: 0 for dest in self.vertiport_ids if dest != self.vertiport_id}
        self.num_aircraft = 0
        self.aircraft_departure_queue_count = 0
        self.num_flight_requests = 0
        self.num_waiting_passengers = 0
        self.cum_waiting_cost = 0
        self.num_holding_aircraft = 0


    def get_demand_type(self):
        if (self.network_and_demand['passenger_schedule_file_path']) or (self.network_and_demand['pick_demand_files_path']) or (self.network_and_demand['pick_rate_files_path']):
            return 'scheduled'
        elif (self.network_and_demand['passenger_arrival_rates_path']):
            return 'rate_based'
        elif self.network_and_demand['autoregressive_demand_files_path']:
            return 'autoregressive'
        else:
            raise ValueError("Please provide a valid demand.")  
        
    def is_node_vertiport_location(self, location):
        return (self.is_taxi_node_location(location) or 
                self.is_fato_location(location) or
                self.is_parking_pad_location(location))

    def is_taxi_node_location(self, location):
        return location in self.vertiport_layout.taxiway_node_ids
    
    def is_fato_location(self, location):
        return location in self.vertiport_layout.fato_ids

    def is_parking_pad_location(self, location):
        return location in self.parking_pad_ids
    
    def map_vertiport_id_to_index(self, vertiport_id: str) -> int:
        """
        Returns the index of the vertiport ID in the vertiport_ids list.
        """
        return self.vertiport_ids.index(vertiport_id)
    
    def get_parking_pad_availability(self):
        """
        Returns the number of available parking pads.
        """
        return len(self.parking_space_store.items)
    
    def get_charger_availability(self):
        """
        Returns the number of available chargers.
        """
        return self.charger_resource.capacity - len(self.active_charging_processes)
    
    def get_parking_pad_occupancy(self):
        """
        Returns the number of occupied parking pads.
        """
        return self.vertiport_layout.num_parking_pad - self.get_parking_pad_availability()
    
    def get_parking_pad_queue_length(self):
        """
        Returns the number of aircraft in the parking queue.
        """
        return len(self.parking_space_store.get_queue)
    
    def get_num_fato(self):
        return len(self.fato_store.items)
    
    def get_fato_occupancy(self):
        return sum(fato.fato_resource.count for fato in self.fato_store.items)
    
    def get_fato_queue_length(self):
        return sum(len(fato.fato_resource.queue) for fato in self.fato_store.items)
    
    def get_next_available_charger_time(self):
        # Check if there are no chargers
        if self.charger_resource.capacity == 0:
            return -1

        # Check if any charger is immediately available
        if len(self.active_charging_processes) < self.charger_resource.capacity:
            return 0  # There is at least one available charger right now
        
        # The heap's smallest element is the next charger to be available
        next_available_time = self.active_charging_processes[0]  # First element of tuple is the finish time

        # Factor in the charger queue
        queued_processes = len(self.charger_resource.queue)
        if self.charger_resource.count == self.charger_resource.capacity and queued_processes > 0:
            # Get the index of the queued_process'ed order as the next_available_time
            # Ensure there are enough active charges to match the queue length
            index = min(queued_processes, len(self.active_charging_processes) - 1)
            next_available_time = self.active_charging_processes[index]

        return max(0, ms_to_min(next_available_time - self.env.now))
    
    def get_holding_time_cost(self, unit: str='minute') -> float:
        # Compute the waiting times by substracting the current time from the holding start times in the holding_times
        if len(self.holding_times) == 0:
            return 0
        holding_times = [self.env.now - holding_time for holding_time in extract_dict_values(self.holding_times)]
        return VertiportConfig.unit_helper(unit=unit, value=sum(holding_times))
    
    def get_pax_waiting_time_bins(self) -> Dict:
        """
        Returns the number of passengers in each waiting time bin for each flight direction.
        The number of bins is provided externally using self.num_waiting_time_bins.
        Passengers with waiting times equal to or exceeding the threshold are placed in the last bin.
        """
        # Calculate bin_size based on the threshold and the number of bins
        bin_size = int(sec_to_min(self.pax_waiting_time_threshold)/ self.num_waiting_time_bins)

        # Initialize the waiting time bins for each destination
        waiting_time_bins = {
            destination: [0 for _ in range(self.num_waiting_time_bins)]
            for destination in self.waiting_room_stores.keys()
        }

        # Process passengers in the waiting rooms and flight request stores
        for store_type in [self.waiting_room_stores, self.flight_request_stores]:
            for destination, store in store_type.items():
                for passenger in store.items:
                    waiting_time = ms_to_min(self.env.now - passenger.waiting_room_arrival_time)
                    
                    # Place passengers in bins, with those waiting exactly 10 minutes in the last bin
                    if waiting_time < 10:
                        bin_index = int(waiting_time // bin_size)
                    else:
                        bin_index = self.num_waiting_time_bins - 1
                    
                    waiting_time_bins[destination][bin_index] += 1

        return waiting_time_bins

    
    def get_waiting_time_cost(self, unit:str, last_decision_time: float, type:str, exclude_pax_ids: list=[]):
        """
        Returns the waiting cost of the vertiport.

        Parameters
        ----------
        unit : str
            The unit of the waiting time cost. One of the following: 'minute', 'second', 'millisecond', 'hour'.
        last_decision_time : float 
            The time of the last decision. If True, the waiting time cost is computed using per time step, 
            NOT total waiting time from passenger waiting room arrival time.
        type : str
            The type of the waiting time cost. One of the following: 'linear', 'exponential'.
        exclude_pax_ids : list
            The list of passenger IDs to exclude from the waiting time cost.
        """
        if last_decision_time:
            if type == 'linear':
                return self.get_linear_per_step_waiting_cost(last_decision_time, 
                                                             unit=unit, 
                                                             exclude_pax_ids=exclude_pax_ids)
            elif type == 'exponential':
                return self.get_exponential_per_step_waiting_cost(last_decision_time, 
                                                                  unit=unit, 
                                                                  exclude_pax_ids=exclude_pax_ids)
            else:
                raise ValueError("Invalid waiting time cost type. Please input one of the following: 'linear', 'exponential'.")
        else:
            if type == 'linear':
                return self.get_total_linear_waiting_cost(unit=unit, 
                                                          exclude_pax_ids=exclude_pax_ids)
            elif type == 'exponential':
                return self.get_total_exponential_waiting_cost(unit=unit, 
                                                               exclude_pax_ids=exclude_pax_ids)
            else:
                raise ValueError("Invalid waiting time cost type. Please input one of the following: 'linear', 'exponential'.")

    def get_linear_per_step_waiting_cost(self, last_decision_time: float, unit: str='hour', exclude_pax_ids: list=[]) -> float:
        """
        Returns the total waiting time of all passengers in the vertiport in the given unit.
        Waiting time is computed using per time step, NOT total waiting time from 
        passenger waiting room arrival time.

        If exclude_pax_ids are provided then we exlude them from the waiting time cost.        
        """
        total_waiting_time = 0
        for waiting_room_store in self.waiting_room_stores.values():
            for passenger in waiting_room_store.items:
                if passenger.passenger_id not in exclude_pax_ids:
                    if self.env.now - passenger.waiting_room_arrival_time != 0:
                        total_waiting_time += self.env.now - last_decision_time
                    else:
                        total_waiting_time += self.env.now - passenger.waiting_room_arrival_time
        for flight_request_store in self.flight_request_stores.values():
            for passenger in flight_request_store.items:
                if passenger.passenger_id not in exclude_pax_ids:
                    if self.env.now - passenger.waiting_room_arrival_time != 0:
                        total_waiting_time += self.env.now - last_decision_time
                    else:
                        total_waiting_time += self.env.now - passenger.waiting_room_arrival_time

        return VertiportConfig.unit_helper(unit=unit, value=total_waiting_time)
    
    def get_exponential_per_step_waiting_cost(self, last_decision_time: float, unit: str='hour', exclude_pax_ids: list=[]) -> float:
        """
        Returns the total waiting time of all passengers in the vertiport in the given unit.
        Waiting time is computed using per time step, NOT total waiting time from
        passenger waiting room arrival time.

        If exclude_pax_ids are provided then we exlude them from the waiting time cost.
        """
        total_weighted_waiting_time = 0
        for waiting_room_store in self.waiting_room_stores.values():
            for passenger in waiting_room_store.items:
                if passenger.passenger_id not in exclude_pax_ids:
                    waiting_time = self.env.now - passenger.waiting_room_arrival_time
                    if waiting_time > sec_to_ms(self.pax_waiting_time_threshold):
                        total_weighted_waiting_time += (self.env.now - last_decision_time)**1.1
                    else:
                        if self.env.now - passenger.waiting_room_arrival_time != 0:
                            total_weighted_waiting_time += self.env.now - last_decision_time
                        else:
                            total_weighted_waiting_time += self.env.now - passenger.waiting_room_arrival_time
        
        for flight_request_store in self.flight_request_stores.values():
            for passenger in flight_request_store.items:
                if passenger.passenger_id not in exclude_pax_ids:
                    waiting_time = self.env.now - passenger.waiting_room_arrival_time
                    if waiting_time > sec_to_ms(self.pax_waiting_time_threshold):
                        total_weighted_waiting_time += (self.env.now - last_decision_time)**1.1
                    else:
                        if self.env.now - passenger.waiting_room_arrival_time != 0:
                            total_weighted_waiting_time += self.env.now - last_decision_time
                        else:
                            total_weighted_waiting_time += self.env.now - passenger.waiting_room_arrival_time
        return VertiportConfig.unit_helper(unit=unit, value=total_weighted_waiting_time)
    
    def get_total_linear_waiting_cost(self, unit='hour', exclude_pax_ids: list=[]) -> Dict:
        """
        Returns the total waiting time of all passengers in the vertiport in the given unit.
        """
        total_waiting_time = {destination: 0 for destination in self.waiting_room_stores.keys()}
        for waiting_room_store in self.waiting_room_stores.values():
            for passenger in waiting_room_store.items:
                if passenger.passenger_id not in exclude_pax_ids:
                    total_waiting_time[passenger.destination_vertiport_id] += self.env.now - passenger.waiting_room_arrival_time
        for flight_request_store in self.flight_request_stores.values():
            for passenger in flight_request_store.items:
                if passenger.passenger_id not in exclude_pax_ids:
                    total_waiting_time[passenger.destination_vertiport_id] += self.env.now - passenger.waiting_room_arrival_time
        for destination in total_waiting_time.keys():
            total_waiting_time[destination] = VertiportConfig.unit_helper(unit=unit, value=total_waiting_time[destination])
        return total_waiting_time
    
    def get_total_exponential_waiting_cost(self, unit='hour', exclude_pax_ids: list=[]) -> float:
        """
        Returns the total waiting time of all passengers in the vertiport in the given unit.
        """
        total_weighted_waiting_time = 0
        for waiting_room_store in self.waiting_room_stores.values():
            for passenger in waiting_room_store.items:
                if passenger.passenger_id not in exclude_pax_ids:
                    waiting_time = self.env.now - passenger.waiting_room_arrival_time
                    if waiting_time > sec_to_ms(self.pax_waiting_time_threshold):
                        total_weighted_waiting_time += sec_to_ms(self.pax_waiting_time_threshold) \
                            + (waiting_time - sec_to_ms(self.pax_waiting_time_threshold))**1.1
                    else:
                        total_weighted_waiting_time += waiting_time
            
        for flight_request_store in self.flight_request_stores.values():
            for passenger in flight_request_store.items:
                if passenger.passenger_id not in exclude_pax_ids:
                    waiting_time = self.env.now - passenger.waiting_room_arrival_time
                    if waiting_time > sec_to_ms(self.pax_waiting_time_threshold):
                        total_weighted_waiting_time += sec_to_ms(self.pax_waiting_time_threshold) \
                            + (waiting_time - sec_to_ms(self.pax_waiting_time_threshold))**1.1
                    else:
                        total_weighted_waiting_time += waiting_time
        return VertiportConfig.unit_helper(unit=unit, value=total_weighted_waiting_time)

    @staticmethod
    def unit_helper(unit, value) -> float:
        if unit == 'hour':
            return ms_to_hr(value)
        elif unit == 'minute':
            return ms_to_min(value)
        elif unit == 'second':
            return ms_to_sec(value)
        elif unit == 'millisecond':
            return value
        else:
            raise ValueError("Invalid unit. Please input one of the following: 'minute', 'second', 'millisecond', 'hour'.")
    
    def is_waiting_room_empty(self):
        return all(len(waiting_room_store.items) == 0 for waiting_room_store in self.waiting_room_stores.values())
    
    def get_passenger_count(self):
        """
        Returns the total number of passengers in the vertiport including the passengers in the waiting rooms and 
        the passengers in the flight request stores. Passengers are added to the flight request stores when they
        there are seat_capacity number of passengers in the waiting room or one of the passengers in the waiting room
        has been waiting for more than the waiting time threshold.
        """
        return sum(len(waiting_room_store.items) for waiting_room_store in self.waiting_room_stores.values()) +\
                sum(len(flight_request_store.items) for flight_request_store in self.flight_request_stores.values())
    
    def get_waiting_passenger_ids(self, destination: str, exclude_pax_ids: list=[]):
        """
        Returns the ids of the passengers in the waiting rooms in the given destination up to aircraft capacity
        """
        waiting_passenger_ids = []
        waiting_room_store = self.waiting_room_stores[destination]
        count = 0
        for _, passenger in enumerate(waiting_room_store.items):
            if count < self.aircraft_capacity and passenger.passenger_id not in exclude_pax_ids:
                waiting_passenger_ids.append(passenger.passenger_id)
                count += 1
        return waiting_passenger_ids

    def get_waiting_room_count(self):
        """
        Returns the total number of passengers in the waiting rooms.
        """
        return sum(len(waiting_room_store.items) for waiting_room_store in self.waiting_room_stores.values())
    
    def get_aircraft_status(self, aircraft_agents) -> List:
        # For each aircraft, check if it's at the current vertiport.
        # If it is, get its status; if it's not, the status is 0.
        return {
            aircraft.tail_number: aircraft.status.value if self.is_parking_pad_location(aircraft.location) else 0
            for aircraft in aircraft_agents.values()
        }

    def get_vertiport_states(self, aircraft_agents, reward_function_parameters, departing_passengers=None) -> Dict:
        """
        Will be deprecated in the future.
        """
        entity_states = {'num_waiting_passengers': {}, 'expected_pax_arr_per_od': {}}

        # for fato in self.fato_store.items:
        #     entity_states[fato.resource_id] = {'num_jobs_in_service': fato.fato_resource.count,
        #                                     'num_jobs_in_queue': len(fato.fato_resource.queue)}

        # if self.parking_space_store is not None:
        #     entity_states['park'] = {'num_jobs_in_service': self.get_parking_pad_occupancy(),
        #                             'num_jobs_in_queue': self.get_parking_pad_queue_length()}

        # if self.taxiway_node_store is not None:
        #     entity_states['taxi'] = {'num_jobs_in_service': len(self.taxiway_node_store.items),
        #                             'num_jobs_in_queue': len(self.taxiway_node_store.get_queue)}

        # entity_states['arrival_fix'] = {'num_jobs_in_service': self.arrival_fix_resource.count,
        #                                 'num_jobs_in_queue': len(self.arrival_fix_resource.queue)}

        # entity_states['departure_fix'] = {'num_jobs_in_service': self.departure_fix_resource.count,
        #                                 'num_jobs_in_queue': len(self.departure_fix_resource.queue)}

        # entity_states['num_aircraft'] = self.num_aircraft
        # entity_states['num_flight_requests'] = self.num_flight_requests
        # entity_states['num_available_fato'] = self.get_num_fato() - self.get_fato_occupancy()
        # entity_states['num_available_parking_pad'] = self.get_parking_pad_availability()

        entity_states['num_waiting_passengers'] = self.get_waiting_passenger_count()

        entity_states['num_spilled_passengers'] = self.spilled_passengers_dict        

        # waiting_passenger_counts = self.get_waiting_passenger_count()
        # for destination, waiting_passenger_count in waiting_passenger_counts.items():
        #     entity_states['num_waiting_passengers'][f'to_{destination}'] = waiting_passenger_count
        #     entity_states['expected_pax'][f'to_{destination}'] = expected_pax_arrivals_for_route(arrival_rate_df=self.pax_arrival_df,
        #                                                                                          current_time=self.env.now,
        #                                                                                          look_ahead_time=sec_to_ms(self.network_and_demand['demand_lookahead_time']),
        #    
        #                                                                                       route=f'{self.vertiport_id}_{destination}')

        for destination, expected_pax in self.get_expected_pax_arrivals_for_vertiport().items():
            entity_states['expected_pax_arr_per_od'][destination] = expected_pax

        # entity_states['total_passenger_waiting_time'] = round(self.get_total_waiting_time(reward_function_parameters), 2)

        # entity_states['aircraft_status'] = self.get_aircraft_status(aircraft_agents)

        # entity_states['num_holding_aircraft'] = self.num_holding_aircraft

        # entity_states['avg_pax_waiting_time_per_od'] = self.average_pax_waiting_time(reward_function_parameters)

        # entity_states['std_pax_waiting_time_per_od'] = self.std_pax_waiting_time_per_od(reward_function_parameters)

        # entity_states['num_available_charger'] = self.get_charger_availability()

        # entity_states['total_pax_waiting_time_per_od'] = self.get_total_waiting_time(reward_function_parameters=reward_function_parameters, exclude_pax_ids=departing_passengers)

        entity_states['waiting_time_bins'] = self.get_pax_waiting_time_bins()

        # entity_states['time_to_next_available_charger'] = self.get_next_available_charger_time()

        # Validate and serialize the data using Pydantic
        valid_entity_states = VertiportStates(**entity_states)

        # Return the validated data
        return valid_entity_states.dict()  

    def get_total_waiting_time(self, reward_function_parameters, exclude_pax_ids):
        """
        Returns the waiting time cost of the vertiport.
        """
        return self.get_waiting_time_cost(unit=reward_function_parameters['waiting_time_cost_unit'],
                                          last_decision_time=None,
                                          type=reward_function_parameters['waiting_time_cost_type'],
                                          exclude_pax_ids=exclude_pax_ids)
    
    def average_pax_waiting_time_per_od(self, reward_function_parameters):
        """
        Returns the average pax waiting time at each waiting room
        """
        waiting_times = {destination: 0 for destination in self.waiting_room_stores.keys()}
        for waiting_room_store in self.waiting_room_stores.values():
            for passenger in waiting_room_store.items:
                waiting_times[passenger.destination_vertiport_id] += self.env.now - passenger.waiting_room_arrival_time
        # Calculate the average waiting time for each waiting room
        waiting_pax_count = self.get_waiting_passenger_count()
        for destination in waiting_times.keys():
            if waiting_pax_count[destination] != 0:
                waiting_times[destination] /= waiting_pax_count[destination]
            # Convert the waiting times to the desired unit                
            waiting_times[destination] = round(VertiportConfig.unit_helper(unit=reward_function_parameters['waiting_time_cost_unit'],
                                                                     value=waiting_times[destination]), 2)            
        return waiting_times
    
    def std_pax_waiting_time_per_od(self, reward_function_parameters):
        """
        Returns the standard deviation of pax waiting time at each waiting room
        """
        waiting_times = {destination: [] for destination in self.waiting_room_stores.keys()}
        for waiting_room_store in self.waiting_room_stores.values():
            for passenger in waiting_room_store.items:
                waiting_times[passenger.destination_vertiport_id].append(self.env.now - passenger.waiting_room_arrival_time)
        # Calculate the standard deviation of waiting time for each waiting room
        for destination in waiting_times.keys():
            waiting_times[destination] = max(0, np.std(waiting_times[destination]))

            # Convert the waiting times to the desired unit                
            waiting_times[destination] = round(VertiportConfig.unit_helper(unit=reward_function_parameters['waiting_time_cost_unit'],
                                                                     value=waiting_times[destination]), 2)   
        return waiting_times
    
    def get_expected_pax_arrivals_for_vertiport(self) -> Dict:
        """
        Returns the expected passenger arrivals for each route.
        """
        expected_pax_arrivals_dict = {}
        for destination, _ in self.waiting_room_stores.items():
            expected_pax_arrivals_dict[destination] = expected_pax_arrivals(arrival_df=self.pax_arrival_df,
                                                                            demand_type=self.demand_type,
                                                                            current_time=self.env.now,
                                                                            look_ahead_time=sec_to_ms(self.network_and_demand['demand_lookahead_time']),
                                                                            origin=self.vertiport_id,
                                                                            destination=destination)
        return expected_pax_arrivals_dict
    
    def get_total_expected_pax_arrival_count(self):
        """
        Returns the total expected passenger arrivals for all routes.
        """
        return sum(self.get_expected_pax_arrivals_for_vertiport().values())

    def get_waiting_passenger_count(self):
        waiting_passenger_count = {destination: 0 for destination in self.waiting_room_stores.keys()}
        for destination, waiting_room_store in self.waiting_room_stores.items():
            waiting_passenger_count[destination] += len(waiting_room_store.items)
        
        for destination, flight_request_store in self.flight_request_stores.items():
            waiting_passenger_count[destination] += len(flight_request_store.items)

        return waiting_passenger_count  
    
    def get_total_waiting_passenger_count(self):
        return sum(self.get_waiting_passenger_count().values())
    
    def get_waiting_passenger_count_for_destination(self, destination):
        return self.get_waiting_passenger_count()[destination]

    def check_charger_resource_type(self):
        # Based on the charger resource specification, this method changes the structure of the parking pads.
        if self.vertiport_data['shared_charger_sets'] and self.vertiport_data['num_chargers'] is None:
            raise ValueError("Both shared_charger_sets and num_chargers variables are None. Please input one.")
        elif self.vertiport_data['shared_charger_sets'] is not None:
            # If the user specified a shared chargers create a ParkingPadAndChargerResource object for each parking pad
            return self.add_parking_spaces_and_chargers_to_store(self.vertiport_data['shared_charger_sets'])
        elif self.vertiport_data['num_chargers'] is not None:
            # If the user didn't specify a shared charger resources then all of the chargers can be used by any of the
            # parking pads
            return self.add_parking_spaces_to_store()

    def create_charger_resources(self, chargers_set: Union[Dict, int]) -> Dict:
        charger_resources = defaultdict(dict)
        if isinstance(chargers_set, dict):
            for shared_charger_list in chargers_set.values():
                charger_resources[simpy.Resource(self.env, capacity=1)] = shared_charger_list
            return charger_resources

        # elif isinstance(chargers_set, int):
        #     for charger in range(chargers_set):
        #         charger_resources[simpy.Resource(self.env, capacity=1)] = charger
        #     return charger_resources
        else:
            raise TypeError("Charger set input type is not correct")

    def add_parking_spaces_and_chargers_to_store(self):
        # Add parking space IDs to parking_space_store
        if self.vertiport_layout.num_parking_pad == 0:
            #TODO: Currently, the simulator requires parking pad to store aircraft.
            # It should be able to store (park) the aircraft on the FATO if there are no parking spaces defined.
            print("Currently, skyport simulator is not able to run without a parking pad.")
            return
        # Create a store for parking spaces
        parking_space_store = simpy.FilterStore(self.env, capacity=self.vertiport_layout.num_parking_pad)
        # Create a dictionary for the shared charger combination.
        shared_charger_resources = self.create_charger_resources(self.self.vertiport_data['shared_charger_sets'])

        # If no parking pad is defined then use FATO as parking pad
        if self.vertiport_layout.num_parking_pad == 0:
            for fato in self.vertiport_layout.fato_ids:
                parking_pad = ParkingPadAndChargerResource(parking_space_id=fato)
                for charger_resource, parking_pad_id_list in shared_charger_resources.items():
                    if fato in parking_pad_id_list:
                        parking_pad.charger_resources.append(charger_resource)
                parking_space_store.put(parking_pad)
            return parking_space_store

        for parking_pad_id in self.vertiport_layout.parking_pad_ids:
            parking_pad = ParkingPadAndChargerResource(parking_space_id=parking_pad_id)

            for charger_resource, parking_pad_id_list in shared_charger_resources.items():
                if parking_pad_id in parking_pad_id_list:
                    parking_pad.charger_resources.append(charger_resource)
            parking_space_store.put(parking_pad)
        return parking_space_store

    def add_parking_spaces_to_store(self):
        """
        Creates parking_space_store. If there is no parking space is defined then uses FATO as parking space
        """
        if self.vertiport_layout.num_fato == 0 and self.vertiport_layout.num_parking_pad == 0:
            raise ValueError("There is no FATO and parking pad in the skyport layout.")

        # If there is no parking space is defined then use FATO as parking space
        if self.vertiport_layout.num_parking_pad == 0:
            parking_space_store = simpy.FilterStore(self.env, capacity=self.vertiport_layout.num_fato)
            for fato in self.vertiport_layout.fato_ids:
                parking_pad = ParkingPadAndChargerResource(parking_space_id=fato)
                parking_space_store.put(parking_pad)
            return parking_space_store
        # Add parking space IDs to parking_space_store
        # Create a store for parking spaces
        parking_space_store = simpy.FilterStore(self.env, capacity=self.vertiport_layout.num_parking_pad)
        for parking_pad_id in self.vertiport_layout.parking_pad_ids:
            parking_pad = ParkingPadAndChargerResource(parking_space_id=parking_pad_id)
            parking_space_store.put(parking_pad)
        return parking_space_store

    def add_fato_ids_to_store(self):
        if self.vertiport_layout.num_fato == 0:
            raise ValueError("There is no FATO in the skyport layout.")
        # Create a store for FATO pads
        fato_store = simpy.FilterStore(self.env, capacity=self.vertiport_layout.num_fato)
        # Add fato ids to fato_store
        for fato_id in self.vertiport_layout.fato_ids:
            # FATOs are PriorityResource to prioritize arrival or departure. Normally, the queueing rule in
            # this simulator is FIFO.
            fato_resourse = simpy.PriorityResource(self.env, 1)
            # Put the FATO resources to the fato store.
            fato_store.put(FatoResource(fato_id, fato_resourse))
        return fato_store

    def add_taxiway_nodes_to_store(self):
        # If the skyport layout has taxiway nodes create a store for them.
        if self.vertiport_layout.num_taxiway_nodes != 0:
            taxiway_node_store = simpy.FilterStore(self.env, capacity=self.vertiport_layout.num_taxiway_nodes)
            # Add node IDs to taxiwayRampNodeStore
            for node_id in self.vertiport_layout.taxiway_node_ids:
                taxiway_node_store.put(node_id)
            return taxiway_node_store

    def add_edges_to_store(self):
        # If the vertiport layout has edges create a store for them.
        if self.vertiport_layout.num_edges != 0:
            edge_store = simpy.FilterStore(self.env, capacity=self.vertiport_layout.num_edges)
            # Add edge IDs to edge store
            for edge_id in self.vertiport_layout.edge_ids:
                edge_store.put(set(edge_id))
            return edge_store

    def create_waiting_room_stores(self):
        waiting_room_stores = defaultdict(dict)
        # Create a store for waiting rooms
        exclude_origin = [dest for dest in self.vertiport_ids if dest != self.vertiport_id]
        for dest in exclude_origin:
            waiting_room_stores[dest] = simpy.FilterStore(self.env)
        return waiting_room_stores
    
    def create_flight_request_stores(self):
        flight_request_stores = defaultdict(dict)
        # Create a store for flight requests
        exclude_origin = [dest for dest in self.vertiport_ids if dest != self.vertiport_id]
        for dest in exclude_origin:
            flight_request_stores[dest] = simpy.FilterStore(self.env)
        return flight_request_stores


class TaxiOperationsConfig:
    # This configuration prevents the usage of taxi area while an aircraft takes off.
    is_simultaneous_taxi_and_take_off_allowed = True
    # is_simultaneous_taxi_and_landing_allowed = True


class FatoResource:
    """
    Creates an object to store a resourse with its ID. Simpy Resources doesn't
    have any identifying names/IDs. So, we need to create a new class to define IDs to resources.

    Attributes
    ----------
    fato_id : str
        ID of a FATO pad
    fato_resource : simpy Resource
        simpy resource for FATO pads
    """

    def __init__(self, fato_id, fato_resource):
        self.resource_id = fato_id
        self.fato_resource = fato_resource


class ParkingPadAndChargerResource:
    """
    Creates an object to store a parking pad with its ID and shared charger resource. Simpy Resources doesn't
    have any identifying names/IDs. So, we need to create a new class to define IDs to resources.

    Attributes
    ----------
    parking_space_id : str
        ID of a parking pad
    """

    def __init__(self, parking_space_id: str) -> str:
        self.resource_id = parking_space_id
        self.charger_resources = []
