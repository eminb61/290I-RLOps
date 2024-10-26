from .base_system_manager import BaseSystemManager
import simpy
import networkx as nx
from typing import Any, Dict, Generator, Optional
from ..utils.units import sec_to_ms, ms_to_sec, ms_to_min, miles_to_m, min_to_ms, sec_to_min, min_to_sec
from ..utils.distribution_generator import DistributionGenerator
from ..utils.weighted_random_chooser import random_choose_exclude_element
from ..utils.helpers import get_passenger_ids_from_passenger_list, miliseconds_to_hms, duplicate_str, \
    get_random_process_id, calculate_passenger_consolidation_time, check_whether_node_exists, careful_round, \
        flatten_dict, write_to_db, current_and_lookahead_pax_count
from ..aircraft.battery_model import BatteryModel
from ..aircraft.aircraft import Aircraft, AircraftStatus
from ..utils.get_state_variables import get_simulator_states
import pandas as pd
from enum import Enum
from collections import defaultdict
from ..utils.calc_required_charge_time_from_required_energy import calc_required_charge_time_from_required_energy
from typing import List, Union

class OnDemandSystemManager(BaseSystemManager):
    def __init__(self,
                 env: simpy.Environment,
                 vertiports: Dict,
                 vertiport_ids: List, 
                 vertiport_id_to_index_map: Dict,
                 vertiport_index_to_id_map: Dict,
                 num_initial_aircraft: int,                                 
                 scheduler: object,
                 wind: object,
                 airspace: object,
                 taxi_config: object,
                 sim_params: Dict,
                 external_optimization_params: Dict,
                 output_params: Dict,
                 aircraft_params: Dict,
                 vertiport_distances: pd.DataFrame,
                 passenger_distributions: Dict,                 
                 event_saver: object,
                 node_locations: Dict,
                 logger: object,
                 aircraft_logger: object,
                 passenger_logger: object,
                 sim_mode: Dict,
                 flight_directions_dict: Dict):

        super().__init__(env=env,
                        vertiports=vertiports,
                        vertiport_ids=vertiport_ids,
                        vertiport_id_to_index_map=vertiport_id_to_index_map,
                        vertiport_index_to_id_map=vertiport_index_to_id_map,
                        num_initial_aircraft=num_initial_aircraft,
                        scheduler=scheduler,
                        wind=wind,
                        airspace=airspace,
                        taxi_config=taxi_config,
                        sim_params=sim_params,
                        output_params=output_params,
                        aircraft_params=aircraft_params,
                        vertiport_distances=vertiport_distances,
                        passenger_distributions=passenger_distributions,
                        event_saver=event_saver,
                        node_locations=node_locations,
                        logger=logger,
                        aircraft_logger=aircraft_logger,
                        passenger_logger=passenger_logger,
                        sim_mode=sim_mode,
                        flight_directions_dict=flight_directions_dict
                        )
        self.aircraft_agents = {}
        self.passenger_agents = {}
        self.taxi_resource = simpy.Resource(self.env, 1)
        self.charging_time_distribution = self.build_charging_time_distribution()
        self.aircraft_battery_models = self.build_aircraft_battery_models()      
        # if (
        #     not self.sim_params['only_aircraft_simulation']
        #     and self.external_optimization_params['spill_optimization']
        # ):
        #     self.env.process(self.remove_spilled_passengers())
        
        self.trip_counter_tracker = 0
        self.total_demand = 0
        self.external_optimization_params = external_optimization_params
        self.spill_counter = {(origin, destination): 0 for origin in vertiport_ids for destination in vertiport_ids}
        self.last_repositioning_time = defaultdict(lambda: defaultdict(lambda: -float('inf')))
        self.repositioning_cooldown = 30 * 60  # 30 minutes in seconds
        self.base_repositioning_threshold = 0.3  # Increased from 0.2


    def build_charging_time_distribution(self) -> Optional[DistributionGenerator]:
        """
        Builds the charging time distribution with the given parameters
        """
        if self.aircraft_params['charging_time_dist'] is None:
            return None
        return DistributionGenerator(self.aircraft_params['charging_time_dist'])         
    
    def trigger_scheduler(self, 
                          origin_vertiport_id: str, 
                          destination_vertiport_id: int, 
                          passengers_waiting_room: simpy.Store) -> Generator:
        if departing_passengers := self.scheduler.check_waiting_room(current_waiting_room=passengers_waiting_room):
            # # Save passenger group completion time
            # for passenger in departing_passengers:
            #     self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='passenger_consolidation')

            # # Save passenger consolidation time
            # self.event_saver.save_passenger_group_consolidation_time(
            #     vertiport_id=origin_vertiport_id,
            #     passenger_ids=get_passenger_ids_from_passenger_list(departing_passengers),
            #     wr_number=destination_vertiport_id,
            #     consolidation_time=calculate_passenger_consolidation_time(departing_passengers)
            # )

            # self.logger.debug(f"Passengers {[passenger.passenger_id for passenger in departing_passengers]} are ready to depart from {origin_vertiport_id} to {destination_vertiport_id}")

            # Put passengers into the flight request queue
            for passenger in departing_passengers:
                self.vertiports[origin_vertiport_id].flight_request_stores[destination_vertiport_id].put(passenger)
                passenger.flight_queue_store = self.vertiports[origin_vertiport_id].flight_request_stores[destination_vertiport_id]

            # Increase passenger queue count
            self.event_saver.update_passenger_departure_queue_counter(vertiport_id=origin_vertiport_id, queue_update=1)

            yield self.env.process(self.reserve_aircraft(
                origin_vertiport_id=origin_vertiport_id,
                destination_vertiport_id=destination_vertiport_id,
                departing_passengers=departing_passengers)
                )
        
        if self.sim_params['fleet_rebalancing'] and not self.external_optimization_params['periodic_time_step']:
            num_available_aircraft_at_origin = self.check_num_available_aircraft(origin_vertiport_id)
            num_available_aircraft_at_dest = self.check_num_available_aircraft(destination_vertiport_id)
            num_pax_groups = int(self.get_num_waiting_passengers(origin_vertiport_id) / self.aircraft_params['pax'])
            required_aircraft = max(num_pax_groups - num_available_aircraft_at_origin, 0)
            self.logger.debug(f"# of available aircraft at {origin_vertiport_id}: {num_available_aircraft_at_origin}/{self.vertiports[origin_vertiport_id].num_aircraft}, # of available aircraft at {destination_vertiport_id}: {num_available_aircraft_at_dest}/{self.vertiports[destination_vertiport_id].num_aircraft}")
            self.logger.debug(f"Required aircraft for rebalancing: {required_aircraft}. # of pax groups: {num_pax_groups}")
            if required_aircraft > 0 and num_available_aircraft_at_dest > 0:
                self.event_saver.update_repositioning_counter(vertiport_id=destination_vertiport_id,
                                                repositioning_count=required_aircraft)
                self.logger.debug(f"Creating {required_aircraft} empty flights to fulfill the demand at {origin_vertiport_id}")
                for _ in range(required_aircraft):
                    self.env.process(self.reserve_aircraft(origin_vertiport_id=destination_vertiport_id,
                                                            destination_vertiport_id=origin_vertiport_id,
                                                            departing_passengers=[]))                

    def get_num_waiting_passengers(self, origin_vertiport_id: str):
        """
        Checks the demand for the given vertiport
        """
        return self.vertiports[origin_vertiport_id].get_passenger_count()

    def reserve_aircraft(self, 
                         origin_vertiport_id: Any, 
                         destination_vertiport_id: int, 
                         departing_passengers: list) -> Generator:        
        # Start tracking aircraft allocation time
        aircraft_allocation_start_time = self.env.now
        # Update the vertiport flight request state
        self.vertiports[origin_vertiport_id].num_flight_requests += 1           
        # Log the aircraft allocation process
        flight_id = get_random_process_id()
        self.logger.debug(f'|{flight_id}| Started: Aircraft allocation process at vertiport {origin_vertiport_id} for {destination_vertiport_id}.')
        # Retrieve aircraft from the store
        # yield self.env.timeout(1)
        aircraft = yield self.retrieve_aircraft(origin_vertiport_id) 
        
        # Set the status of the aircraft
        aircraft.status = AircraftStatus.FLY
        # Update the vertiport flight request state
        self.vertiports[origin_vertiport_id].num_flight_requests -= 1            
        # Set the real pushback time of the aircraft 
        aircraft.pushback_time = aircraft_allocation_start_time
        # Set the flight direction of the aircraft
        aircraft.flight_direction = f'{origin_vertiport_id}_{destination_vertiport_id}'
        # Set the flight ID of the aircraft
        aircraft.flight_id = flight_id
        # Measure and save the aircraft allocation time
        self.save_aircraft_allocation_time(origin_vertiport_id, aircraft, aircraft_allocation_start_time)

        # Remove passengers from the flight queue
        # self.scheduler.pop_passengers_from_flight_queue_by_id(passengers=departing_passengers)

        # Check the waiting room for additional passengers if there is still space on the aircraft
        departing_passengers = self.add_additional_passengers_if_needed(origin_vertiport_id, destination_vertiport_id, departing_passengers)

        self.logger.debug(f'|{flight_id}| Completed: Aircraft allocation process at vertiport {origin_vertiport_id} for {destination_vertiport_id}. Aircraft tail number: {aircraft.tail_number}, SOC: {careful_round(aircraft.soc, 2)}, Departing passengers: {len(departing_passengers)}')

        # Save aircraft allocation and idle times
        self.save_aircraft_times(origin_vertiport_id, aircraft)

        # Assign passengers to the aircraft
        aircraft.passengers_onboard = departing_passengers

        # Save total passenger waiting time and flight assignment time
        self.save_passenger_times(origin_vertiport_id=origin_vertiport_id, passengers=aircraft.passengers_onboard) 
        self.passenger_logger.info(f"Passenger waiting times for flight from {origin_vertiport_id} to {destination_vertiport_id}: Passenger ids : {[p.passenger_id for p in departing_passengers]} waiting times: {[miliseconds_to_hms(self.env.now - passenger.waiting_room_arrival_time) for passenger in departing_passengers]}")

        self.calculate_flight_reward(origin_vertiport_id, destination_vertiport_id, len(departing_passengers))

        # Start the passenger departure process
        if departing_passengers:
            # self.logger.debug(f'|{flight_id}| Started: Passengers exited the waiting room at vertiport {origin_vertiport_id} for {destination_vertiport_id} flight.')
            self.env.process(self.initiate_passenger_departure(departing_passengers=departing_passengers))
            # self.logger.debug(f'|{flight_id}| Completed: Passengers arrived at the boarding gate at vertiport {origin_vertiport_id} for {destination_vertiport_id} flight.')

        # Start the aircraft departure process
        yield self.env.process(
            self.simulate_aircraft_departure_process(aircraft=aircraft,
                                                    origin_vertiport_id=origin_vertiport_id, 
                                                    destination_vertiport_id=destination_vertiport_id)
        )        

    def calculate_flight_reward(self, origin_vertiport_id: str, destination_vertiport_id: str, num_passengers: int):
        # Get the distance between the origin and destination vertiports
        distance = self.get_mission_length(origin_vertiport_id, destination_vertiport_id)
        # Calculate the reward for the flight
        usd_per_mile = self.external_optimization_params['reward_function_parameters']['usd_pax_mile']
        base_fare = self.external_optimization_params['reward_function_parameters']['usd_base_fare']
        usd_casm = self.external_optimization_params['reward_function_parameters']['usd_casm']
        num_seats = self.aircraft_params['pax']
        reward = (base_fare + usd_per_mile * distance) * num_passengers - (usd_casm * distance * num_seats)
        self.event_saver.ondemand_reward_tracker['total_reward'] += reward

    def check_passenger_max_waiting_time_threshold(self):
        if departing_passenger_groups := self.scheduler.check_max_waiting_time_threshold(
            sec_to_ms(self.sim_params['max_passenger_waiting_time'])):
            for origin in departing_passenger_groups:
                for destination in departing_passenger_groups[origin]:
                    departing_passengers = departing_passenger_groups[origin][destination]
                    for passenger in departing_passengers:
                        self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='passenger_waiting_room_departure') 
                        # Put passengers into the flight request queue
                        self.vertiports[origin].flight_request_stores[destination].put(passenger)
                        passenger.flight_queue_store = self.vertiports[origin].flight_request_stores[destination]  
                    # Increase passenger queue count
                    self.event_saver.update_passenger_departure_queue_counter(vertiport_id=origin, 
                                                                            queue_update=len(departing_passenger_groups))   
                    flight_queue = [self.env.process(
                        self.reserve_aircraft(origin_vertiport_id=origin, 
                                            destination_vertiport_id=departing_passengers[0].destination_vertiport_id,
                                            departing_passengers=departing_passengers))
                    ]

                yield self.env.all_of(flight_queue) 

    
    def remove_spilled_passengers(self):
        # TODO: You just need to do this check after the decision making time step if you remove the pax that will spill until the next decision making time step!

        max_waiting_time = sec_to_ms(self.sim_params['max_passenger_waiting_time'] - min_to_sec(1)) # Subtract 1 min to account for exceeding the max waiting time by 1 min
        while True:
            yield self.env.timeout(min_to_ms(1))  # Check every minute
            # If the time is the multiple of the decision making time step, skip the spillover process
            if self.env.now % sec_to_ms(self.external_optimization_params['periodic_time_step']) == 0:
                self.logger.debug(f"Skipping spillover process at time {miliseconds_to_hms(self.env.now)}")
                continue
        
            self.logger.debug(f"Checking for spilled passengers at time {miliseconds_to_hms(self.env.now)}")

            for origin in self.vertiport_ids:
                for destination in self.vertiport_ids:
                    if origin != destination:
                        # Get the passengers that have been waiting for more than the maximum waiting time
                        pax_in_wr = self.get_pax_for_od_pair(origin, destination)
                        passengers_exceeding_wait = [p for p in pax_in_wr if self.env.now - p.waiting_room_arrival_time > max_waiting_time]
                        for passenger in passengers_exceeding_wait:
                            self.spill_passenger(passenger, origin, destination)

    def make_ondemand_decision(self):
        """
        Make an on-demand decision based on the current state of the system.
        """
        self.logger.debug(f"Making on-demand decision at time {miliseconds_to_hms(self.env.now)}")
        max_waiting_time = sec_to_ms(self.sim_params['max_passenger_waiting_time'])
        decision_making_time_step = sec_to_ms(self.external_optimization_params['periodic_time_step'])
        
        # Get available aircraft count at each vertiport
        available_aircraft = {vertiport_id: self.check_num_available_aircraft(vertiport_id) 
                              for vertiport_id in self.vertiport_ids}
        
        # Check each vertiport for passengers exceeding max waiting time
        for origin in self.vertiport_ids:
            for destination in self.vertiport_ids:
                if origin != destination:
                    # Get the total number of passengers in the waiting room
                    pax_in_wr = self.get_pax_for_od_pair(origin, destination)

                    # Check if any passenger has exceeded max waiting time
                    passengers_exceeding_wait = [p for p in pax_in_wr if self.env.now - p.waiting_room_arrival_time > max_waiting_time]
                    # Add the passengers that are going to exceed the max waiting time by the next decision making time step. 
                    pax_will_exceed_waiting_time_threshold = [p for p in pax_in_wr if self.env.now - p.waiting_room_arrival_time \
                                                                + decision_making_time_step > max_waiting_time and p not in passengers_exceeding_wait]
                    # Merge the two lists
                    passengers_exceeding_wait += pax_will_exceed_waiting_time_threshold
                                        
                    # If there are available aircraft, create flights for passengers exceeding max waiting time
                    while pax_in_wr and available_aircraft[origin] > 0:
                        # Check if there are any waiting rooms with more than equal to aircraft seat capacity number of passengers or any passengers exceeding max waiting time
                        if passengers_exceeding_wait or len(pax_in_wr) >= self.aircraft_params['pax']:
                            # Create a flight with up to seat_capacity passengers
                            passengers_to_fly = pax_in_wr[:min(len(pax_in_wr), self.aircraft_params['pax'])]
                            self.env.process(self.create_flight(origin, destination, passengers_to_fly))
                            available_aircraft[origin] -= 1
                            
                            # Remove these passengers from the total_passengers list
                            pax_in_wr = pax_in_wr[len(passengers_to_fly):]
                        else:
                            # If no passengers exceeded wait time, we're done
                            break

                    # If there are still passengers exceeding wait time but no available aircraft, spill them
                    remaining_exceeded_wait = [p for p in pax_in_wr if self.env.now - p.waiting_room_arrival_time > max_waiting_time]
                    pax_will_exceed_waiting_time_threshold = [p for p in pax_in_wr if self.env.now - p.waiting_room_arrival_time \
                                                                + decision_making_time_step > max_waiting_time and p not in remaining_exceeded_wait]
                    remaining_exceeded_wait += pax_will_exceed_waiting_time_threshold
                    for passenger in remaining_exceeded_wait:
                        self.spill_passenger(passenger, origin, destination)

        self.env.process(self.consider_repositioning_flights())
        yield self.env.timeout(0)

    def spill_passenger(self, passenger, origin, destination):
        """
        Handle the spillover of a passenger.
        """
        self.event_saver.update_spilled_passenger_counter(flight_dir=f"{origin}_{destination}")
        self.logger.info(f'Passenger {passenger.passenger_id} spilled at {origin} after waiting {ms_to_min(self.env.now - passenger.waiting_room_arrival_time)} min.')
        self.spill_counter[(origin, destination)] += 1
        self.vertiports[origin].spilled_passengers_dict[destination] = self.spill_counter[(origin, destination)] 
        self.remove_pax_from_waiting_room(origin, destination, passenger, for_flight=False)

    def create_flight(self, origin, destination, passengers):
        """
        Create a flight for the given passengers.
        """
        for passenger in passengers:
            self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='passenger_departure')
            # Remove passengers from the waiting room
            self.remove_pax_from_waiting_room(origin, destination, passenger)
        
        # Create flight process
        yield self.env.process(self.reserve_aircraft(origin_vertiport_id=origin,
                                               destination_vertiport_id=destination,
                                               departing_passengers=passengers))       

    def remove_pax_from_waiting_room(self, origin, destination, passenger, for_flight=True) -> None:
        """
        Remove passengers from the waiting room.
        """
        waiting_room = self.vertiports[origin].waiting_room_stores[destination]
        if not for_flight:
            self.logger.info(f'Will remove passenger from waiting room at {origin}. Pax waiting times: {[{p.passenger_id: ms_to_min(self.env.now - p.waiting_room_arrival_time)} for p in waiting_room.items]}')
        waiting_room.get(lambda x: x.passenger_id == passenger.passenger_id)   
        # waiting_room.items.remove(passenger)   
        if not for_flight:  
            self.logger.info(f'Passenger {passenger.passenger_id} removed from the waiting room at {origin}. Waiting passengers: {[p.passenger_id for p in waiting_room.items]}')

    def get_pax_for_od_pair(self, origin_vertiport_id: str, destination_vertiport_id: str) -> list:
        """
        Get the passengers waiting for a flight from the origin to the destination
        """
        return self.vertiports[origin_vertiport_id].waiting_room_stores[destination_vertiport_id].items
    
    def get_pax_count_for_od_pair(self, origin_vertiport_id: str, destination_vertiport_id: str) -> int:
        """
        Get the number of passengers waiting for a flight from the origin to the destination
        """
        return len(self.vertiports[origin_vertiport_id].waiting_room_stores[destination_vertiport_id].items)
    
    def consider_repositioning_flights(self):
        """
        Decides whether to create repositioning flights based on current and future demand and aircraft distribution.
        """
        REPOSITIONING_THRESHOLD = 0.1  # Adjust this value as needed

        total_aircraft_count = sum(self.check_num_available_aircraft(v_id) for v_id in self.vertiport_ids)
        total_demand = sum(self.get_adjusted_demand(v_id) for v_id in self.vertiport_ids)

        # Calculate ideal aircraft distribution based on current and future demand
        ideal_distribution = {v_id: self.get_adjusted_demand(v_id) / max(1, total_demand) for v_id in self.vertiport_ids}

        for origin in self.vertiport_ids:
            available_aircraft = self.check_num_available_aircraft(origin)
            if available_aircraft == 0:
                continue

            best_destination = None
            best_score = -float('inf')

            for destination in self.vertiport_ids:
                if origin == destination:
                    continue

                score = self.calculate_repositioning_score(origin, destination, ideal_distribution, total_aircraft_count)
                if score > best_score:
                    best_score = score
                    best_destination = destination

            if best_destination and best_score > REPOSITIONING_THRESHOLD:
                self.env.process(self.create_repositioning_flight(origin, best_destination))

        yield self.env.timeout(0)

    def calculate_repositioning_score(self, origin, destination, ideal_distribution, total_aircraft_count):
        adjusted_demand = self.get_adjusted_demand(destination)
        
        if adjusted_demand == 0:
            return -1.0  # No need for repositioning if there's no adjusted demand
        
        current_aircraft_count = self.check_num_available_aircraft(destination)
        flying_to_destination = self.count_aircraft_flying_to_destination(destination)
        
        total_incoming_aircraft = current_aircraft_count + flying_to_destination
        ideal_aircraft_count = ideal_distribution[destination] * total_aircraft_count

        demand_satisfaction_rate = min(1, total_incoming_aircraft / max(1, adjusted_demand / self.aircraft_params['pax']))
        network_balance_score = (ideal_aircraft_count - total_incoming_aircraft) / max(1, ideal_aircraft_count)

        current_demand = self.get_total_demand_for_vertiport(destination)
        future_demand = current_and_lookahead_pax_count(self.vertiports[destination])
        demand_growth_rate = (future_demand - current_demand) / max(1, current_demand)

        demand_weight = 0.5 + (0.1 * demand_growth_rate)
        balance_weight = 1 - demand_weight

        score = (1 - demand_satisfaction_rate) * demand_weight + network_balance_score * balance_weight

        if abs(network_balance_score) < 0.15 and abs(1 - demand_satisfaction_rate) < 0.15:
            return -1  # Demand is perfectly balanced

        origin_surplus = self.check_num_available_aircraft(origin) - (ideal_distribution[origin] * total_aircraft_count)
        if origin_surplus <= 0:
            score *= 0.5  # Reduce score if origin doesn't have surplus aircraft

        # repositioning_distance = self.get_mission_length(origin, destination)
        # cost_factor = 1 / (1 + repositioning_distance * 0.0001)  # Adjust the 0.0001 as needed
        # score *= cost_factor

        return score

    def calculate_system_utilization(self):
        total_aircraft = sum(len(self.aircraft_agents) for _ in self.vertiport_ids)
        flying_aircraft = sum(self.count_aircraft_flying_to_destination(v_id) for v_id in self.vertiport_ids)
        return flying_aircraft / max(1, total_aircraft)
    
    def calculate_dynamic_threshold(self, system_utilization):
        # Increase threshold as system utilization increases
        return self.base_repositioning_threshold + (system_utilization * 0.2)


    # def consider_repositioning_flights(self):
    #     """
    #     Decides whether to create repositioning flights based on current and future demand and aircraft distribution.
    #     """
    #     REPOSITIONING_THRESHOLD = 0.2  # Adjust this value as needed

    #     total_aircraft_count = sum(self.check_num_available_aircraft(v_id) for v_id in self.vertiport_ids)
    #     total_demand = sum(self.get_adjusted_future_demand(v_id) for v_id in self.vertiport_ids)

    #     # Calculate ideal aircraft distribution based on current and future demand
    #     ideal_distribution = {v_id: self.get_adjusted_future_demand(v_id) / max(1, total_demand) for v_id in self.vertiport_ids}

    #     for origin in self.vertiport_ids:

    #         available_aircraft = self.check_num_available_aircraft(origin)
    #         if available_aircraft == 0:
    #             continue

    #         # If the supply is less than demand, don't reposition aircraft from this vertiport
    #         if self.supply_demand_difference(origin) < 0:
    #             continue

    #         for destination in self.vertiport_ids:
    #             if origin == destination:
    #                 continue

    #             best_destination = None
    #             best_score = -float('inf')

    #             score = self.calculate_repositioning_score(origin, destination, ideal_distribution, total_aircraft_count)
    #             if score > best_score:
    #                 best_score = score
    #                 best_destination = destination

    #         if best_destination and best_score > REPOSITIONING_THRESHOLD:
    #             # Calculate the optimal number of aircraft to reposition
    #             current_aircraft_count = self.check_num_available_aircraft(best_destination)
    #             flying_to_destination = self.count_aircraft_flying_to_destination(best_destination)
    #             ideal_aircraft_count = ideal_distribution[best_destination] * total_aircraft_count
                
    #             num_to_reposition = min(
    #                 available_aircraft,
    #                 max(0, int(ideal_aircraft_count - (current_aircraft_count + flying_to_destination)))
    #             )
                
    #             # Create repositioning flights for the calculated number of aircraft
    #             for _ in range(num_to_reposition):
    #                 self.env.process(self.create_repositioning_flight(origin, best_destination))


    #     yield self.env.timeout(0)

    # def calculate_repositioning_score(self, origin, destination, ideal_distribution, total_aircraft_count):
    #     """
    #     Calculate a refined score for repositioning an aircraft from origin to destination.
    #     """
    #     # Calculate the supply-demand difference at the destination
    #     destination_difference = self.supply_demand_difference_for_future(destination)
        
    #     if destination_difference <= 0:
    #         return -1.0  # No need for repositioning if destination has enough or too many aircraft
        
    #     # Calculate the ideal number of aircraft for the destination
    #     ideal_aircraft_count = ideal_distribution[destination] * total_aircraft_count
        
    #     # Calculate current aircraft at destination (including those flying there)
    #     current_aircraft_count = self.check_num_available_aircraft(destination)
    #     flying_to_destination = self.count_aircraft_flying_to_destination(destination)
    #     total_destination_aircraft = current_aircraft_count + flying_to_destination
        
    #     # Calculate network imbalance
    #     network_imbalance = (ideal_aircraft_count - total_destination_aircraft) * self.aircraft_params['pax']
        
    #     # Calculate origin's capacity to provide aircraft
    #     origin_surplus = self.supply_demand_difference_for_future(origin)
    #     if origin_surplus >= 0:
    #         origin_capacity = min(origin_surplus, self.check_num_available_aircraft(origin) * self.aircraft_params['pax'])
    #     else:
    #         return -1.0  # Origin can't provide aircraft if it has a deficit
        
    #     # Calculate the score based on destination need and network balance
    #     score = min(destination_difference, network_imbalance, origin_capacity)
        
    #     # Normalize the score
    #     normalized_score = score / self.aircraft_params['pax']
        
    #     return normalized_score

    def get_adjusted_demand(self, vertiport_id: str) -> int:
        return max(0, self.supply_demand_difference(vertiport_id))
    
    def supply_demand_difference(self, vertiport_id: str) -> int:
        current_and_future_demand = current_and_lookahead_pax_count(self.vertiports[vertiport_id])
        flying_to_vertiport = self.count_aircraft_flying_to_destination(vertiport_id)
        num_available_aircraft = self.check_num_available_aircraft(vertiport_id)
        return current_and_future_demand - (num_available_aircraft + flying_to_vertiport) * self.aircraft_params['pax']
    
    def get_adjusted_future_demand(self, vertiport_id: str) -> int:
        return max(0, self.supply_demand_difference_for_future(vertiport_id))
    
    def supply_demand_difference_for_future(self, vertiport_id: str) -> int:
        future_demand = self.vertiports[vertiport_id].get_total_expected_pax_arrival_count()
        flying_to_vertiport = self.count_aircraft_flying_to_destination(vertiport_id)
        num_available_aircraft = self.check_num_available_aircraft(vertiport_id)
        return future_demand - (num_available_aircraft + flying_to_vertiport) * self.aircraft_params['pax']
    
    def count_aircraft_flying_to_destination(self, dest_vertiport_id: str) -> int:
        """
        Count the number of aircraft flying to the destination vertiport.
        """
        return sum(
            1 for aircraft in self.aircraft_agents.values()
            if aircraft.status == AircraftStatus.FLY and
            aircraft.flight_direction.split('_')[1] == dest_vertiport_id
        )

    def get_total_demand_for_vertiport(self, vertiport_id):
        """
        Get the total number of waiting passengers for all destinations from the given vertiport.
        """
        return sum(self.get_pax_count_for_od_pair(vertiport_id, dest) 
                   for dest in self.vertiport_ids if dest != vertiport_id)

    def create_repositioning_flight(self, origin, destination):
        """
        Create a repositioning flight from origin to destination.
        """
        self.logger.info(f"Creating repositioning flight from {origin} to {destination}")
        self.event_saver.update_repositioning_counter(vertiport_id=destination,
                                                        repositioning_count=1)        
        yield self.env.process(self.reserve_aircraft(
            origin_vertiport_id=origin,
            destination_vertiport_id=destination,
            departing_passengers=[]
        ))


    def put_passenger_into_waiting_room(self, passenger: object) -> None:
        """
        Puts the passenger into the waiting room based on their flight destination and triggers scheduler
        :param passenger:
        :return:
        """
        # Define the location of the passenger as the waiting room at their current location
        passenger.location = f'{passenger.origin_vertiport_id}_ROOM'
        # Save the current state of the passenger
        self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='enter_waiting_room')
        # Record the current time as the passenger's waiting room arrival time
        passenger.waiting_room_arrival_time = self.env.now
        # Get the waiting room that corresponds to the passenger's destination
        passengers_waiting_room = self.vertiports[passenger.origin_vertiport_id].waiting_room_stores[passenger.destination_vertiport_id]
        # Put the passenger into the waiting room and keep a reference to it
        passengers_waiting_room.put(passenger)
        passenger.waiting_room_store = passengers_waiting_room
        # Log the passenger's arrival
        self.logger.debug(f'Passenger {passenger.passenger_id} entered the waiting room at {passenger.origin_vertiport_id}. Currently waiting passengers: {[p.passenger_id for p in passengers_waiting_room.items]}.')

        if self.sim_params['training_data_collection']:
            self.save_state_variables_for_training(passenger_id=passenger.passenger_id)

        # print(f"Passenger {passenger.passenger_id} entered the waiting room at {passenger.origin_vertiport_id}. Waiting passengers: {[p.passenger_id for p in passengers_waiting_room.items]}. Flight queue: {[p.passenger_id for p in self.vertiports[passenger.origin_vertiport_id].flight_request_stores[passenger.destination_vertiport_id].items]}")
        # Trigger scheduler
        if self.external_optimization_params['periodic_time_step']:
            yield self.env.timeout(0)
        else:
            yield self.env.process(self.trigger_scheduler(origin_vertiport_id=passenger.origin_vertiport_id,
                                                        destination_vertiport_id=passenger.destination_vertiport_id, 
                                                        passengers_waiting_room=passengers_waiting_room))  

    def save_state_variables_for_training(self, passenger_id):
        states = get_simulator_states(vertiports=self.vertiports,
                                      aircraft_agents=self.aircraft_agents,
                                      num_initial_aircraft=self.num_initial_aircraft,
                                      simulation_states=self.sim_params['simulation_states'])
        states['passenger_id'] = passenger_id
        states['sim_id'] = self.sim_params['sim_id']
        states = flatten_dict(states)
        write_to_db(db_path=self.output_params['state_trajectory_db_path'], 
                    table_name=self.output_params['state_trajectory_db_tablename'],
                    dic=states)
        
    def check_holding_aircraft(self, flight_direction: str, origin_vertiport_id: Any, destination_vertiport_id: Any):
        arrival_fix_resource = self.get_second_to_last_airlink_resource(flight_direction=flight_direction).airnode_resource
        if len(arrival_fix_resource.queue) > 0:
            num_queued = len(arrival_fix_resource.queue)
            num_available_aircraft_at_dest = self.check_num_available_aircraft(destination_vertiport_id)
            num_available_aircraft_at_origin = self.check_num_available_aircraft(origin_vertiport_id)
            # additional_repositioning = 0
            # if (num_available_aircraft_at_origin+1) * 2 < num_available_aircraft_at_dest:
            #     additional_repositioning += 1
            num_scheduled_for_repositioning = min(num_queued, num_available_aircraft_at_dest) 
            self.event_saver.update_repositioning_counter(vertiport_id=destination_vertiport_id,
                                                          repositioning_count=num_scheduled_for_repositioning)
            # num_scheduled_for_repositioning += additional_repositioning
            self.logger.debug(f'Number of holding aircraft queued at {destination_vertiport_id}: {num_queued}. Number of available aircraft at {destination_vertiport_id}: {num_available_aircraft_at_dest}')
            for _ in range(num_scheduled_for_repositioning):
                self.env.process(self.assign_empty_flight(vertiport_id=destination_vertiport_id))

    def assign_empty_flight(self, vertiport_id: Any):
        # TODO: Check the waiting rooms and the parking pad availability of the other vertiports and 
        # pick the one with the highest number of passengers
        destination_id = random_choose_exclude_element(elements_list=self.vertiport_ids, 
                                                       exclude_element=vertiport_id, 
                                                       num_selection=1)[0]
        num_available_aircraft_at_origin = self.check_num_available_aircraft(vertiport_id)
        num_available_aircraft_at_destination = self.check_num_available_aircraft(destination_id)

        self.logger.debug(f'Assigning empty flight from {vertiport_id} to {destination_id}.'
                         f' Number of available aircraft at {vertiport_id}: {num_available_aircraft_at_origin}. Number of available aircraft at {destination_id}: {num_available_aircraft_at_destination}.')
        yield self.env.process(self.reserve_aircraft(origin_vertiport_id=vertiport_id,
                                                destination_vertiport_id=destination_id,
                                                departing_passengers=[]))        
        

    def simulate_terminal_airspace_arrival_process(self, aircraft: object, arriving_passengers: list):
        holding_start = self.env.now

        aircraft.status = AircraftStatus.HOLD
        
        # Increase aircraft arrival queue counter
        self.event_saver.update_aircraft_arrival_queue_counter(vertiport_id=aircraft.destination_vertiport_id,
                                                               queue_update=1)
        
        # Request arrival fix resource
        arrival_fix_usage_request, arrival_fix_resource = self.request_fix_resource(flight_direction=aircraft.flight_direction, operation_type='arrival')
        self.logger.debug(f'Aircraft {aircraft.tail_number} requesting arrival fix resource at {aircraft.destination_vertiport_id}.'
                         f' Number of holding aircraft queued at {aircraft.destination_vertiport_id}: {len(arrival_fix_resource.queue)}'
                         f' Number of available aircraft at {aircraft.destination_vertiport_id}: {self.check_num_available_aircraft(aircraft.destination_vertiport_id)}')
        if self.sim_params['fleet_rebalancing'] and not self.external_optimization_params['periodic_time_step']:
            self.check_holding_aircraft(flight_direction=aircraft.flight_direction,
                                        origin_vertiport_id=aircraft.origin_vertiport_id,
                                        destination_vertiport_id=aircraft.destination_vertiport_id)
            
        # Start the SOC update process
        self.env.process(self.update_soc_during_holding(aircraft))

        yield arrival_fix_usage_request
        self.logger.debug(f'Aircraft {aircraft.tail_number} has been assigned to the arrival fix resource at {aircraft.destination_vertiport_id}')

        aircraft.status = AircraftStatus.FLY

        holding_end = self.env.now
        aircraft.holding_time = holding_end - holding_start
        # aircraft.update_holding_energy_consumption(aircraft.holding_time)
        aircraft.save_process_time(event='holding', process_time=aircraft.holding_time)

        self.event_saver.save_aircraft_holding_time(vertiport_id=aircraft.destination_vertiport_id,
                                                    waiting_time=aircraft.holding_time)        

        aircraft.arrival_fix_resource = arrival_fix_resource
        aircraft.arrival_fix_usage_request = arrival_fix_usage_request  

        yield self.env.process(self.fato_and_parking_pad_usage_process(aircraft=aircraft))       

    def update_soc_during_holding(self, aircraft: object):
        """Regularly update the state of charge (SoC) of the aircraft while holding."""
        while aircraft.status == AircraftStatus.HOLD:
            yield self.env.timeout(sec_to_ms(aircraft.aircraft_params['soc_update_interval']))  # Wait for X seconds
            if aircraft.status == AircraftStatus.HOLD:
                aircraft.update_holding_energy_consumption(sec_to_ms(aircraft.aircraft_params['soc_update_interval']))  # Update SoC for X seconds of holding
                self.logger.debug(f'Aircraft {aircraft.tail_number} updated SoC during holding at {aircraft.destination_vertiport_id}. Current SoC: {careful_round(aircraft.soc, 2)}')

    def fato_and_parking_pad_usage_process(self, aircraft: object):
        if aircraft.parking_space_id is None and \
            aircraft.assigned_fato_id is None and \
                self.vertiports[aircraft.destination_vertiport_id].vertiport_layout.num_parking_pad > 0:
            # Get Parking pad
            yield self.env.process(self.parking_pad_request(aircraft=aircraft))
            # Save flight_direction
            flight_direction = aircraft.flight_direction
            # Landing process
            yield self.env.process(self.simulate_landing_process(aircraft=aircraft))
            # Taxi
            yield self.env.process(self.simulate_taxi_process(aircraft=aircraft, operation_type='arrival'))

            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()
            # Put aircraft into available aircraft store
            # self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
            #                                                 aircraft=aircraft)    

            # print(f"Aircraft {aircraft.tail_number} parked with passengers: {[p.passenger_id for p in aircraft.passengers_onboard]} to {aircraft.destination_vertiport_id} at time : {miliseconds_to_hms(self.env.now)}. Available aircraft at that location: {[a.tail_number for a in self.vertiports[aircraft.destination_vertiport_id].available_aircraft_store.items]}")

            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)
            self.update_flying_aircraft_count(update=-1)
            self.save_passenger_trip_times(aircraft=aircraft, flight_direction=flight_direction)

            self.trip_counter_tracker += len(aircraft.passengers_onboard)

            if self.sim_mode['optim_rl_comparison'] and self.external_optimization_params['flight_duration_constant']:
                # Check optimizer and simulation flight time compatibility
                if aircraft.flight_duration and self.env.now - aircraft.pushback_time < \
                    sec_to_ms(self.external_optimization_params['flight_duration_constant']):
                    ground_holding_time = sec_to_ms(self.external_optimization_params['flight_duration_constant']) \
                        - (self.env.now - aircraft.pushback_time) - 1
                    self.logger.debug(f"|{duplicate_str(aircraft.tail_number)}| Flight duration is less than {sec_to_min(self.external_optimization_params['flight_duration_constant'])} minutes."
                                        f' Pushback time was {miliseconds_to_hms(aircraft.pushback_time)}. Aircraft {aircraft.tail_number}'
                                        f' will hold for {ms_to_min(ground_holding_time)} ({miliseconds_to_hms(ground_holding_time)}) mins at {aircraft.location}')
                    yield self.env.timeout(ground_holding_time)
                else:
                    ground_holding_time = 0            

            yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))
       
        # Only FATO case and starting point is not a FATO
        elif aircraft.parking_space_id is None and aircraft.assigned_fato_id is None:
            # NOTE: We don't request FATO here because it will be requested in the landing_process. However
            # in the case of single FATO, we are increasing the FATO reservation time from 
            # time_descend_transition + time_hover_descend to time_descend_transition + time_hover_descend + time_descend
            # yield self.env.process(self.fato_pad_request(aircraft=aircraft, operation_type='arrival'))  

            # Landing process
            yield self.env.process(self.simulate_landing_process(aircraft=aircraft)) 
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)
            self.update_flying_aircraft_count(update=-1)   
            self.save_passenger_trip_times(aircraft=aircraft)
                   
            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()
            # Charging process
            yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))         

        # If the aircraft is not assigned to a parking pad but its starting location is a FATO
        elif aircraft.parking_space_id is None and \
            aircraft.assigned_fato_id is not None and \
                self.vertiports[aircraft.destination_vertiport_id].vertiport_layout.num_parking_pad > 0:
            # Get Parking pad
            yield self.env.process(self.parking_pad_request(aircraft=aircraft))
            # Get FATO
            yield self.env.process(self.fato_pad_request(aircraft=aircraft, operation_type='arrival'))
            # Taxi
            yield self.env.process(self.simulate_taxi_process(aircraft=aircraft, operation_type='arrival'))

            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)

            self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}. Config: Parking pad: {aircraft.parking_space_id}, FATO: {aircraft.assigned_fato_id}')

            # Charging process
            yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))
 
        
        # Only FATO case and the starting location is a FATO
        elif aircraft.assigned_fato_id is not None:
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)

            self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}. Config: Parking pad: {aircraft.parking_space_id}, FATO: {aircraft.assigned_fato_id}')

            # Get FATO
            yield self.env.process(self.fato_pad_request(aircraft=aircraft, operation_type='arrival', fato_id=aircraft.assigned_fato_id))

            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()

            if aircraft.initial_process in ['charging', None]:
                # Charging process
                yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))
                aircraft.initial_process = None
            elif aircraft.initial_process == 'parking':
                # Put aircraft into available aircraft store
                self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                                aircraft=aircraft)
                aircraft.initial_process = None
            else:
                raise ValueError('Unknown initial process for the aircraft')            
            
        # If the aircraft is assigned to a parking pad
        elif aircraft.parking_space_id is not None:
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)

            self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}. Config: Parking pad: {aircraft.parking_space_id}, FATO: {aircraft.assigned_fato_id}')

            # Get Parking pad
            yield self.env.process(self.parking_pad_request(aircraft=aircraft, parking_space_id=aircraft.parking_space_id))

            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()
            
            if aircraft.initial_process in [AircraftStatus.CHARGE, None]:
                # Charging process
                yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))
                aircraft.initial_process = None
            elif aircraft.initial_process == AircraftStatus.IDLE:
                # Put aircraft into available aircraft store
                self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                                aircraft=aircraft)
                aircraft.initial_process = None
            else:
                raise ValueError('Unknown initial process for the aircraft')                 
            
    def aircraft_charging_process(self, aircraft: object):

        # Charge the aircraft
        yield self.env.process(
            aircraft.charge_aircraft(
                parking_space=aircraft.parking_space,
                shared_charger=self.vertiports[aircraft.destination_vertiport_id].shared_charger_sets
            )
        )

        if self.sim_params['only_aircraft_simulation']:
            yield self.env.process(
                self.simulate_aircraft_departure_process(aircraft=aircraft, origin_vertiport_id=aircraft.origin_vertiport_id, destination_vertiport_id=None)
            )                 

    def is_all_passenger_travelled(self):
        return self.trip_counter_tracker == self.total_demand