from .base_system_manager import BaseSystemManager
import simpy
from ..aircraft.aircraft import AircraftStatus
from typing import Dict, List, Any
from collections import deque
import numpy as np
import pandas as pd
import random
from ..utils.get_state_variables import get_simulator_states
from ..utils.units import sec_to_ms, ms_to_sec, ms_to_min, miles_to_m, sec_to_min, min_to_sec, min_to_ms
from ..utils.helpers import get_random_process_id, miliseconds_to_hms, careful_round, duplicate_str
from ..utils.weighted_random_chooser import random_choose_exclude_element

class RLSystemManager(BaseSystemManager):
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
                 stopping_events: Dict,
                 truncation_event: simpy.Event,
                 sim_mode: Dict,
                 periodic_stopping: bool = False,
                 periodic_and_event_driven: bool = False,
                 flight_directions_dict: Dict = None
                 ):
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
        
        self.average_flight_durations_per_od_pair = self._initialize_avg_flight_durations()
        self.external_optimization_params = external_optimization_params
        self.periodic_stopping = periodic_stopping
        self.periodic_and_event_driven = periodic_and_event_driven
        self.stopping_events = stopping_events
        self.truncation_event = truncation_event
        self.departing_passenger_tracker = set()
        self.decision_making_interval = sec_to_ms(self.external_optimization_params['periodic_time_step'])
        self.actions = []
        self.trip_counter = 0
        self.trip_counter_tracker = 0
        self.spill_counter = {(origin, destination): 0 for origin in vertiport_ids for destination in vertiport_ids if origin != destination}
        self.total_demand = None
        self.trip_time_counter = 0
        self.holding_time_counter = 0
        self.truncation_penalty = 0
        self.charge_reward = 0
        self.triggered_stopping_event_queue = deque()
        # self.env.process(self.check_passenger_max_waiting_time_threshold())
    
    def get_random_destination(self, aircraft: object):
        # If the destination vertiport is not assigned, then randomly choose one
        return random_choose_exclude_element(elements_list=self.vertiport_ids,
                                             exclude_element=aircraft.current_vertiport_id,
                                             num_selection=1)[0]

    def reserve_aircraft(self, aircraft: object, 
                         destination_vertiport_id: str = None, 
                         departing_passengers: list = None):
        
        if destination_vertiport_id is None:
            destination_vertiport_id = self.get_random_destination(aircraft=aircraft)
            
        # Truncate if the aircraft is not at the vertiport
        if aircraft.tail_number not in self.get_available_aircraft_tail_numbers(vertiport_id=aircraft.current_vertiport_id):
            print(f"Applying truncation. Aircraft {aircraft.tail_number} is not idling at the vertiport {aircraft.current_vertiport_id}. Aircraft location: {aircraft.location}. Status: {aircraft.status}. Current time: {self.env.now}")
            self.truncation_penalty += 1
            self.trigger_truncation_event(event_name="not_at_vertiport_truncation_event", id=aircraft.tail_number)

        # Start tracking aircraft allocation time
        aircraft_allocation_start_time = self.env.now
        # Update the vertiport flight request state
        self.vertiports[aircraft.current_vertiport_id].num_flight_requests += 1           
        # Log the aircraft allocation process
        flight_id = get_random_process_id()
        # self.logger.debug(f'|{flight_id}| Started: Aircraft allocation process at vertiport {aircraft.current_vertiport_id} for {destination_vertiport_id}. Aircraft tail number: {aircraft.tail_number}')
        # Get the intended pushback time of the aircraft
        pushback_time = self.env.now             
        # Retrieve aircraft from the store
        aircraft = yield self.retrieve_aircraft(aircraft.current_vertiport_id, aircraft.tail_number) 
        # Set the status of the aircraft
        aircraft.status = AircraftStatus.FLY   
        # Set the destination vertiport of the aircraft
        aircraft.destination_vertiport_id = destination_vertiport_id  
        # print(f"Reserved aircraft: {aircraft.tail_number} at {aircraft.location}. Time: {self.env.now}")
        # Update the number of aircraft at the vertiport
        self.update_ground_aircraft_count(vertiport_id=aircraft.current_vertiport_id, update=-1)        
        # Update the vertiport flight request state
        self.vertiports[aircraft.current_vertiport_id].num_flight_requests -= 1    
        # # Check the waiting room for passengers
        # departing_passengers = self.scheduler.collect_departing_passengers_by_od_vertiport(
        #     origin_vertiport_id=aircraft.current_vertiport_id, destination_vertiport_id=destination_vertiport_id)                  
        # Set the real pushback time of the aircraft 
        aircraft.pushback_time = pushback_time
        # Set the flight direction of the aircraft
        aircraft.flight_direction = f'{aircraft.current_vertiport_id}_{destination_vertiport_id}'
        # Set the flight ID of the aircraft
        aircraft.flight_id = flight_id
        # Measure and save the aircraft allocation time
        self.save_aircraft_allocation_time(aircraft.current_vertiport_id, aircraft, aircraft_allocation_start_time)
        # Log the aircraft allocation process
        # self.logger.debug(f'|{flight_id}| Completed: Aircraft allocation process at vertiport {aircraft.current_vertiport_id} for {destination_vertiport_id}. Aircraft tail number: {aircraft.tail_number}, SOC: {careful_round(aircraft.soc, 2)}, Departing passengers: {len(departing_passengers)}')

        # Save aircraft allocation and idle times
        if not self.sim_params['light_simulation']:
            self.save_aircraft_times(origin_vertiport_id=aircraft.current_vertiport_id, aircraft=aircraft)

        # Assign passengers to the aircraft
        aircraft.passengers_onboard = departing_passengers     

        # if len(departing_passengers) == 0:
        #     occupancy = self.external_optimization_params['reward_function_parameters']['empty_flight_penalty']
        # elif len(departing_passengers) > 0:
        #     occupancy = len(departing_passengers)
        occupancy = len(departing_passengers)

        # Increase the passenger trip counter
        self.trip_counter += occupancy             

        self.trip_time_counter += sec_to_min(self.get_average_flight_time(flight_direction=aircraft.flight_direction))
        # self.trip_time_counter += 1

        # Save total passenger waiting time and flight assignment time
        self.save_passenger_times(origin_vertiport_id=aircraft.current_vertiport_id, passengers=aircraft.passengers_onboard) 
        # self.passenger_logger.info(f"Passenger waiting times for flight from {aircraft.current_vertiport_id} to {destination_vertiport_id}: Passenger ids : {[p.passenger_id for p in departing_passengers]} waiting times: {[miliseconds_to_hms(self.env.now - passenger.waiting_room_arrival_time) for passenger in departing_passengers]}")

        # Start the passenger departure process
        if departing_passengers:
            # self.logger.debug(f'|{flight_id}| Started: Passengers exited the waiting room at vertiport {aircraft.current_vertiport_id} for {destination_vertiport_id} flight.')
            self.env.process(self.initiate_passenger_departure(departing_passengers=departing_passengers))
            # self.logger.debug(f'|{flight_id}| Completed: Passengers arrived at the boarding gate at vertiport {aircraft.current_vertiport_id} for {destination_vertiport_id} flight.')

        # # Trigger the stopping event for aircraft departure
        # self.logger.debug(f'aircraft_departure_event is triggered for {aircraft.tail_number} time {self.env.now}')
        # self.trigger_stopping_event(event_name="aircraft_departure_event", id=aircraft.tail_number)
        
        # Start the aircraft departure process
        yield self.env.process(
            self.simulate_aircraft_departure_process(aircraft=aircraft,
                                                    origin_vertiport_id=aircraft.current_vertiport_id, 
                                                    destination_vertiport_id=destination_vertiport_id)
        )

    def simulate_terminal_airspace_arrival_process(self, aircraft: object, arriving_passengers: list):
        # print(f"Terminal airspace arrival process for {aircraft.tail_number} at {aircraft.location}. Origin: {aircraft.origin_vertiport_id}. Destination: {aircraft.destination_vertiport_id} Time: {self.env.now}")
        aircraft.holding_start = self.env.now

        aircraft.status = AircraftStatus.HOLD

        # Increase the holding aircraft counter of the vertiport
        self.vertiports[aircraft.destination_vertiport_id].num_holding_aircraft += 1
        self.vertiports[aircraft.destination_vertiport_id].holding_times[aircraft.tail_number]['start'] = self.env.now
        
        # Increase aircraft arrival queue counter
        self.event_saver.update_aircraft_arrival_queue_counter(vertiport_id=aircraft.destination_vertiport_id,
                                                               queue_update=1)
        
        # Request arrival fix resource
        arrival_fix_usage_request, arrival_fix_resource = self.request_fix_resource(flight_direction=aircraft.flight_direction, operation_type='arrival')
        # self.logger.debug(f'Aircraft {aircraft.tail_number} requesting arrival fix resource at {aircraft.destination_vertiport_id}.'
        #                  f' Number of holding aircraft queued at {aircraft.destination_vertiport_id}: {len(arrival_fix_resource.queue)}'
        #                  f' Number of available aircraft at {aircraft.destination_vertiport_id}: {self.check_num_available_aircraft(aircraft.destination_vertiport_id)}')
        # Trigger the stopping event
        # self.logger.debug(f'aircraft_faf_arrival_event is triggered at {aircraft.destination_vertiport_id} time {self.env.now}')
        # self.trigger_stopping_event(event_name="aircraft_faf_arrival_event", id=aircraft.destination_vertiport_id)

        # # Start the SOC update process
        self.env.process(self.update_soc_during_holding(aircraft))

        # Wait until getting the arrival fix usage request
        yield arrival_fix_usage_request
        # self.logger.debug(f'Aircraft {aircraft.tail_number} has been assigned to the arrival fix resource at {aircraft.destination_vertiport_id}')

        aircraft.status = AircraftStatus.FLY

        aircraft.holding_end = self.env.now
        aircraft.holding_time = aircraft.holding_end - aircraft.holding_start
        
        # Delete the holding time from the vertiport holding times
        del self.vertiports[aircraft.destination_vertiport_id].holding_times[aircraft.tail_number]

        # Decrease the holding aircraft counter of the vertiport
        self.vertiports[aircraft.destination_vertiport_id].num_holding_aircraft -= 1

        self.holding_time_counter += ms_to_min(aircraft.holding_time)

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
                # self.logger.debug(f'Aircraft {aircraft.tail_number} updated SoC during holding at {aircraft.destination_vertiport_id}. Current SoC: {careful_round(aircraft.soc, 2)}')

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

            aircraft.flight_direction = None

            # Put aircraft into available aircraft store
            self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                            aircraft=aircraft)  
                       
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)
            self.update_flying_aircraft_count(update=-1)
            self.save_passenger_trip_times(aircraft=aircraft, flight_direction=flight_direction)

            aircraft.status = AircraftStatus.IDLE  

            # self.logger.info(f"Aircraft {aircraft.tail_number} arrived with passengers: "
            #                     f"{RLSystemManager.get_pax_ids_onboard(aircraft)} at "
            #                     f"{aircraft.destination_vertiport_id} at time : {miliseconds_to_hms(self.env.now)}. "
            #                     f"Available aircraft at {aircraft.destination_vertiport_id}: "
            #                     f"{self.get_available_aircraft_ids_at_vertiport(aircraft.destination_vertiport_id)}. "
            #                     f"Number of aircraft at {aircraft.destination_vertiport_id}: "
            #                     f"{self.get_num_aircraft_at_vertiport(aircraft.destination_vertiport_id)}. "
            #                     f"Aircraft status at {aircraft.destination_vertiport_id}: "
            #                     f"{self.get_status_of_aircraft_at_vertiport(aircraft.destination_vertiport_id)}.")

            # self.logger.debug(f'|{duplicate_str(aircraft.tail_number)}| Aircraft {aircraft.tail_number} is now ready for allocation.')

            # Set the FLY process complition True
            aircraft.is_process_completed = True 

            self.trip_counter_tracker += len(aircraft.passengers_onboard) 
            aircraft.passengers_onboard = []
                    
            # Trigger the stopping event
            # self.logger.debug(f'aircraft_parking_pad_arrival_event is triggered for {aircraft.tail_number} time {self.env.now}')
            self.trigger_stopping_event(event_name="aircraft_parking_pad_arrival_event", id=aircraft.tail_number)

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
            
            aircraft.status = AircraftStatus.IDLE
                   
            # Put aircraft into available aircraft store
            self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                            aircraft=aircraft)   

            # Trigger the stopping event
            # self.logger.debug(f'aircraft_parking_pad_arrival_event is triggered for {aircraft.tail_number} time {self.env.now}')
            self.trigger_stopping_event(event_name="aircraft_parking_pad_arrival_event", id=aircraft.tail_number)   

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

            # self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}. Config: Parking pad: {aircraft.parking_space_id}, FATO: {aircraft.assigned_fato_id}')

            # Put aircraft into available aircraft store
            self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                            aircraft=aircraft) 
            aircraft.status = AircraftStatus.IDLE
            # self.logger.debug(f'|{duplicate_str(aircraft.tail_number)}| Aircraft {aircraft.tail_number} is now ready for allocation.')         
            # Trigger the stopping event
            # self.logger.debug(f'aircraft_parking_pad_arrival_event is triggered for {aircraft.tail_number} time {self.env.now}')
            self.trigger_stopping_event(event_name="aircraft_parking_pad_arrival_event", id=aircraft.tail_number)
        
        # Only FATO case and the starting location is a FATO
        elif aircraft.assigned_fato_id is not None:
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)

            # self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}. Config: Parking pad: {aircraft.parking_space_id}, FATO: {aircraft.assigned_fato_id}')

            # Get FATO
            yield self.env.process(self.fato_pad_request(aircraft=aircraft, operation_type='arrival', fato_id=aircraft.assigned_fato_id))
            # Put aircraft into available aircraft store

            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()

            self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                            aircraft=aircraft) 
            # self.logger.debug(f'|{duplicate_str(aircraft.tail_number)}| Aircraft {aircraft.tail_number} is now ready for allocation.')
            aircraft.status = AircraftStatus.IDLE            
            # Trigger the stopping event
            # self.logger.debug(f'aircraft_parking_pad_arrival_event is triggered for {aircraft.tail_number} time {self.env.now}')
            self.trigger_stopping_event(event_name="aircraft_parking_pad_arrival_event", id=aircraft.tail_number) 
            
        # If the aircraft is assigned to a parking pad
        elif aircraft.parking_space_id is not None:
            # Update number of aircraft status at the vertiports
            self.update_ground_aircraft_count(vertiport_id=aircraft.destination_vertiport_id, update=1)

            # self.logger.debug(f'Created: Aircraft {aircraft.tail_number} created at {aircraft.origin_vertiport_id} with SOC: {aircraft.soc}, location: {aircraft.location}. Config: Parking pad: {aircraft.parking_space_id}, FATO: {aircraft.assigned_fato_id}')

            # Get Parking pad
            yield self.env.process(self.parking_pad_request(aircraft=aircraft, parking_space_id=aircraft.parking_space_id))

            aircraft.current_vertiport_id = aircraft.get_aircraft_vertiport()

            if aircraft.initial_process == AircraftStatus.CHARGE:
                # Charging process
                yield self.env.process(self.aircraft_charging_process(aircraft=aircraft))
            elif aircraft.initial_process == AircraftStatus.IDLE:
                # Put aircraft into available aircraft store
                self.put_aircraft_into_available_aircraft_store(vertiport_id=aircraft.destination_vertiport_id,
                                                                aircraft=aircraft)
                aircraft.status = AircraftStatus.IDLE
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

# PASSENGER RELATED METHODS
# -------------------------
    def simulate_passenger_arrival(self, passenger):
        """ Simulate passenger."""
        # Save passenger creation time
        self.event_saver.save_agent_state(agent=passenger, agent_type='passenger', event='vertiport_entrance')
        passenger.vertiport_arrival_time = self.env.now

        # Put the passenger into the waiting room
        yield self.env.process(self.put_passenger_into_waiting_room(passenger))

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
        # self.passenger_logger.info(f'Passenger {passenger.passenger_id} entered the waiting room at {passenger.origin_vertiport_id}. Currently waiting passengers: {[p.passenger_id for p in passengers_waiting_room.items]}.')

        event_name=f"passenger_arrival_event_{passenger.passenger_id % 100}"
        # self.logger.info(f'passenger_arrival_event is triggered for passenger {passenger.passenger_id} time: {self.env.now}')
        # print(f"Passenger {passenger.passenger_id} entered the waiting room at {passenger.origin_vertiport_id}. Currently waiting passengers: {[p.passenger_id for p in passengers_waiting_room.items]}. Time (sec): {round(self.env.now/1000)}")        
        self.trigger_stopping_event(event_name=event_name)

        if passenger.passenger_id == self.last_passenger_id:
            self.passenger_arrival_complete = True
            self.logger.info(f'Passenger arrival process is completed. Last passenger id: {self.last_passenger_id}')        

        yield self.env.timeout(0)
        
    def remove_spilled_passengers(self, actions: List):
        """
        Remove the spilled passengers from the waiting rooms only if there's no scheduled flight,
        or if there are more passengers than the flight's capacity.
        """
        # self.logger.debug(f'Checking potential spilled passengers at {miliseconds_to_hms(self.env.now)}')

        max_waiting_time_threshold = sec_to_ms(self.sim_params['max_passenger_waiting_time']) 
        waiting_time_threshold_check = max_waiting_time_threshold - min_to_sec(1) - self.decision_making_interval
        # First identify which flights are scheduled and the capacity of each flight direction
        scheduled_flights = {(origin, destination): 0 for origin in self.vertiport_ids for destination in self.vertiport_ids if origin != destination}
        for aircraft_id, aircraft in self.aircraft_agents.items():
            if actions[aircraft_id] < len(self.vertiport_ids):
                origin = aircraft.current_vertiport_id
                destination = self.vertiport_index_to_id_map[actions[aircraft_id]]
                scheduled_flights[(origin, destination)] += aircraft.passenger_capacity

        # Remove passengers only if there are no scheduled flights or the waiting passengers exceed flight capacity
        for origin, vertiport in self.vertiports.items():
            for destination, wr_store in vertiport.waiting_room_stores.items():
                num_agents_in_the_wr = len(wr_store.items)
                if num_agents_in_the_wr > 0:
                    # Identify passengers that have waited too long
                    over_waiting_passengers = [
                        passenger for passenger in wr_store.items
                        if self.env.now - passenger.waiting_room_arrival_time > waiting_time_threshold_check # To account for the included flying pax waiting time
                    ]
                    num_over_waiting_passengers = len(over_waiting_passengers)

                    # Check if there's a scheduled flight from origin to destination
                    if scheduled_flights[(origin, destination)] > 0:
                        # Get the passenger capacity of the flight
                        flight_capacity = scheduled_flights[(origin, destination)]

                        # If the number of over-waiting passengers exceeds the flight capacity
                        if num_over_waiting_passengers > flight_capacity:
                            # Remove only the passengers exceeding the capacity
                            excess_passengers = num_over_waiting_passengers - flight_capacity
                            for _ in range(excess_passengers):
                                spilled_passenger = over_waiting_passengers.pop(0)  # Remove the first passenger in excess

                                # Remove the passenger from the waiting room store
                                self.spill_passenger(spilled_passenger, origin, destination)
                                # self.event_saver.update_spilled_passenger_counter(flight_dir=f"{origin}_{destination}")
                                # self.logger.info(f'Excess passenger {spilled_passenger.passenger_id} has been spilled at {origin} vertiport after waiting {ms_to_min(self.env.now - spilled_passenger.waiting_room_arrival_time)} min.')
                                # self.logger.info(f'Flight capacity: {flight_capacity}. Number of over-waiting passengers: {num_over_waiting_passengers}. Number of excess passengers: {excess_passengers}. Flights scheduled: {scheduled_flights}')
                        else:
                            # If there's enough capacity, keep the passengers
                            continue
                    else:
                        # If no flight is scheduled, spill all over-waiting passengers
                        for waiting_passenger in over_waiting_passengers:
                            self.spill_passenger(waiting_passenger, origin, destination)
                            # self.logger.info(f'Passenger {waiting_passenger.passenger_id} has been spilled at {origin} vertiport after waiting {ms_to_min(self.env.now - waiting_passenger.waiting_room_arrival_time)} min.')
                            # self.logger.info(f'Number of over-waiting passengers: {num_over_waiting_passengers}. Flights scheduled: {scheduled_flights}')
        # yield self.env.timeout(0)

    def check_passenger_max_waiting_time_threshold(self):
        """
        Remove passengers who have waited longer than the maximum waiting time threshold.
        """
        max_waiting_time_threshold = sec_to_ms(self.sim_params['max_passenger_waiting_time'] - min_to_sec(1))  # To account for the included flying pax waiting time

        while True:
            yield self.env.timeout(min_to_ms(1))
            # Do not operate at the multiples of 5 minutes
            if self.env.now % sec_to_ms(self.external_optimization_params['periodic_time_step']) == 0:
                # self.logger.debug(f'Skipping passenger waiting time threshold at {miliseconds_to_hms(self.env.now)}')
                continue

            # self.logger.debug(f'Checking passenger waiting time threshold at {miliseconds_to_hms(self.env.now)}')

            for origin, vertiport in self.vertiports.items():
                for destination, wr_store in vertiport.waiting_room_stores.items():
                    num_agents_in_the_wr = len(wr_store.items)
                    if num_agents_in_the_wr > 0:
                        # Identify passengers that have waited too long
                        over_waiting_passengers = [
                            passenger for passenger in wr_store.items
                            if self.env.now - passenger.waiting_room_arrival_time >= max_waiting_time_threshold
                        ]
                        for waiting_passenger in over_waiting_passengers:
                            self.spill_passenger(waiting_passenger, origin, destination)

    def spill_passenger(self, passenger, origin, destination):
        """
        Handle the spillover of a passenger.
        """
        self.event_saver.update_spilled_passenger_counter(flight_dir=f"{origin}_{destination}")
        self.logger.info(f'Passenger {passenger.passenger_id} spilled at {origin} after waiting {ms_to_min(self.env.now - passenger.waiting_room_arrival_time)} min. Waiting time of the passengers: {[ms_to_min(self.env.now - p.waiting_room_arrival_time) for p in self.vertiports[origin].waiting_room_stores[destination].items]}')
        self.spill_counter[(origin, destination)] += 1
        self.vertiports[origin].spilled_passengers_dict[destination] = self.spill_counter[(origin, destination)] 
        self.logger.debug(f"Spilled passenger dict: {self.vertiports[origin].spilled_passengers_dict}")
        self.remove_pax_from_waiting_room(origin, destination, passenger)

    def remove_pax_from_waiting_room(self, origin, destination, passenger):
        """
        Remove the passenger from the waiting room.
        """
        self.vertiports[origin].waiting_room_stores[destination].get(lambda x: x.passenger_id == passenger.passenger_id)
        # self.logger.debug(f'Passenger {passenger.passenger_id} has been removed from the waiting room at {origin} vertiport.')

    def trigger_stopping_event(self, event_name: str, id: int = None):
        """
        Trigger the stopping event.
        """
        if self.periodic_stopping and not self.periodic_and_event_driven:
            return
        if id is not None:
            event_name = f"{event_name}_{id}"

        # If there are already triggered stopping events happening at the same time, merge them
        
        if len(self.triggered_stopping_event_queue) > 0:
            last_event_name, last_time = self.triggered_stopping_event_queue[-1]
            if last_time == self.env.now:
                self.triggered_stopping_event_queue[-1] = (f"{last_event_name}_{event_name}", last_time)
                event_name = f"{last_event_name}_{event_name}"
            else:
                self.triggered_stopping_event_queue.append((event_name, self.env.now))

    def trigger_truncation_event(self, event_name: str, id: int = None):
        """
        Trigger the truncation event.
        """
        # if id is not None:
        #     event_name = f"{event_name}_{id}"
        # self.truncation_events[event_name].succeed() 
        self.truncation_event.succeed()

    def is_all_passenger_travelled(self):
        return self.trip_counter_tracker == self.total_demand
    
    @staticmethod
    def get_pax_ids_onboard(aircraft: object):
        return [p.passenger_id for p in aircraft.passengers_onboard]
    
    def get_available_aircraft_ids_at_vertiport(self, vertiport_id: str):
        return [a.tail_number for a in self.vertiports[vertiport_id].available_aircraft_store.items]

    def get_num_aircraft_at_vertiport(self, vertiport_id: str):
        return self.vertiports[vertiport_id].num_aircraft
    
    def get_status_of_aircraft_at_vertiport(self, vertiport_id: str):
        # return [(aircraft.current_vertiport_id, aircraft.status) for aircraft in self.aircraft_agents.values()]
        return [aircraft.status for aircraft in self.aircraft_agents.values() if aircraft.current_vertiport_id == vertiport_id]
    
    def get_num_waiting_passengers_per_vertiport(self, vertiport_id):
        """
        Check the number of waiting passengers at the given vertiport.
        """
        return self.vertiports[vertiport_id].get_total_waiting_passenger_count()