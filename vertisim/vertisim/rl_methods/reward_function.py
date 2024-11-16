from typing import List, Dict, Set
import numpy as np
from ..aircraft.aircraft import AircraftStatus
from ..utils.units import sec_to_ms, sec_to_min
from ..utils import rl_utils
from ..utils.helpers import (
    current_and_lookahead_pax_count,
    get_waiting_passenger_ids,
    get_total_waiting_passengers_at_vertiport,
    extract_dict_values
)
from collections import Counter, defaultdict
import math

class RewardFunction:
    def __init__(self, config: Dict, reverse_action_dict: Dict, sim_setup: object, logger: object) -> None:
        self.config = config
        self.reverse_action_dict = reverse_action_dict
        self.sim_setup = sim_setup
        self.reward = 0
        self.logger = logger

    ### ------- REWARD FUNCTIONS ------- ###
    def spill_reward(self, actions: List[int]) -> None:
        revenue = self._calculate_revenue(actions)
        # Energy cost
        energy_cost = self._calculate_energy_cost(actions)
        # Spit cost
        spill_cost = self._calculate_spill_cost()
        # Operating cost
        operating_cost = self._calculate_operating_cost(actions)
        # Charge reward
        # charge_reward = self._apply_charge_rewards(actions)
        # Repositioning reward
        # repositioning_reward = self._repositioning_reward_for_expected_pax(actions)
        # FLight action reward/penalty
        # flight_rewards = self._flight_rewards(actions)
        # Calculate the reward
        self.reward = revenue - (energy_cost + spill_cost + operating_cost)

    def simplified_spill_reward(self, actions: List[int]) -> None:
        # Spit cost
        self.reward = self._calculate_spill_cost()

    def dollar_cost_reward(self, actions: List[int]) -> None:
        # Reward Function: Profit = Revenue - (Energy Cost + Operating Cost)
        revenue = self._calculate_revenue(actions)
        if self.config['external_optimization_params']['reward_function_parameters']['usd_energy_per_kwh'] != 0:
            energy_cost = self._calculate_energy_cost(actions)
        else:
            energy_cost = 0
        operating_cost = self._calculate_operating_cost(actions)
        spill_cost = self._calculate_spill_cost()
        repositioning_reward = self._calculate_repositioning_reward(actions)
        charge_reward = self._apply_charge_rewards(actions)
        self.reward = repositioning_reward + revenue + charge_reward - (energy_cost + operating_cost + spill_cost)         

    def compute_reward(self, actions: List[int]) -> None:
        """
        Calculate the reward based on the provided actions.
        """
        # self.dollar_cost_reward(actions)
        # self.simplified_spill_reward(actions)
        self.spill_reward(actions)
        
        # departing_passengers = self._get_departing_passengers(actions)

        # self._apply_simplified_flight_rewards(actions=actions, 
        #                                   departing_passengers=departing_passengers)
        # self.reward += self._compute_waiting_time_penalty(departing_passengers=departing_passengers)
        # self.reward += self._compute_holding_penalty()    

    def add_spill_cost(self) -> None:
        """
        Add spill cost to the reward.
        """
        self.reward -= self._calculate_spill_cost()

    def _compute_energy_draw(self, actions: List[int]) -> float:
        """
        Checks each aircraft's SoC and the charging time from the config to calculate the energy draw.
        """
        total_draw_kwh = 0
        # Iterate through each aircraft and calculate the energy draw
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] == self.reverse_action_dict['CHARGE']:
                # Calculate the final SoC after charging for the given charge duration and the initial SoC
                final_soc = aircraft.charging_strategy.calc_soc_from_charge_time(
                    charge_time=sec_to_ms(self.config['external_optimization_params']['charge_time_per_charge_event']),
                    initial_soc=aircraft.soc,
                    df=aircraft.system_manager.aircraft_battery_models)
                # Calculate the energy draw using the SoC difference
                energy_draw_soc = final_soc - aircraft.soc
                # Convert the energy draw from SoC to kWh
                energy_draw_kwh = round(energy_draw_soc * self.config['aircraft_params']['battery_capacity'] / 100, 2)
                total_draw_kwh += energy_draw_kwh
        return total_draw_kwh
    
    def _calculate_energy_cost(self, actions: List[int]) -> float:
        """
        Calculate the energy cost for the given actions.
        """
        return self._compute_energy_draw(actions) * self.config['external_optimization_params']['reward_function_parameters']['usd_energy_per_kwh']

    def _calculate_operating_cost(self, actions: List[int]) -> float:
        """
        Calculate the operating cost for the given actions.
        """
        return self._calculate_total_flight_distance(actions) *\
              self.config['external_optimization_params']['reward_function_parameters']['usd_casm'] *\
            self.config['aircraft_params']['pax']
    
    def _calculate_spill_cost(self) -> float:
        """
        Calculate the spill cost for the given actions.
        """
        return np.sum(extract_dict_values(self.sim_setup.system_manager.spill_counter)) * \
            self.config['external_optimization_params']['reward_function_parameters']['spill_cost']

    def _calculate_revenue(self, actions: List[int]) -> float:
        """
        Calculate the revenue for the given departing passengers.
        """
        dest_departing_pax_pair = self._get_destination_departing_pax_pair(actions)
        pax_mile = 0
        total_departing_pax = sum(dest_departing_pax_pair.values())
        for (origin_id, dest_id), num_departing_pax in dest_departing_pax_pair.items():
            pax_mile += num_departing_pax * self._lookup_trip_length(origin_id, dest_id)
        revenue = pax_mile * self.config['external_optimization_params']['reward_function_parameters']['usd_pax_mile']
        revenue +=  total_departing_pax * self.config['external_optimization_params']['reward_function_parameters']['usd_base_fare']
        return revenue

    def _get_destination_departing_pax_pair(self, actions: List[int]) -> Dict:
        """
        Get the departing passengers count for Origin-destination pair.
        """
        dest_departing_pax_pair = {(origin_id, dest_id): set() for origin_id in self.sim_setup.vertiports.keys() for dest_id in self.sim_setup.vertiports.keys() if origin_id != dest_id}
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] < self.sim_setup.num_vertiports():
                dest_vertiport_id = self._get_destination_vertiport_id(aircraft_id=aircraft_id, actions=actions)
                dest_departing_pax_pair[(aircraft.current_vertiport_id, dest_vertiport_id)].update(
                    get_waiting_passenger_ids(
                        sim_setup=self.sim_setup,
                        exclude_pax_ids=dest_departing_pax_pair[(aircraft.current_vertiport_id, dest_vertiport_id)],
                        origin_vertiport_id=aircraft.current_vertiport_id,
                        destination_vertiport_id=dest_vertiport_id
                    ))
        # Replace the passenger set with the count
        for key, value in dest_departing_pax_pair.items():
            dest_departing_pax_pair[key] = len(value)
        return dest_departing_pax_pair

    def _compute_holding_penalty(self) -> float:
        """
        Compute the holding penalty.
        """
        holding_time = 0
        for aircraft in self.sim_setup.system_manager.aircraft_agents.values():
            holding_time += aircraft.get_holding_time()
        return holding_time * self.config['external_optimization_params']['reward_function_parameters']['holding_penalty']

    def _get_departing_passengers(self, actions: List[int]) -> Set[int]:
        """
        Compute the set of departing passengers.
        """
        departing_passengers = set()
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] < self.sim_setup.num_vertiports():
                departing_passengers.update(get_waiting_passenger_ids(
                    sim_setup=self.sim_setup,
                    exclude_pax_ids=departing_passengers,
                    origin_vertiport_id=aircraft.current_vertiport_id,
                    destination_vertiport_id=self._get_destination_vertiport_id(aircraft_id=aircraft_id, actions=actions)
                ))
        
        self.sim_setup.system_manager.departing_passenger_tracker = departing_passengers
        return departing_passengers
    
    def _get_total_departing_pax_count(self, departing_passengers: Set) -> int:
        """
        Get the total departing passengers count.
        """
        return len(departing_passengers)

    def _get_num_flights(self, actions: List[int]) -> int:
        """
        Get the number of flights for the given actions.
        """
        return sum(1 for action in actions if action < self.sim_setup.num_vertiports())
    
    def _calculate_total_flight_distance(self, actions: List[int]) -> int:
        """
        Calculate the total flight distance for the given actions.
        """
        total_distance = 0
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] < self.sim_setup.num_vertiports():
                dest_vertiport_id = self._get_destination_vertiport_id(aircraft_id=aircraft_id, actions=actions)
                total_distance += self._lookup_trip_length(origin_id=aircraft.current_vertiport_id, destination_id=dest_vertiport_id)
        return total_distance
    
    def _flight_rewards(self, actions: List[int]) -> float:
        """
        1. Penalizes the agent if it selects more than enough flight actions from the origin vertiport and
          there are enough aircraft to serve the demand at the destination vertiport.
          
          """
    
    def _apply_simplified_flight_rewards(self, actions: List[int], departing_passengers: Set[int]) -> None:
        pax_count = len(departing_passengers)
        total_trip_length = self._calculate_total_flight_distance(actions)        
        self._apply_trip_and_flight_cost_reward(pax_count, total_trip_length)

    def _apply_charge_rewards(self, actions: List[int]) -> None:
        """
        Apply charge-based rewards and penalties.
        """
        charge_reward = 0
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] == self.reverse_action_dict['CHARGE']:
                charge_reward += self._get_charge_reward(aircraft)
        return charge_reward

    def _apply_flight_rewards(self, actions: List[int], departing_passengers: Set[int]) -> None:
        """
        Apply flight-based rewards and penalties.
        """
        pax_count = len(departing_passengers)
        trip_time_counter = 0
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] < self.sim_setup.num_vertiports():
                dest_vertiport_id = self._get_destination_vertiport_id(aircraft_id=aircraft_id, actions=actions)
                trip_time_counter += self._lookup_trip_length(origin_id=aircraft.current_vertiport_id, destination_id=dest_vertiport_id)
                self._apply_unnecessary_flight_penalty(dest_id=dest_vertiport_id, 
                                                       origin_id=aircraft.current_vertiport_id)
                
                # Penalize if there is an aircraft already headed to the same destination vertiport and
                # that aircraft + existing aircraft is suffient to serve the demand.
                self._penalize_for_not_counting_arriving_aircraft(dest_id=dest_vertiport_id)

            elif actions[aircraft_id] == self.reverse_action_dict['CHARGE']:
                self.reward += self._get_charge_reward(aircraft)

            self.reward += self.compute_action_reward_wrt_pax_count(aircraft=aircraft, action=actions[aircraft_id])

        self._apply_trip_and_flight_cost_reward(pax_count, trip_time_counter)

    def _lookup_trip_length(self, origin_id: int, destination_id: int) -> int:
        """
        Calculate the trip time for a flight.
        """
        return self.sim_setup.system_manager.get_mission_length(
            origin_vertiport_id=origin_id,
            destination_vertiport_id=destination_id
        )
    
    def _get_destination_vertiport_id(self, aircraft_id: int, actions: List[int]) -> int:
        """
        Get the destination vertiport id for a given aircraft.
        """
        return self.sim_setup.vertiport_index_to_id_map[actions[aircraft_id]]
    
    def _penalize_for_not_counting_arriving_aircraft(self, dest_id: int) -> None:
        # Get the aircraft that is actually flying (not on the ground) and is headed to the destination vertiport
        lookahead_aircraft_count = 0
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if aircraft.status == AircraftStatus.FLY and \
                aircraft.flight_direction == f"{aircraft.current_vertiport_id}_{dest_id}" and \
                    aircraft.soc > self.config['aircraft_params']['min_reserve_soc']:
                lookahead_aircraft_count += 1
        lookahead_aircraft_count += self.sim_setup.system_manager.get_num_aircraft_at_vertiport(vertiport_id=dest_id)
        if lookahead_aircraft_count * self.config['aircraft_params']['pax'] >= current_and_lookahead_pax_count(self.sim_setup.vertiports[dest_id]):
            self.reward += self.config['external_optimization_params']['reward_function_parameters']['unnecessary_flight_penalty']

    def _apply_unnecessary_flight_penalty(self, dest_id: int, origin_id: int) -> None:
        """
        Apply penalty for unnecessary flights and reward for useful repositioning.
        """
        if current_and_lookahead_pax_count(self.sim_setup.vertiports[dest_id]) == 0:
            self.reward += self.config['external_optimization_params']['reward_function_parameters']['unnecessary_flight_penalty']
        elif current_and_lookahead_pax_count(self.sim_setup.vertiports[origin_id]) == 0:
            self.reward += self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']

        # If there are available aircraft at the destination vertiport and the aircraft is sufficient to serve the demand
        # then penalize the empty flight.
        # num_aircraft_at_destination = self.sim_setup.system_manager.get_num_aircraft_at_vertiport(vertiport_id=dest_id)
        # num_expected_pax = current_and_lookahead_pax_count(self.sim_setup.vertiports[dest_id])
        # if num_aircraft_at_destination > 0 and num_expected_pax <= num_aircraft_at_destination * self.config['aircraft_params']['pax']:
        #     self.reward += self.config['external_optimization_params']['reward_function_parameters']['unnecessary_flight_penalty']

    def _get_charge_reward(self, aircraft: object) -> float:
        """
        Calculate the reward for charging.
        """
        if aircraft.system_manager.sim_mode['rl']:
            current_demand = sum(current_and_lookahead_pax_count(vertiport) for vertiport in aircraft.system_manager.vertiports.values()) 
            return rl_utils.charge_reward(
                current_soc=aircraft.soc,
                soc_reward_threshold=aircraft.charging_strategy.soc_reward_threshold,
                current_demand=current_demand
            ) * self.config['external_optimization_params']['reward_function_parameters']['charge_reward']
        return 0

    def _apply_trip_and_flight_cost_reward(self, pax_count: int, trip_time: int) -> None:
        """
        Apply rewards and costs related to trips and flight time.
        """
        self.reward += self.config['external_optimization_params']['reward_function_parameters']['trip_reward'] * pax_count
        self.reward += self.config['external_optimization_params']['reward_function_parameters']['flight_cost'] * trip_time

    def _compute_waiting_time_penalty(self, departing_passengers) -> float:
        """
        Calculate the waiting time penalty.
        """
        waiting_time_cost_type = self.config['external_optimization_params']['reward_function_parameters']['waiting_time_cost_type']
        waiting_time_cost_unit = self.config['external_optimization_params']['reward_function_parameters']['waiting_time_cost_unit']

        waiting_time_total = 0
        for _, vertiport in self.sim_setup.vertiports.items():
            waiting_time_total += np.sum(extract_dict_values(vertiport.get_waiting_time_cost(
                unit=waiting_time_cost_unit,
                type=waiting_time_cost_type,
                exclude_pax_ids=departing_passengers,
                last_decision_time=None
            )))
        return waiting_time_total * self.config['external_optimization_params']['reward_function_parameters']['waiting_time_cost']

    def compute_action_reward_wrt_pax_count(self, aircraft, action):
        """
        Compute the reward or penalty for the given number of waiting passengers. TODO: Needs update
        Danger: The rewards for 'do nothing' and 'charge' when there are no passengers might 
        encourage the agent to avoid repositioning even when it's necessary for future demand.
        """
        num_waiting_pax = get_total_waiting_passengers_at_vertiport(sim_setup=self.sim_setup, vertiport_id=aircraft.current_vertiport_id)
        if num_waiting_pax <= self.config['aircraft_params']['pax']//2 and \
            action == self.reverse_action_dict['CHARGE']:
            return self.config['external_optimization_params']['reward_function_parameters']['no_pax_and_charge_reward']
        elif num_waiting_pax <= self.config['aircraft_params']['pax']//2 and \
            action == self.reverse_action_dict['DO_NOTHING']:
            return self.config['external_optimization_params']['reward_function_parameters']['no_pax_and_do_nothing_reward']   
        # # TODO: This requires to check the energy cons of the flight and the SoC level of the aircraft.
        # # Also needs to check if there are a
        # elif num_waiting_pax == self.config['aircraft_params']['pax'] and \
        #     action == self.reverse_action_dict['CHARGE'] and \
        #     soc >= self.config['aircraft_params']['min_reserve_soc']:
        #     return self.config['external_optimization_params']['reward_function_parameters']['waiting_full_pax_but_charge_penalty']
        # elif num_waiting_pax == self.config['aircraft_params']['pax'] and \
        #     action == self.reverse_action_dict['DO_NOTHING'] and \
        #         soc >= self.config['aircraft_params']['min_reserve_soc']:
        #     return self.config['external_optimization_params']['reward_function_parameters']['waiting_full_pax_but_do_nothing_penalty']
        else:
            return 0  

    def _get_aircraft_assigned_passengers(self, actions: List[int]) -> Dict[int, List]:
        """
        Assign passengers to each aircraft based on capacity and return a mapping
        of aircraft IDs to their assigned passenger IDs.
        """
        # Initialize waiting passengers for each OD pair
        waiting_passengers = {}
        for origin_id in self.sim_setup.vertiports.keys():
            for dest_id in self.sim_setup.vertiports.keys():
                if origin_id != dest_id:
                    waiting_passengers[(origin_id, dest_id)] = list(
                        get_waiting_passenger_ids(
                            sim_setup=self.sim_setup,
                            origin_vertiport_id=origin_id,
                            destination_vertiport_id=dest_id
                        )
                    )

        # Assign passengers to aircraft
        aircraft_passengers = {}
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if actions[aircraft_id] < self.sim_setup.num_vertiports():
                dest_vertiport_id = self._get_destination_vertiport_id(aircraft_id=aircraft_id, actions=actions)
                origin_vertiport_id = aircraft.current_vertiport_id
                od_pair = (origin_vertiport_id, dest_vertiport_id)
                available_passengers = waiting_passengers.get(od_pair, [])
                num_to_assign = min(len(available_passengers), aircraft.passenger_capacity)
                assigned_passengers = available_passengers[:num_to_assign]
                # Remove assigned passengers from waiting_passengers
                waiting_passengers[od_pair] = available_passengers[num_to_assign:]
                # Record assigned passengers
                aircraft_passengers[aircraft_id] = assigned_passengers
            else:
                # Not a flight action
                aircraft_passengers[aircraft_id] = []

        return aircraft_passengers 

    def _repositioning_reward_for_expected_pax(self, actions: List[int]) -> float:
        repositioning_reward = 0
        vertiports = self.sim_setup.vertiports
        system_manager = self.sim_setup.system_manager
        num_seats = self.config['aircraft_params']['pax']

        # Extract the reward parameter once to avoid repeated dictionary lookups
        reward_param = self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']

        # Get departing passengers for all flights
        aircraft_passengers = self._get_aircraft_assigned_passengers(actions)

        # Identify repositioning flights (empty flights)
        repositioning_flights = []
        for aircraft_id, action in enumerate(actions):
            if action < self.sim_setup.num_vertiports() and self._is_empty_flight(aircraft_id, actions, aircraft_passengers):
                origin_vertiport_id = system_manager.aircraft_agents[aircraft_id].current_vertiport_id
                destination_vertiport_id = self._get_destination_vertiport_id(aircraft_id, actions)
                repositioning_flights.append({
                    'aircraft_id': aircraft_id,
                    'origin_id': origin_vertiport_id,
                    'destination_id': destination_vertiport_id
                })

        if not repositioning_flights:
            return 0

        # For each destination, calculate additional supply needed
        additional_supply_needed = {}
        for repositioning_flight in repositioning_flights:
            destination_id = repositioning_flight['destination_id']
            origin_id = repositioning_flight['origin_id']
            destination_vertiport = vertiports[destination_id]
            origin_vertiport = vertiports[origin_id]

            # Skip if origin and destination are the same
            if origin_id == destination_id:
                continue

            # Get future demand at the destination vertiport
            future_demand_at_destination = destination_vertiport.get_total_expected_pax_arrival_count()

            # Calculate total supply at the destination vertiport
            num_aircraft_at_destination = system_manager.get_num_aircraft_at_vertiport(vertiport_id=destination_id)
            incoming_supply_to_destination = self._flying_aircraft_count_to_destination(
                dest_vertiport_id=destination_id, actions=actions
            )
            outgoing_supply_from_destination = self._flying_aircraft_action_count(
                origin_vertiport_id=destination_id, aircraft_passengers=aircraft_passengers, system_manager=system_manager
            )
            # Exclude the repositioning flights themselves for now
            total_supply_at_destination = max(0, num_aircraft_at_destination + incoming_supply_to_destination - outgoing_supply_from_destination)

            # Calculate seats needed at destination
            seats_needed = max(0, future_demand_at_destination - total_supply_at_destination * num_seats)
            additional_supply_needed[destination_id] = seats_needed

        # Keep track of supplied seats to each destination from repositioning flights
        seats_supplied = {destination_id: 0 for destination_id in additional_supply_needed.keys()}

        # Now, process each repositioning flight and reward only if needed
        for repositioning_flight in repositioning_flights:
            aircraft_id = repositioning_flight['aircraft_id']
            origin_id = repositioning_flight['origin_id']
            destination_id = repositioning_flight['destination_id']
            origin_vertiport = vertiports[origin_id]
            destination_vertiport = vertiports[destination_id]

            # Check if additional seats are still needed at the destination
            if seats_supplied[destination_id] >= additional_supply_needed[destination_id]:
                continue  # No more seats needed at this destination

            # Check future demand at the origin vertiport after this aircraft leaves
            origin_future_demand = origin_vertiport.get_total_expected_pax_arrival_count()
            num_aircraft_at_origin = system_manager.get_num_aircraft_at_vertiport(vertiport_id=origin_id)
            incoming_supply_to_origin = self._flying_aircraft_count_to_destination(
                dest_vertiport_id=origin_id, actions=actions
            )
            outgoing_supply_from_origin = self._flying_aircraft_action_count(
                origin_vertiport_id=origin_id, aircraft_passengers=aircraft_passengers, system_manager=system_manager
            )
            # Subtract one from supply at origin since this aircraft is leaving
            total_supply_at_origin = max(0, num_aircraft_at_origin + incoming_supply_to_origin - outgoing_supply_from_origin)

            if origin_future_demand > total_supply_at_origin * num_seats:
                continue  # Not enough supply at origin after departure, skip reward

            # All checks passed, add repositioning reward
            repositioning_reward += reward_param
            # Update seats supplied to destination
            seats_supplied[destination_id] += num_seats

        if repositioning_reward != 0:
            system_manager.logger.debug(f"Repositioning reward is processed: {round(repositioning_reward, 2)}")

        return repositioning_reward


    def _flying_aircraft_action_count(self, origin_vertiport_id: str, aircraft_passengers: Dict, system_manager: object, destination_vertiport_id=None) -> int:
        """
        Count the number of aircraft flying to a vertiport at the current time step.
        """
        if destination_vertiport_id:
            return sum(1 for aircraft_id, pax in aircraft_passengers.items()
                    if system_manager.aircraft_agents[aircraft_id].flight_direction == f"{origin_vertiport_id}_{destination_vertiport_id}" and 
                    len(pax) > 0)
        else:
            return sum(1 for aircraft_id, pax in aircraft_passengers.items()
                    if system_manager.aircraft_agents[aircraft_id].current_vertiport_id == origin_vertiport_id and 
                    system_manager.aircraft_agents[aircraft_id].status == AircraftStatus.FLY)

    def _flying_aircraft_count_to_destination(self, dest_vertiport_id: int, actions: List) -> int:
        """
        Get the number of aircraft flying to the destination vertiport.
        Conditions:
        1. Aircraft is flying
        2. Aircraft is headed to the destination vertiport
        3. The most recent action is NOT a flight action
        """
        flying_aircraft_count = 0
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if aircraft.status == AircraftStatus.FLY and \
                aircraft.flight_direction == f"{aircraft.current_vertiport_id}_{dest_vertiport_id}" and \
                    actions[aircraft_id] >= self.sim_setup.num_vertiports():
                flying_aircraft_count += 1
        return flying_aircraft_count

    def _calculate_repositioning_reward(self, actions: List[int]) -> float:
        """
        Calculates the repositioning reward based on vertiport states and actions.

        Args:
            actions (List[int]): A list of action indices representing current actions for each aircraft.

        Returns:
            float: The total repositioning reward.
        """
        repositioning_reward = 0
        vertiports = self.sim_setup.vertiports
        system_manager = self.sim_setup.system_manager
        num_seats = self.config['aircraft_params']['pax']

        # Extract the reward parameter once to avoid repeated dictionary lookups
        reward_param = self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']

        # # Map each action index to the corresponding destination vertiport ID
        # od_pairs = []
        # for aircraft_id, action_index in enumerate(actions):
        #     if action_index < self.sim_setup.num_vertiports():
        #         destination_id = self._get_destination_vertiport_id(aircraft_id, actions)
        #         origin_id = system_manager.aircraft_agents[aircraft_id].current_vertiport_id
        #         od_pairs.append((origin_id, destination_id))

        # # Precompute the number of flights per destination using Counter for efficiency
        # flights_per_od = Counter(od_pairs)

        # # No repositioning reward if there are no flights
        # if len(flights_per_od) == 0:
        #     return 0  

        # Get departing passengers for all flights
        aircraft_passengers = self._get_aircraft_assigned_passengers(actions)

        if sum(len(pax) for pax in aircraft_passengers.values()) == 0:
            return 0

        # Calculate remaining passengers at each vertiport
        remaining_passengers = {}
        for vertiport_id, vertiport in vertiports.items():
            current_pax = vertiport.get_total_waiting_passenger_count()
            departing_pax = sum(len(pax) for aircraft_id, pax in aircraft_passengers.items() 
                                if system_manager.aircraft_agents[aircraft_id].current_vertiport_id == vertiport_id)
            remaining_passengers[vertiport_id] = max(0, current_pax - departing_pax)        

        # Identify repositioning flights (empty flights)
        repositioning_flights = [
            (
                system_manager.aircraft_agents[aircraft_id].current_vertiport_id,
                self._get_destination_vertiport_id(aircraft_id, actions)
            )
            for aircraft_id, action in enumerate(actions)
            if action < self.sim_setup.num_vertiports() and self._is_empty_flight(aircraft_id, actions, aircraft_passengers)
        ]

        # Precompute the current and lookahead passenger count for each vertiport
        for origin_vertiport_id, origin_vertiport in vertiports.items():
            for destination_id, destination_vertiport in vertiports.items():
                if origin_vertiport_id == destination_id:
                    continue  # Skip same origin and destination

                if (origin_vertiport_id, destination_id) not in repositioning_flights:
                    continue

                demand_at_destination = current_and_lookahead_pax_count(destination_vertiport)

                # Check if there is a passenger demand at the destination
                if demand_at_destination == 0:
                    continue

                # Check if the supply at the destination is already enough
                num_aircraft = system_manager.get_num_aircraft_at_vertiport(vertiport_id=destination_id)
                if (self._flying_aircraft_count_to_destination(destination_id) + num_aircraft) * num_seats >= demand_at_destination:
                    continue

                # If the origin future demand is greater than the supply, then skip the repositioning
                origin_future_demand = origin_vertiport.get_total_expected_pax_arrival_count()
                if origin_future_demand > (system_manager.get_num_aircraft_at_vertiport(vertiport_id=origin_vertiport_id) \
                                           + self._flying_aircraft_count_to_destination(origin_vertiport_id)) * num_seats:
                    continue

                # Lookup trip length once for the valid origin-destination pair
                trip_length = self._lookup_trip_length(origin_id=origin_vertiport_id, destination_id=destination_id)
                repositioning_reward += reward_param * trip_length

        if repositioning_reward != 0:
            system_manager.logger.debug(f"Repositioning reward is processed: {round(repositioning_reward, 2)}")

        return repositioning_reward
    
    def _is_empty_flight(self, aircraft_id: int, actions: List, aircraft_passengers: Dict[int, List]) -> bool:
        """
        Determine if the flight is an empty repositioning flight.
        """
        if actions[aircraft_id] >= self.sim_setup.num_vertiports():
            return False  # Not a flight action

        # Check if there are any passengers assigned to this aircraft
        return len(aircraft_passengers.get(aircraft_id, [])) == 0
    
    def _is_aircraft_flying_to_destination(self, dest_vertiport_id: int, actions: List) -> bool:
        """
        Check if there are no aircraft flying to the destination vertiport.
        Conditions:
        1. Aircraft is flying
        2. Aircraft is headed to the destination vertiport
        3. The most recent action is NOT a flight action        
        """
        for aircraft_id, aircraft in self.sim_setup.system_manager.aircraft_agents.items():
            if aircraft.status == AircraftStatus.FLY and \
                aircraft.flight_direction == f"{aircraft.current_vertiport_id}_{dest_vertiport_id}" and \
                    actions[aircraft_id] >= self.sim_setup.num_vertiports():
                return True
        return False
    
    def _is_repositioning_needed(self, dest_vertiport_id: int) -> bool:
        """
        Check if repositioning is needed for the given destination vertiport.
        """
        num_aircraft_at_destination = self.sim_setup.system_manager.get_num_aircraft_at_vertiport(vertiport_id=dest_vertiport_id)
        # Check if there are no aircraft at the destination vertiport and no aircraft flying to that destination vertiport and
        # there is a passenger demand at the destination vertiport.
        if num_aircraft_at_destination == 0 and \
            not self._is_aircraft_flying_to_destination(dest_vertiport_id) and \
                current_and_lookahead_pax_count(self.sim_setup.vertiports[dest_vertiport_id]) > 0:
            return True
        return False

    def new_calculate_repositioning_reward(self, actions: List[int]) -> float:
        """
        Calculates a repositioning reward based on vertiport states, actions, and near-future demand.
        
        This function encourages dynamic repositioning by considering:
        1. Immediate future passenger demand (next 10 minutes)
        2. Current aircraft distribution
        3. Network balance based on immediate future demand
        
        Args:
        actions (List[int]): A list of action indices representing current actions for each aircraft.
        
        Returns:
        float: The total repositioning reward.
        """
        repositioning_reward = 0
        vertiports = self.sim_setup.vertiports
        system_manager = self.sim_setup.system_manager
        total_aircraft_count = len(system_manager.aircraft_agents)
        reward_param = self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']

        # Map actions to destination vertiport IDs
        destination_ids = [self._get_destination_vertiport_id(aircraft_id, actions) for aircraft_id, action in enumerate(actions) if action < self.sim_setup.num_vertiports()]
        flights_per_destination = Counter(destination_ids)

        if not flights_per_destination:
            return 0

        # Calculate total immediate future demand across all vertiports
        total_future_demand = sum(current_and_lookahead_pax_count(vertiport) for vertiport in vertiports.values())
        
        # Calculate ideal aircraft distribution based on immediate future demand
        ideal_distribution = {v_id: current_and_lookahead_pax_count(vertiport) / max(1, total_future_demand) for v_id, vertiport in vertiports.items()}

        for origin_vertiport_id, origin_vertiport in vertiports.items():
            for destination_id, destination_vertiport in vertiports.items():
                if origin_vertiport_id == destination_id:
                    continue

                # Check immediate future demand
                future_demand = current_and_lookahead_pax_count(destination_vertiport)
                if future_demand == 0:
                    continue

                # Check current aircraft distribution
                current_aircraft_count = system_manager.get_num_aircraft_at_vertiport(vertiport_id=destination_id)
                ideal_aircraft_count = ideal_distribution[destination_id] * total_aircraft_count

                # Calculate demand satisfaction rate
                demand_satisfaction_rate = min(1, current_aircraft_count / max(1, future_demand))

                # Calculate repositioning score
                reposition_score = (
                    (1 - demand_satisfaction_rate) * 0.6 +  # Increased weight for demand satisfaction
                    (ideal_aircraft_count - current_aircraft_count) / max(1, ideal_aircraft_count) * 0.4  # Network balance
                )

                # Apply repositioning reward
                num_flights = flights_per_destination.get(destination_id, 0)
                if num_flights == 1:  # We encourage at most one flight per destination
                    trip_length = self._lookup_trip_length(origin_id=origin_vertiport_id, destination_id=destination_id)
                    repositioning_reward += reward_param * trip_length * reposition_score

        if repositioning_reward != 0:
            system_manager.logger.debug(f"Repositioning reward: {round(repositioning_reward, 2)}")

        return repositioning_reward
    

    # def calculate_repositioning_reward_with_limited_info(self, actions: List[int]) -> float:
    #     """
    #     Calculates a repositioning reward based on vertiport states, actions, and current passenger counts.
        
    #     This function encourages dynamic repositioning by considering:
    #     1. Current passenger demand at each vertiport
    #     2. Current aircraft distribution
    #     3. Network balance based on current demand
        
    #     Args:
    #     actions (List[int]): A list of action indices representing current actions for each aircraft.
        
    #     Returns:
    #     float: The total repositioning reward.
    #     """
    #     repositioning_reward = 0
    #     vertiports = self.sim_setup.vertiports
    #     system_manager = self.sim_setup.system_manager
    #     total_aircraft_count = len(system_manager.aircraft_agents)
    #     reward_param = self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']

    #     # Map actions to destination vertiport IDs
    #     destination_ids = [self._get_destination_vertiport_id(aircraft_id, actions) for aircraft_id, action in enumerate(actions) if action < self.sim_setup.num_vertiports()]
    #     flights_per_destination = Counter(destination_ids)

    #     if not flights_per_destination:
    #         return 0

    #     # Calculate total current demand across all vertiports
    #     total_current_demand = sum(vertiport.get_total_waiting_passenger_count() for vertiport in vertiports.values())
        
    #     # Calculate ideal aircraft distribution based on current demand
    #     ideal_distribution = {v_id: vertiport.get_total_waiting_passenger_count() / max(1, total_current_demand) for v_id, vertiport in vertiports.items()}

    #     for origin_vertiport_id, origin_vertiport in vertiports.items():
    #         for destination_id, destination_vertiport in vertiports.items():
    #             if origin_vertiport_id == destination_id:
    #                 continue

    #             # Get current demand
    #             current_demand = destination_vertiport.get_total_waiting_passenger_count()
    #             if current_demand == 0:
    #                 continue

    #             # Check current aircraft distribution
    #             current_aircraft_count = system_manager.get_num_aircraft_at_vertiport(vertiport_id=destination_id)
    #             ideal_aircraft_count = ideal_distribution[destination_id] * total_aircraft_count

    #             # Calculate demand satisfaction rate
    #             # We'll use a more conservative estimate since we don't have future demand information
    #             demand_satisfaction_rate = min(1, (current_aircraft_count * 4) / max(1, current_demand))  # Assuming 4 seats per aircraft

    #             # Calculate urgency factor
    #             # This factor increases as the number of waiting passengers increases
    #             urgency_factor = min(1, current_demand / (4 * max(1, current_aircraft_count)))

    #             # Calculate repositioning score
    #             reposition_score = (
    #                 (1 - demand_satisfaction_rate) * 0.4 +  # Demand satisfaction
    #                 (ideal_aircraft_count - current_aircraft_count) / max(1, ideal_aircraft_count) * 0.3 +  # Network balance
    #                 urgency_factor * 0.3  # Urgency based on current demand
    #             )

    #             # Apply repositioning reward
    #             num_flights = flights_per_destination.get(destination_id, 0)
    #             if num_flights == 1:  # We still encourage at most one flight per destination
    #                 trip_length = self._lookup_trip_length(origin_id=origin_vertiport_id, destination_id=destination_id)
    #                 repositioning_reward += reward_param * trip_length * reposition_score

    #     if repositioning_reward != 0:
    #         system_manager.logger.debug(f"Current demand repositioning reward: {repositioning_reward}")

    #     return repositioning_reward
    
    def calculate_repositioning_reward_with_limited_info(self, actions: List[int]) -> float:
        """
        Calculates a repositioning reward based on vertiport states, actions, and current passenger counts.
        This function encourages dynamic repositioning by considering:
        1. Current passenger demand at each vertiport
        2. Current aircraft distribution
        3. Network balance based on current demand
        4. Origin vertiport vulnerability
        5. Avoiding excessive repositioning

        Args:
        actions (List[int]): A list of action indices representing current actions for each aircraft.

        Returns:
        float: The total repositioning reward.
        """
        repositioning_reward = 0
        vertiports = self.sim_setup.vertiports
        system_manager = self.sim_setup.system_manager
        total_aircraft_count = len(system_manager.aircraft_agents)
        reward_param = self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']
        
        # Get departing passengers for all flights
        dest_departing_pax_pair = self._get_destination_departing_pax_pair(actions)
        
        # Identify repositioning flights (empty flights)
        repositioning_flights = [
            (aircraft_id, self._get_destination_vertiport_id(aircraft_id, actions))
            for aircraft_id, action in enumerate(actions)
            if action < self.sim_setup.num_vertiports() and self._is_empty_flight(aircraft_id, actions, dest_departing_pax_pair)
        ]

        # Calculate total current demand and ideal distribution
        total_current_demand = sum(vertiport.get_total_waiting_passenger_count() for vertiport in vertiports.values())
        ideal_distribution = {v_id: vertiport.get_total_waiting_passenger_count() / max(1, total_current_demand) for v_id, vertiport in vertiports.items()}

        # Track remaining demand at each vertiport
        remaining_demand = {v_id: vertiport.get_total_waiting_passenger_count() for v_id, vertiport in vertiports.items()}

        for aircraft_id, destination_id in repositioning_flights:
            origin_vertiport_id = system_manager.aircraft_agents[aircraft_id].current_vertiport_id
            origin_vertiport = vertiports[origin_vertiport_id]
            destination_vertiport = vertiports[destination_id]

            # Check origin vertiport vulnerability
            origin_demand = origin_vertiport.get_total_waiting_passenger_count()
            origin_aircraft_count = system_manager.get_num_aircraft_at_vertiport(vertiport_id=origin_vertiport_id)
            if origin_demand > (origin_aircraft_count - 1) * 4:  # Assuming 4 passengers per aircraft
                continue  # Skip this repositioning to avoid potential passenger loss at origin

            # Check if repositioning is needed at the destination
            if remaining_demand[destination_id] <= 0:
                continue  # Skip this repositioning as it's not needed

            current_aircraft_count = system_manager.get_num_aircraft_at_vertiport(vertiport_id=destination_id)
            ideal_aircraft_count = ideal_distribution[destination_id] * total_aircraft_count

            # Calculate demand satisfaction rate
            demand_satisfaction_rate = min(1, (current_aircraft_count * 4) / max(1, remaining_demand[destination_id]))

            # Calculate urgency factor
            urgency_factor = min(1, remaining_demand[destination_id] / (4 * max(1, current_aircraft_count)))

            # Calculate repositioning score
            reposition_score = (
                (1 - demand_satisfaction_rate) * 0.4 +  # Demand satisfaction
                (ideal_aircraft_count - current_aircraft_count) / max(1, ideal_aircraft_count) * 0.3 +  # Network balance
                urgency_factor * 0.3  # Urgency based on current demand
            )

            # Apply repositioning reward
            trip_length = self._lookup_trip_length(origin_id=origin_vertiport_id, destination_id=destination_id)
            repositioning_reward += reward_param * trip_length * reposition_score

            # Update remaining demand
            remaining_demand[destination_id] = max(0, remaining_demand[destination_id] - 4)

        if repositioning_reward != 0:
            system_manager.logger.debug(f"Improved limited info repositioning reward: {round(repositioning_reward, 2)}")

        return repositioning_reward



    def reset_rewards(self):
        """
        Reset the rewards for the current state of the simulation.
        """
        vertiport_ids = self.sim_setup.vertiports.keys()
        self.sim_setup.system_manager.trip_counter = 0
        self.sim_setup.system_manager.trip_time_counter = 0
        self.sim_setup.system_manager.truncation_penalty = 0
        # self.self.sim_setup.system_manager.occupancy_reward = 0
        self.sim_setup.system_manager.spill_counter = {(origin, destination): 0 for origin in vertiport_ids for destination in vertiport_ids if origin != destination}
        self.sim_setup.system_manager.charge_reward = 0
        self.sim_setup.system_manager.holding_time_counter = 0
        self.reward = 0

        # Flush the holding time of the aircraft
        for aircraft in self.sim_setup.system_manager.aircraft_agents.values():
            if aircraft.status != AircraftStatus.HOLD:
                aircraft.holding_time = None
                aircraft.holding_start_time = None
                aircraft.holding_end_time = None

        # Flush the spill passenger count
        for vertiport in self.sim_setup.vertiports.values():
            for destination in vertiport.spilled_passengers_dict:
                vertiport.spilled_passengers_dict[destination] = 0


    def repositioning_reward_3(self, actions: List[int]) -> float:
        """
        Calculates a reward for empty repositioning flights, considering:
        1. All flights (both passenger-carrying and empty)
        2. Current waiting passengers and future demand at both origin and destination
        3. Up-to-date aircraft counts at vertiports after actions
        4. Potential for losing passengers at the origin in the next time step
        5. Excessive repositioning flights relative to remaining demand

        Args:
        actions (List[int]): A list of action indices representing current actions for each aircraft.

        Returns:
        float: The total repositioning reward.
        """
        repositioning_reward = 0
        seat_capacity = self.config['aircraft_params']['pax']
        vertiports = self.sim_setup.vertiports
        system_manager = self.sim_setup.system_manager
        reward_param = self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']
        
        # Get departing passengers for all flights
        dest_departing_pax_pair = self._get_destination_departing_pax_pair(actions)
        
        # Calculate updated waiting bins for all vertiports after considering all flights
        updated_waiting_bins = self._calculate_updated_waiting_bins(actions, dest_departing_pax_pair)

        # Identify repositioning flights (empty flights)
        repositioning_flights = [
            (aircraft_id, self._get_destination_vertiport_id(aircraft_id, actions))
            for aircraft_id, action in enumerate(actions)
            if action < self.sim_setup.num_vertiports() and self._is_empty_flight(aircraft_id, actions, dest_departing_pax_pair)
        ]

        # Track the number of repositioning flights to each destination
        repositioning_count = {}
        # Track the remaining demand at each destination
        remaining_demand = {}

        for aircraft_id, destination_id in repositioning_flights:
            origin_vertiport_id = system_manager.aircraft_agents[aircraft_id].current_vertiport_id
            origin_vertiport = vertiports[origin_vertiport_id]
            destination_vertiport = vertiports[destination_id]

            # Check origin vertiport conditions
            origin_waiting_bins = updated_waiting_bins[origin_vertiport_id]
            origin_current_demand = sum(sum(bins) for bins in origin_waiting_bins.values())
            origin_future_demand = origin_vertiport.get_total_expected_pax_arrival_count()
            origin_total_demand = origin_current_demand + origin_future_demand
            origin_aircraft_count = system_manager.get_num_aircraft_at_vertiport(vertiport_id=origin_vertiport_id)

            # Check if repositioning would leave origin vulnerable
            if origin_total_demand > origin_aircraft_count * 4:  # Assuming 4 passengers per aircraft
                continue  # Skip this repositioning to avoid potential passenger loss at origin

            # Calculate remaining demand at destination if not already calculated
            if destination_id not in remaining_demand:
                dest_current_demand = sum(sum(bins) for bins in updated_waiting_bins[destination_id].values())
                dest_future_demand = destination_vertiport.get_total_expected_pax_arrival_count()
                dest_total_demand = dest_current_demand + dest_future_demand
                dest_aircraft_count = system_manager.get_num_aircraft_at_vertiport(vertiport_id=destination_id)
                remaining_demand[destination_id] = max(0, dest_total_demand - dest_aircraft_count * seat_capacity)

            # Check if this repositioning flight is needed
            if remaining_demand[destination_id] <= 0:
                continue  # Skip rewarding this repositioning flight as it's not needed

            # Update repositioning count
            repositioning_count[destination_id] = repositioning_count.get(destination_id, 0) + 1

            # Calculate urgency score for remaining waiting passengers at destination
            dest_waiting_bins = updated_waiting_bins[destination_id][origin_vertiport_id]
            od_urgency = self._calculate_urgency_score(dest_waiting_bins)

            # Calculate demand satisfaction rate at destination
            demand_satisfaction_rate = min(1, (dest_aircraft_count * seat_capacity) / max(1, dest_total_demand))

            # Calculate repositioning score
            reposition_score = (
                (1 - demand_satisfaction_rate) * 0.6 +  # Demand satisfaction at destination
                od_urgency * 0.4  # Urgency for remaining waiting passengers at destination
            )

            # Apply repositioning reward
            trip_length = self._lookup_trip_length(origin_id=origin_vertiport_id, destination_id=destination_id)

            repositioning_reward += reward_param * trip_length * reposition_score

            # Reduce remaining demand for this destination
            remaining_demand[destination_id] = max(0, remaining_demand[destination_id] - seat_capacity)

        if repositioning_reward != 0:
            self.logger.debug(f"Demand-based repositioning reward: {round(repositioning_reward, 2)}. Repositioning count: {repositioning_count}. Raw repostioning score: {reposition_score}")

        return repositioning_reward
    
    def _calculate_urgency_score(self, waiting_bins):
        """
        Calculate an urgency score based on waiting time bins.
        Higher scores for bins with longer waiting times.
        """
        total_score = 0
        total_passengers = 0
        for i, bin_count in enumerate(waiting_bins):
            bin_score = bin_count * (i + 1)  # More weight to later bins
            total_score += bin_score
            total_passengers += bin_count
        return total_score / max(1, total_passengers)  # Normalize by total passengers    

    def _calculate_updated_waiting_bins(self, actions: List[int], dest_departing_pax_pair: Dict) -> Dict:
        """
        Calculate updated waiting bins for all vertiports after considering all flights.
        """
        updated_waiting_bins = {
            v_id: {
                o_id: self.sim_setup.vertiports[v_id].get_pax_waiting_time_bins()[o_id].copy()
                for o_id in self.sim_setup.vertiports if o_id != v_id
            } 
            for v_id in self.sim_setup.vertiports
        }
        
        # Update waiting bins based on departing passengers
        for (origin_id, destination_id), passengers_to_serve in dest_departing_pax_pair.items():
            bins = updated_waiting_bins[origin_id][destination_id]
            
            # Remove passengers from bins, starting from the highest bin
            for i in range(len(bins) - 1, -1, -1):
                if passengers_to_serve >= bins[i]:
                    passengers_to_serve -= bins[i]
                    bins[i] = 0
                else:
                    bins[i] -= passengers_to_serve
                    passengers_to_serve = 0
                    break
        
        return updated_waiting_bins
    


    def simplified_repositioning_reward(self, actions: List[int]) -> float:
        """
        Calculates a simplified reward for empty repositioning flights.
        
        Args:
        actions (List[int]): A list of action indices representing current actions for each aircraft.
        
        Returns:
        float: The total repositioning reward.
        """
        repositioning_reward = 0
        seat_capacity = self.config['aircraft_params']['pax']
        vertiports = self.sim_setup.vertiports
        system_manager = self.sim_setup.system_manager
        reward_param = self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']
        
        # Get departing passengers for all flights
        dest_departing_pax_pair = self._get_destination_departing_pax_pair(actions)
        
        # Identify repositioning flights (empty flights)
        repositioning_flights = [
            (aircraft_id, self._get_destination_vertiport_id(aircraft_id, actions))
            for aircraft_id, action in enumerate(actions)
            if action < self.sim_setup.num_vertiports() and self._is_empty_flight(aircraft_id, actions, dest_departing_pax_pair)
        ]

        for aircraft_id, destination_id in repositioning_flights:
            origin_vertiport_id = system_manager.aircraft_agents[aircraft_id].current_vertiport_id
            origin_vertiport = vertiports[origin_vertiport_id]
            destination_vertiport = vertiports[destination_id]

            # Calculate demand imbalance
            origin_demand = origin_vertiport.get_total_expected_pax_arrival_count() + sum(sum(bins) for bins in origin_vertiport.get_pax_waiting_time_bins().values())
            dest_demand = destination_vertiport.get_total_expected_pax_arrival_count() + sum(sum(bins) for bins in destination_vertiport.get_pax_waiting_time_bins().values())
            
            origin_supply = system_manager.get_num_aircraft_at_vertiport(vertiport_id=origin_vertiport_id) * seat_capacity
            dest_supply = system_manager.get_num_aircraft_at_vertiport(vertiport_id=destination_id) * seat_capacity

            origin_imbalance = max(0, origin_demand - origin_supply) / max(1, origin_demand)
            dest_imbalance = max(0, dest_demand - dest_supply) / max(1, dest_demand)

            # Only reward if destination imbalance is greater than origin imbalance
            if dest_imbalance > origin_imbalance:
                imbalance_difference = dest_imbalance - origin_imbalance
                trip_length = self._lookup_trip_length(origin_id=origin_vertiport_id, destination_id=destination_id)
                repositioning_reward += reward_param * trip_length * imbalance_difference

        return repositioning_reward
    



    def repositioning_reward_future_demand(self, actions: List[int]) -> float:
        """
        Calculates repositioning reward based on future demand, considering timing constraints.
        
        Args:
        actions (List[int]): A list of action indices representing current actions for each aircraft.
        
        Returns:
        float: The total repositioning reward.
        """
        repositioning_reward = 0
        seat_capacity = self.config['aircraft_params']['pax']
        vertiports = self.sim_setup.vertiports
        system_manager = self.sim_setup.system_manager
        reward_param = self.config['external_optimization_params']['reward_function_parameters']['repositioning_reward']
        decision_making_interval = sec_to_min(self.config['external_optimization_params']['periodic_time_step'])
        
        # Get departing passengers for all flights
        dest_departing_pax_pair = self._get_destination_departing_pax_pair(actions)
        
        # Identify repositioning flights (empty flights)
        repositioning_flights = [
            (aircraft_id, self._get_destination_vertiport_id(aircraft_id, actions))
            for aircraft_id, action in enumerate(actions)
            if action < self.sim_setup.num_vertiports() and self._is_empty_flight(aircraft_id, actions, dest_departing_pax_pair)
        ]
        
        for aircraft_id, destination_id in repositioning_flights:
            origin_vertiport_id = system_manager.aircraft_agents[aircraft_id].current_vertiport_id
            origin_vertiport = vertiports[origin_vertiport_id]
            destination_vertiport = vertiports[destination_id]
            
            # Time until aircraft is available at destination
            flight_duration = system_manager.get_average_flight_time(flight_direction=f"{origin_vertiport_id}_{destination_id}")
            
            # **Assess Future Demand at Destination Vertiport**
            future_demand = destination_vertiport.get_total_expected_pax_arrival_count()
            
            # **Assess Supply at Destination Vertiport**
            future_supply = system_manager.get_num_aircraft_at_vertiport(vertiport_id=destination_id)
            
            # **Calculate Potential Spill at Destination**
            potential_spill = max(0, future_demand - (future_supply * seat_capacity))
            
            # **Assess Impact on Origin Vertiport**
            # Ensure repositioning doesn't cause spill at the origin
            origin_future_demand = origin_vertiport.get_total_expected_pax_arrival_count()
            origin_future_supply = (system_manager.get_num_aircraft_at_vertiport(vertiport_id=origin_vertiport_id) - 1)
            
            origin_potential_spill = max(0, origin_future_demand - (origin_future_supply * seat_capacity))
            
            # **Only consider repositioning if it reduces net spill**
            net_spill_reduction = potential_spill - origin_potential_spill
            
            if net_spill_reduction > 0:
                # **Calculate Reward**
                repositioning_reward += reward_param * net_spill_reduction
            else:
                # Repositioning doesn't reduce net spill; no reward
                continue
        
        return repositioning_reward
