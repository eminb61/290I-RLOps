from typing import Dict
from ..aircraft.aircraft import AircraftStatus
from ..utils.helpers import current_and_lookahead_pax_count
import numpy as np
from ..utils.units import sec_to_ms, ms_to_hr, sec_to_min, ms_to_min
from collections import defaultdict
import itertools

class ActionMask:
    def __init__(self, sim_setup, config: Dict, num_aircraft: int):
        self.sim_setup = sim_setup
        self.config = config
    # Pre-cache frequently accessed attributes for faster access
        self.system_manager = self.sim_setup.system_manager
        self.aircraft_agents = self.system_manager.aircraft_agents
        self.num_aircraft = num_aircraft
        self.decision_making_interval = sec_to_ms(config['external_optimization_params']['periodic_time_step'])   
        self.vertiports = self.sim_setup.vertiports
        self.central_hub_id = self.config['network_and_demand_params']['hub_vertiport_id']
        self.seat_capacity = self.sim_setup.aircraft_params['pax']
        self.vertiport_ids = self.sim_setup.vertiport_ids
        self.num_vertiports = len(self.vertiport_ids)
        self.vertiport_id_to_index_map = self.sim_setup.vertiport_id_to_index_map
        self.vertiport_index_to_id_map = self.sim_setup.vertiport_index_to_id_map
        self.min_reserve_soc = self.config['aircraft_params']['min_reserve_soc']
        self.battery_capacity = self.config['aircraft_params']['battery_capacity']
        self.num_waiting_time_bins = int(self.config['sim_params']['max_passenger_waiting_time'] / self.config['external_optimization_params']['periodic_time_step'])
        # Cache SOC and charge increment configurations
        self.soc_increment_per_charge_event = self.config['external_optimization_params'].get('soc_increment_per_charge_event')
        self.charge_time_per_charge_event = self.config['external_optimization_params'].get('charge_time_per_charge_event')

    def get_action_mask(self, initial_state=False, final_state=False):
        """
        Create the action mask for the current state of the simulation.
        [Fly: list, Charge: int, Do nothing: int]
        
        Returns:
            List[int]: A flattened list representing the mask where 1 allows the action and 0 masks it.
        """
        if self.config['network_and_demand_params']['star_network']:
            return self.star_network_action_mask(initial_state, final_state)
        else:
            return self.vec_non_star_network_action_mask(initial_state, final_state)
        
    def vec_non_star_network_action_mask(self, initial_state=False, final_state=False):
        # Initialize the mask
        mask = np.zeros((self.num_aircraft, self.num_vertiports + 2), dtype=int)

        if initial_state:
            mask[:, self.num_vertiports:self.num_vertiports + 2] = 1
            return mask.flatten().tolist()

        if final_state:
            mask[:, self.num_vertiports + 1] = 1
            return mask.flatten().tolist()

        # Precompute flight_soc_required
        flight_soc_required_map = {
            flight_direction: values['average'] + self.min_reserve_soc
            for flight_direction, values in self.sim_setup.event_saver.average_energy_consumption_per_od_pair.items()
        }
        min_soc_required = min(flight_soc_required_map.values())

        # Extract aircraft data
        aircraft_status = np.array([a.status for a in self.aircraft_agents.values()])
        aircraft_soc = np.array([a.soc for a in self.aircraft_agents.values()])

        # Initialize aircraft_location with -1
        aircraft_location = np.full(self.num_aircraft, -1, dtype=int)

        # Identify idling aircraft
        idle_mask = (aircraft_status == AircraftStatus.IDLE)
        idle_indices = np.where(idle_mask)[0]

        # Set aircraft_location for idling aircraft
        aircraft_location[idle_indices] = np.array([
            self.vertiport_id_to_index_map[self.aircraft_agents[idx].current_vertiport_id]
            for idx in idle_indices
        ])

        # Vectorized SOC increment calculation
        if self.soc_increment_per_charge_event is not None:
            soc_increment = np.full(self.num_aircraft, self.soc_increment_per_charge_event)
        elif self.charge_time_per_charge_event is not None:
            soc_increment = np.array([
                aircraft.charging_strategy.calc_soc_from_charge_time(
                    charge_time=sec_to_ms(self.charge_time_per_charge_event),
                    initial_soc=aircraft.soc,
                    df=self.system_manager.aircraft_battery_models
                ) - aircraft.soc
                for aircraft in self.aircraft_agents.values()
            ])
        else:
            raise ValueError("Invalid config for soc_increment_per_charge_event or charge_time_per_charge_event. One must be specified.")

        # Determine action capabilities
        can_do_any = idle_mask & (aircraft_soc <= 95 - soc_increment) & (aircraft_soc >= min_soc_required)
        can_fly_and_do_nothing = idle_mask & (aircraft_soc > 95 - soc_increment)
        can_only_charge = idle_mask & (aircraft_soc < min_soc_required)

        # Define fly_mask
        fly_mask = can_do_any | can_fly_and_do_nothing

        # Set flight actions for aircraft that can fly
        mask[fly_mask, :self.num_vertiports] = 1

        # Mask out flight to current location for aircraft with valid location
        valid_location_mask = aircraft_location != -1
        mask[valid_location_mask, aircraft_location[valid_location_mask]] = 0

        # Set non-flight actions
        mask[can_do_any | can_only_charge, self.num_vertiports] = 1  # Charge
        mask[fly_mask, self.num_vertiports + 1] = 1  # Do Nothing

        # For non-idling aircraft, only allow "Do Nothing"
        non_idle_mask = ~idle_mask
        mask[non_idle_mask, :] = 0
        mask[non_idle_mask, self.num_vertiports + 1] = 1  # Do Nothing

        # Apply SOC constraints per aircraft per OD pair
        for idx in range(self.num_aircraft):
            if not idle_mask[idx]:
                continue  # Skip non-idling aircraft
            
            origin_index = aircraft_location[idx]
            for dest_index in range(self.num_vertiports):
                if origin_index == dest_index:
                    mask[idx, dest_index] = 0  # Cannot fly to the same vertiport
                    continue

                od_pair = f"{self.vertiport_ids[origin_index]}_{self.vertiport_ids[dest_index]}"
                soc_required = flight_soc_required_map.get(od_pair, 0)
                if aircraft_soc[idx] < soc_required:
                    mask[idx, dest_index] = 0  # Mask action due to insufficient SoC        

        # Process enforced flights if required
        if "waiting_time_bins" in self.config['sim_params']['simulation_states']['vertiport_states'] and \
                self.config['sim_params']['algorithm'] not in ['RandomPolicy', 'VertiSimHeuristics']:
            mask = self.process_enforced_flights(mask, aircraft_soc, aircraft_location, flight_soc_required_map, fly_mask)

        return mask.flatten().tolist()

        
    def non_star_network_action_mask(self, initial_state=False, final_state=False):

        # Initialize the mask as a NumPy array for efficient operations
        # Each aircraft has (num_vertiports + 2) actions
        mask = np.zeros((self.num_aircraft, self.num_vertiports + 2), dtype=int)

        if initial_state:
            # All flight actions are masked (0), Charge and Do Nothing are allowed (1)
            mask[:, self.num_vertiports:self.num_vertiports+2] = 1
            return mask.flatten().tolist()
        
        if final_state:
            # Only Do Nothing is allowed (1), others are masked (0)
            mask[:, self.num_vertiports+1] = 1
            return mask.flatten().tolist()
        
        # Precompute frequently accessed attributes
        flight_soc_required_map = {flight_direction: values['average'] + self.min_reserve_soc for flight_direction, values \
                                   in self.sim_setup.event_saver.average_energy_consumption_per_od_pair.items()}
        
        min_soc_required = min(flight_soc_required_map.values())
        
        # Dictionaries to track enforced OD pairs and aircraft SoC
        enforced_od_pairs = set()
        aircraft_soc_map = {}  # {vertiport_id: [(aircraft_id, soc), ...]}

        # Check if there's any demand at the vertiports
        # no_demand_in_system = not any(current_and_lookahead_pax_count(vertiport) for vertiport in self.system_manager.vertiports.values()) 

        # First pass: Determine initial masks and populate aircraft_soc_map
        for idx, (aircraft_id, aircraft) in enumerate(self.aircraft_agents.items()):
            current_vertiport_id = aircraft.current_vertiport_id
            current_vertiport = self.vertiports[current_vertiport_id]
            
            # Determine SOC increment
            if self.soc_increment_per_charge_event is not None:
                soc_increment = self.soc_increment_per_charge_event
            elif self.charge_time_per_charge_event is not None:
                charge_lookup_table = self.system_manager.aircraft_battery_models
                new_soc = aircraft.charging_strategy.calc_soc_from_charge_time(
                    charge_time=sec_to_ms(self.charge_time_per_charge_event),
                    initial_soc=aircraft.soc,
                    df=charge_lookup_table
                )
                soc_increment = new_soc - aircraft.soc
            else:
                raise ValueError("Invalid config for soc_increment_per_charge_event or charge_time_per_charge_event. One must be specified.")

            # Define conditions
            can_do_any = (
                aircraft.status == AircraftStatus.IDLE and 
                aircraft.soc <= 95 - soc_increment and 
                aircraft.soc > self.min_reserve_soc
            )
            
            can_fly_and_do_nothing = (
                aircraft.status == AircraftStatus.IDLE and
                aircraft.soc > 95 - soc_increment
            )
            
            # Initialize flight_mask and non_flight_actions
            if can_do_any:
                # Allow flying to any vertiport except the current one
                flight_mask = np.ones(self.num_vertiports, dtype=int)
                current_index = self.vertiport_id_to_index_map[current_vertiport_id]
                flight_mask[current_index] = 0  # Mask current vertiport
                non_flight_actions = [1, 1]  # Allow Charge and Do Nothing
                # Populate aircraft_soc_map for potential enforced flights
                aircraft_soc_map.setdefault(current_vertiport_id, []).append((idx, aircraft.soc))
            
            # If the aircraft is idle and its SoC is below the minimum reserve SoC + the minimum value in flight_soc_required dict, it can only charge

            elif aircraft.status == AircraftStatus.IDLE and aircraft.soc <= min_soc_required:
                # Can only Charge
                flight_mask = np.zeros(self.num_vertiports, dtype=int)
                non_flight_actions = [1, 0]  # Allow Charge only
            elif can_fly_and_do_nothing:
                # Can Fly and Do Nothing
                flight_mask = np.ones(self.num_vertiports, dtype=int)
                current_index = self.vertiport_id_to_index_map[current_vertiport_id]
                flight_mask[current_index] = 0  # Mask current vertiport
                non_flight_actions = [0, 1]  # Allow Do Nothing only
                # Populate aircraft_soc_map for potential enforced flights
                aircraft_soc_map.setdefault(current_vertiport_id, []).append((idx, aircraft.soc))
            elif aircraft.status in [AircraftStatus.CHARGE, AircraftStatus.FLY, AircraftStatus.HOLD]:
                # Can only Do Nothing
                flight_mask = np.zeros(self.num_vertiports, dtype=int)
                non_flight_actions = [0, 1]  # Allow Do Nothing only
            else:
                raise ValueError(f"Invalid aircraft status: {aircraft.status}")

            # Disable flight actions to destinations that the aircraft cannot fly to based on SoC
            for dest_vertiport_id in self.vertiport_ids:
                if dest_vertiport_id == current_vertiport_id:
                    continue
                dest_index = self.vertiport_id_to_index_map[dest_vertiport_id]
                od_pair = f"{current_vertiport_id}_{dest_vertiport_id}"
                flight_soc_required = flight_soc_required_map.get(od_pair, 0)
                if flight_soc_required > aircraft.soc:
                    flight_mask[dest_index] = 0  # Mask the action
            
            # Update the mask for the current aircraft
            mask[idx, :self.num_vertiports] = flight_mask
            mask[idx, self.num_vertiports:self.num_vertiports+2] = non_flight_actions


        # If waiting_time_bins is in the states of the simulation, use it to determine enforced flights
        if "waiting_time_bins" in self.config['sim_params']['simulation_states']['vertiport_states'] and self.config['sim_params']['algorithm'] not in ['RandomPolicy', 'VertiSimHeuristics']:            
            # Second pass: Process enforced flights and update masks
            for current_vertiport_id, aircraft_list in aircraft_soc_map.items():
                current_vertiport = self.vertiports[current_vertiport_id]
                
                for dest_vertiport_id in self.vertiport_ids:
                    if dest_vertiport_id == current_vertiport_id:
                        continue
                    
                    total_pax_count_per_dest = current_vertiport.get_waiting_passenger_count()
                    # passengers_in_last_bin = waiting_time_bins[dest_vertiport_id][-1]
                    if (self.will_passengers_spill(vertiport=current_vertiport, 
                                                   dest_vertiport_id=dest_vertiport_id, 
                                                   spill_limit=0) \
                                                    and (current_vertiport_id, dest_vertiport_id) not in enforced_od_pairs) \
                        or (total_pax_count_per_dest[dest_vertiport_id] >= self.seat_capacity and (current_vertiport_id, dest_vertiport_id) not in enforced_od_pairs):
                        dest_index = self.vertiport_id_to_index_map.get(dest_vertiport_id)
                        
                        # Find the aircraft with the highest SoC that can make the flight
                        suitable_aircraft = [
                            (idx, soc) for idx, soc in aircraft_list 
                            if soc >= flight_soc_required_map.get(f"{current_vertiport_id}_{dest_vertiport_id}", 0)
                        ]
                        
                        if suitable_aircraft:
                            # Select the aircraft with the highest SoC
                            selected_idx, _ = max(suitable_aircraft, key=lambda x: x[1])
                            # Enforce the flight by allowing only the specific destination
                            enforced_od_pairs.add((current_vertiport_id, dest_vertiport_id))
                            
                            # Update the mask: allow only the enforced destination and mask all other actions
                            enforced_flight_mask = np.zeros(self.num_vertiports, dtype=int)
                            enforced_flight_mask[dest_index] = 1
                            mask[selected_idx, :self.num_vertiports] = enforced_flight_mask
                            mask[selected_idx, self.num_vertiports:self.num_vertiports+2] = [0, 0]  # Disable Charge and Do Nothing
                            
                            # Remove the aircraft from the list to prevent multiple assignments
                            aircraft_list.remove((selected_idx, _))
        
        # Convert the mask to a flattened list
        return mask.flatten().tolist()

    def dep_star_network_action_mask(self, initial_state=False, final_state=False):
        # Initialize the mask as a NumPy array for efficient operations
        # Each aircraft has (num_vertiports + 2) actions
        mask = np.zeros((self.num_aircraft, self.num_vertiports + 2), dtype=int)

        if initial_state:
            # All flight actions are masked (0), Charge and Do Nothing are allowed (1)
            mask[:, self.num_vertiports:self.num_vertiports+2] = 1
            return mask.flatten().tolist()
        
        if final_state:
            # Only Do Nothing is allowed (1), others are masked (0)
            mask[:, self.num_vertiports+1] = 1
            return mask.flatten().tolist()
        
        # Precompute frequently accessed attributes
        flight_soc_required_map = {flight_direction: values['average'] + self.min_reserve_soc for flight_direction, values \
                                   in self.sim_setup.event_saver.average_energy_consumption_per_od_pair.items()}
        
        min_soc_required = min(flight_soc_required_map.values())

        # Dictionaries to track enforced OD pairs and aircraft SoC
        enforced_od_pairs = set()
        aircraft_soc_map = {}  # {vertiport_id: [(aircraft_id, soc), ...]}

        # Calculate aircraft supply and demand at each vertiport
        aircraft_supply = {vertiport_id: 0 for vertiport_id in self.vertiport_ids}
        passenger_demand = {vertiport_id: 0 for vertiport_id in self.vertiport_ids}
        
        for aircraft in self.aircraft_agents.values():
            if aircraft.status == AircraftStatus.IDLE:
                aircraft_supply[aircraft.current_vertiport_id] += self.seat_capacity
        
        for vertiport in self.vertiports.values():
            passenger_demand[vertiport.vertiport_id] = current_and_lookahead_pax_count(vertiport)
        
        # Check if there's any demand at the vertiports
        # no_demand_in_system = not any(current_and_lookahead_pax_count(vertiport) for vertiport in self.system_manager.vertiports.values()) 

        # First pass: Determine initial masks and populate aircraft_soc_map
        for idx, (aircraft_id, aircraft) in enumerate(self.aircraft_agents.items()):
            current_vertiport_id = aircraft.current_vertiport_id

            # Determine SOC increment
            if self.soc_increment_per_charge_event is not None:
                soc_increment = self.soc_increment_per_charge_event
            elif self.charge_time_per_charge_event is not None:
                charge_lookup_table = self.system_manager.aircraft_battery_models
                new_soc = aircraft.charging_strategy.calc_soc_from_charge_time(
                    charge_time=sec_to_ms(self.charge_time_per_charge_event),
                    initial_soc=aircraft.soc,
                    df=charge_lookup_table
                )
                soc_increment = new_soc - aircraft.soc
            else:
                raise ValueError("Invalid config for soc_increment_per_charge_event or charge_time_per_charge_event. One must be specified.")

            # Define conditions
            can_do_any = (
                aircraft.status == AircraftStatus.IDLE and 
                aircraft.soc <= 95 - soc_increment and 
                aircraft.soc > self.min_reserve_soc
            )
            
            can_fly_and_do_nothing = (
                aircraft.status == AircraftStatus.IDLE and
                aircraft.soc > 95 - soc_increment
            )
            
            # Initialize flight_mask and non_flight_actions
            if can_do_any:
                # Allow flying to any vertiport except the current one if it's the central hub
                if current_vertiport_id == self.central_hub_id:
                    flight_mask = np.ones(self.num_vertiports, dtype=int)
                    flight_mask[self.vertiport_id_to_index_map[self.central_hub_id]] = 0
                # Allow flying to the central hub and consider repositioning
                else:
                    flight_mask = np.zeros(self.num_vertiports, dtype=int)
                    flight_mask[self.vertiport_id_to_index_map[self.central_hub_id]] = 1
                    # Check for repositioning needs
                    for dest_vertiport_id in self.sim_setup.vertiport_ids:
                        if dest_vertiport_id != current_vertiport_id and dest_vertiport_id != self.central_hub_id:
                            if self.check_repositioning_need(dest_vertiport_id):
                                flight_mask[self.vertiport_id_to_index_map[dest_vertiport_id]] = 1

                non_flight_actions = [1, 1]  # Allow Charge and Do Nothing
                # Populate aircraft_soc_map for potential enforced flights
                aircraft_soc_map.setdefault(current_vertiport_id, []).append((idx, aircraft.soc))
            
            # If the aircraft is idle and its SoC is below the minimum reserve SoC + the minimum value in flight_soc_required dict, it can only charge

            elif aircraft.status == AircraftStatus.IDLE and aircraft.soc <= min_soc_required:
                # Can only Charge
                flight_mask = np.zeros(self.num_vertiports, dtype=int)
                non_flight_actions = [1, 0]  # Allow Charge only
            elif can_fly_and_do_nothing:
                # Can Fly and Do Nothing. If the aircraft is at the central hub, it can fly to any vertiport except the hub
                if current_vertiport_id == self.central_hub_id:
                    flight_mask = np.ones(self.num_vertiports, dtype=int)
                    flight_mask[self.vertiport_id_to_index_map[self.central_hub_id]] = 0
                # Can fly to the central hub and consider repositioning
                else:
                    flight_mask = np.zeros(self.num_vertiports, dtype=int)
                    flight_mask[self.vertiport_id_to_index_map[self.central_hub_id]] = 1
                    # Check for repositioning needs
                    for dest_vertiport_id in self.sim_setup.vertiport_ids:
                        if dest_vertiport_id != current_vertiport_id and dest_vertiport_id != self.central_hub_id:
                            if self.check_repositioning_need(dest_vertiport_id):
                                flight_mask[self.vertiport_id_to_index_map[dest_vertiport_id]] = 1
                non_flight_actions = [0, 1]  # Allow Do Nothing only
                # Populate aircraft_soc_map for potential enforced flights
                aircraft_soc_map.setdefault(current_vertiport_id, []).append((idx, aircraft.soc))

            elif aircraft.status in [AircraftStatus.CHARGE, AircraftStatus.FLY, AircraftStatus.HOLD]:
                # Can only Do Nothing
                flight_mask = np.zeros(self.num_vertiports, dtype=int)
                non_flight_actions = [0, 1]  # Allow Do Nothing only
            else:
                raise ValueError(f"Invalid aircraft status: {aircraft.status}")

            # Disable flight actions to destinations that the aircraft cannot fly to based on SoC
            for dest_vertiport_id in self.vertiport_ids:
                if dest_vertiport_id == current_vertiport_id:
                    continue
                dest_index = self.vertiport_id_to_index_map[dest_vertiport_id]
                od_pair = f"{current_vertiport_id}_{dest_vertiport_id}"
                flight_soc_required = flight_soc_required_map.get(od_pair, 0)
                if flight_soc_required > aircraft.soc:
                    flight_mask[dest_index] = 0  # Mask the action
            
            # Update the mask for the current aircraft
            mask[idx, :self.num_vertiports] = flight_mask
            mask[idx, self.num_vertiports:self.num_vertiports+2] = non_flight_actions


        # If waiting_time_bins is in the states of the simulation, use it to determine enforced flights
        if "waiting_time_bins" in self.config['sim_params']['simulation_states']['vertiport_states'] and self.config['sim_params']['algorithm'] not in ['RandomPolicy', 'VertiSimHeuristics']:            
            # Second pass: Process enforced flights and update masks
            for current_vertiport_id, aircraft_list in aircraft_soc_map.items():
                current_vertiport = self.vertiports[current_vertiport_id]
                
                for dest_vertiport_id in self.vertiport_ids:
                    if dest_vertiport_id == current_vertiport_id:
                        continue
                    
                    total_pax_count = current_vertiport.get_total_waiting_passenger_count()
                    # passengers_in_last_bin = waiting_time_bins[dest_vertiport_id][-1]
                    if (self.will_passengers_spill(vertiport=current_vertiport, 
                                                   dest_vertiport_id=dest_vertiport_id, 
                                                   spill_limit=0) \
                                                    and (current_vertiport_id, dest_vertiport_id) not in enforced_od_pairs) \
                        or (total_pax_count >= self.seat_capacity and (current_vertiport_id, dest_vertiport_id) not in enforced_od_pairs):
                        dest_index = self.vertiport_id_to_index_map.get(dest_vertiport_id)
                        
                        # Find the aircraft with the highest SoC that can make the flight
                        suitable_aircraft = [
                            (idx, soc) for idx, soc in aircraft_list 
                            if soc >= flight_soc_required_map.get(f"{current_vertiport_id}_{dest_vertiport_id}", 0)
                        ]
                        
                        if suitable_aircraft:
                            # Select the aircraft with the highest SoC
                            selected_idx, _ = max(suitable_aircraft, key=lambda x: x[1])
                            # Enforce the flight by allowing only the specific destination
                            enforced_od_pairs.add((current_vertiport_id, dest_vertiport_id))
                            
                            # Update the mask: allow only the enforced destination and mask all other actions
                            enforced_flight_mask = np.zeros(self.num_vertiports, dtype=int)
                            enforced_flight_mask[dest_index] = 1
                            mask[selected_idx, :self.num_vertiports] = enforced_flight_mask
                            mask[selected_idx, self.num_vertiports:self.num_vertiports+2] = [0, 0]  # Disable Charge and Do Nothing
                            
                            # Remove the aircraft from the list to prevent multiple assignments
                            aircraft_list.remove((selected_idx, _))
        
        # Convert the mask to a flattened list
        return mask.flatten().tolist()
    
    def check_repositioning_need(self, vertiport_id):
        """
        Check if repositioning is needed at the vertiport. We only consider the future demand because the repositioning
        aircfaft will not be able to serve the current demand.
        """
        seat_supply = self.vertiport_aircraft_supply(vertiport_id) * self.seat_capacity
        passenger_demand = self.vertiports[vertiport_id].get_total_expected_pax_arrival_count()
        return seat_supply < passenger_demand

    def vertiport_aircraft_supply(self, vertiport_id):
        """
        Get the total supply of aircraft at a vertiport.
        """
        num_aircraft = self.system_manager.get_num_aircraft_at_vertiport(vertiport_id)
        outgoing_aircraft_supply = self._outgoing_aircraft_supply(vertiport_id)
        incoming_aircraft_supply = self._incoming_aircraft_supply(vertiport_id)
        return max(0, num_aircraft + outgoing_aircraft_supply + incoming_aircraft_supply)

    def _outgoing_aircraft_supply(self, origin_vertiport_id: int) -> int:
        """
        Get the number of aircraft flying from  vertiport.
        """
        return sum(1 for aircraft_id, _ in self.system_manager.aircraft_agents.items()
                    if self.system_manager.aircraft_agents[aircraft_id].current_vertiport_id == origin_vertiport_id and 
                    self.system_manager.aircraft_agents[aircraft_id].status == AircraftStatus.FLY)
    
    def _incoming_aircraft_supply(self, destination_vertiport_id: int) -> int:
        """
        Get the number of aircraft flying to vertiport.
        """
        return sum(1 for aircraft_id, _ in self.system_manager.aircraft_agents.items()
                    if self.system_manager.aircraft_agents[aircraft_id].destination_vertiport_id == destination_vertiport_id and 
                    self.system_manager.aircraft_agents[aircraft_id].status == AircraftStatus.FLY)
        
    def will_passengers_spill(self, vertiport, dest_vertiport_id=None, spill_limit=0):
        """
        Check if passengers will spill at the vertiport in the next decision-making interval.

        Args:
            vertiport: The vertiport object.
            dest_vertiport_id (optional): The ID of the destination vertiport. If provided, check for spills to this destination only.
        
        Returns:
            bool: True if passengers will spill, False otherwise.
        """
        # Convert max passenger waiting time and decision interval to minutes
        max_waiting_time_min = sec_to_min(self.config['sim_params']['max_passenger_waiting_time'])
        decision_interval_min = ms_to_min(self.decision_making_interval)

        # Calculate the index threshold for spilling passengers
        # If a passenger's waiting time is greater than max_waiting_time_min - decision_interval_min,
        # they will spill in the next decision-making interval.
        # We find this threshold index based on num_waiting_time_bins.
        spill_time_threshold = max_waiting_time_min - decision_interval_min
        bin_size = max_waiting_time_min / self.num_waiting_time_bins
        threshold_index = int(spill_time_threshold / bin_size)

        # Get the waiting time bins from the vertiport
        waiting_time_bins = vertiport.get_pax_waiting_time_bins()

        if dest_vertiport_id is not None:
            # Get the bins for the specific destination vertiport
            bins = waiting_time_bins.get(dest_vertiport_id, [])
            spill_pax_count = sum(bins[threshold_index:])  # Count passengers in bins exceeding threshold
            return spill_pax_count > spill_limit
        else:
            # Check if any passengers at any destination will spill
            return any(
                sum(bins[threshold_index:]) > spill_limit
                for bins in waiting_time_bins.values()
            )

    def star_network_action_mask(self, initial_state=False, final_state=False):
        # Initialize the mask
        mask = np.zeros((self.num_aircraft, self.num_vertiports + 2), dtype=int)

        if initial_state:
            mask[:, self.num_vertiports:self.num_vertiports + 2] = 1
            return mask.flatten().tolist()

        if final_state:
            mask[:, self.num_vertiports + 1] = 1
            return mask.flatten().tolist()

        # Precompute flight_soc_required
        flight_soc_required_map = {
            flight_direction: values['average'] + self.min_reserve_soc
            for flight_direction, values in self.sim_setup.event_saver.average_energy_consumption_per_od_pair.items()
        }
        min_soc_required = min(flight_soc_required_map.values())

        # Extract aircraft data
        aircraft_status = np.array([a.status for a in self.aircraft_agents.values()])
        aircraft_soc = np.array([a.soc for a in self.aircraft_agents.values()])

        # Initialize aircraft_location with -1
        aircraft_location = np.full(self.num_aircraft, -1, dtype=int)

        # Identify idling aircraft
        idle_mask = (aircraft_status == AircraftStatus.IDLE)
        idle_indices = np.where(idle_mask)[0]

        # Set aircraft_location for idling aircraft
        aircraft_location[idle_indices] = np.array([
            self.vertiport_id_to_index_map[self.aircraft_agents[idx].current_vertiport_id]
            for idx in idle_indices
        ])

        # Vectorized SOC increment calculation
        if self.soc_increment_per_charge_event is not None:
            soc_increment = np.full(self.num_aircraft, self.soc_increment_per_charge_event)
        elif self.charge_time_per_charge_event is not None:
            soc_increment = np.array([
                aircraft.charging_strategy.calc_soc_from_charge_time(
                    charge_time=sec_to_ms(self.charge_time_per_charge_event),
                    initial_soc=aircraft.soc,
                    df=self.system_manager.aircraft_battery_models
                ) - aircraft.soc
                for aircraft in self.aircraft_agents.values()
            ])
        else:
            raise ValueError("Invalid config for soc_increment_per_charge_event or charge_time_per_charge_event. One must be specified.")

        # Determine action capabilities
        can_do_any = idle_mask & (aircraft_soc <= 95 - soc_increment) & (aircraft_soc >= min_soc_required)
        can_fly_and_do_nothing = idle_mask & (aircraft_soc > 95 - soc_increment)
        can_only_charge = idle_mask & (aircraft_soc < min_soc_required)

        # Define fly_mask
        fly_mask = can_do_any | can_fly_and_do_nothing

        # Set flight actions for aircraft that can fly
        mask[fly_mask, :self.num_vertiports] = 1

        # Mask out flight to current location for aircraft with valid location
        valid_location_mask = aircraft_location != -1
        mask[valid_location_mask, aircraft_location[valid_location_mask]] = 0

        # Set non-flight actions
        mask[can_do_any | can_only_charge, self.num_vertiports] = 1  # Charge
        mask[fly_mask, self.num_vertiports + 1] = 1  # Do Nothing

        # For non-idling aircraft, only allow "Do Nothing"
        non_idle_mask = ~idle_mask
        mask[non_idle_mask, :] = 0
        mask[non_idle_mask, self.num_vertiports + 1] = 1  # Do Nothing

        # Apply star network constraints
        central_hub_index = self.vertiport_id_to_index_map[self.central_hub_id]
        non_hub_mask = (aircraft_location != central_hub_index) & (aircraft_location != -1)

        # Mask out flight actions for non-hub aircraft
        mask[non_hub_mask, :self.num_vertiports] = 0

        # Allow non-hub aircraft that can fly to fly to central hub
        mask[non_hub_mask & fly_mask, central_hub_index] = 1

    # Repositioning logic for aircraft not at the central hub
        for dest in self.vertiport_ids:
            # Check if "expected_pax_arr_per_od" is in the configuration
            reposition_check_required = "expected_pax_arr_per_od" in self.config['sim_params']['simulation_states']['vertiport_states']
            if dest != self.central_hub_id and (self.check_repositioning_need(dest) if reposition_check_required else True):
                dest_index = self.vertiport_id_to_index_map[dest]

                # Exclude aircraft where origin and destination are the same
                origin_dest_different = aircraft_location != dest_index

                # Eligible aircraft:
                eligible_aircraft_mask = (
                    non_hub_mask &
                    fly_mask &
                    origin_dest_different
                )

                # Get indices of eligible aircraft
                eligible_aircraft_indices = np.where(eligible_aircraft_mask)[0]

                if len(eligible_aircraft_indices) == 0:
                    continue  # No eligible aircraft for this destination

                # Get origin vertiport IDs
                origin_indices = aircraft_location[eligible_aircraft_indices]
                origin_ids = [self.vertiport_ids[idx] for idx in origin_indices]
                dest_id = dest

                # Construct OD pairs and get SoC requirements
                od_pairs = [f"{origin}_{dest_id}" for origin in origin_ids]
                soc_requirements = np.array([
                    flight_soc_required_map.get(od_pair, np.inf)
                    for od_pair in od_pairs
                ])

                # Get SoC of the aircraft
                aircraft_socs = aircraft_soc[eligible_aircraft_indices]

                # Mask aircraft with sufficient SoC
                sufficient_soc_mask = aircraft_socs >= soc_requirements

                # Final eligible aircraft indices
                final_aircraft_indices = eligible_aircraft_indices[sufficient_soc_mask]

                # Update the mask
                mask[final_aircraft_indices, dest_index] = 1

        # Process enforced flights if required
        if "waiting_time_bins" in self.config['sim_params']['simulation_states']['vertiport_states'] and \
                self.config['sim_params']['algorithm'] not in ['RandomPolicy', 'VertiSimHeuristics']:
            mask = self.process_enforced_flights(mask, aircraft_soc, aircraft_location, flight_soc_required_map, fly_mask)

        return mask.flatten().tolist()

    def process_enforced_flights(self, mask, aircraft_soc, aircraft_location, flight_soc_required_map, fly_mask):
        enforced_od_pairs = set()
        for origin in self.vertiport_ids:
            origin_index = self.vertiport_id_to_index_map[origin]

            # Get eligible aircraft for the origin vertiport
            eligible_aircraft = np.where(fly_mask & (aircraft_location == origin_index))[0]
            # eligible_aircraft = np.where(aircraft_location == self.vertiport_id_to_index_map[origin])[0]
            if np.sum(eligible_aircraft) == 0:
                continue

            vertiport = self.vertiports[origin]
            pax_count_per_destination = vertiport.get_waiting_passenger_count()

            for dest in self.vertiport_ids:
                if origin == dest or (origin, dest) in enforced_od_pairs:
                    continue

                if self.will_passengers_spill(vertiport=vertiport, dest_vertiport_id=dest) or \
                        pax_count_per_destination[dest] >= self.seat_capacity:
                    soc_required = flight_soc_required_map.get(f"{origin}_{dest}", 0)
                    suitable_aircraft = [idx for idx in eligible_aircraft if aircraft_soc[idx] >= soc_required]

                    if len(suitable_aircraft) > 0:
                        selected_idx = max(suitable_aircraft, key=lambda idx: aircraft_soc[idx])
                        mask[selected_idx] = 0
                        mask[selected_idx, self.vertiport_id_to_index_map[dest]] = 1
                        enforced_od_pairs.add((origin, dest))
                        # Remove selected aircraft from eligible_aircraft to avoid reassigning
                        eligible_aircraft = eligible_aircraft[eligible_aircraft != selected_idx]
        return mask