from .base_charging_strategy import BaseChargingStrategy
from ...utils.units import ms_to_sec, sec_to_ms
from ...utils.helpers import check_magnitude_order, miliseconds_to_hms, careful_round
from ..aircraft import AircraftStatus
import numpy as np
import heapq

class RLChargingStrategy(BaseChargingStrategy):
    def __init__(self, soc_increment_per_charge_event=10, charge_time_per_charge_event=600, soc_reward_threshold=80):
        self.soc_increment_per_charge_event = soc_increment_per_charge_event
        self.charge_time_per_charge_event = sec_to_ms(charge_time_per_charge_event)
        self.soc_reward_threshold = soc_reward_threshold

    def get_charging_time(self, aircraft):
        """
        Get charging time - Two options: Constant charging time or draw a number from distribution
        """
        if self.soc_increment_per_charge_event:
            init_soc = round(aircraft.soc)
            target_soc = init_soc + self.soc_increment_per_charge_event
            charge_time = self.get_charge_time_from_soc(aircraft=aircraft, init_soc=init_soc, target_soc=target_soc)
            # charge_time = sec_to_ms(5*60)

            aircraft.event_saver.save_aircraft_charging_time(vertiport_id=aircraft.current_vertiport_id,
                                                        charging_time=charge_time)
            aircraft.logger.debug(f"Started Charging: Aircraft {aircraft.tail_number} will charge for {round(ms_to_sec(charge_time)/60,2)} mins at {aircraft.current_vertiport_id}. init_soc: {round(aircraft.soc, 2)}")
            
            return charge_time
        elif self.charge_time_per_charge_event:
            return self.charge_time_per_charge_event
        else:
            raise ValueError("Charging time is not defined.")

    def get_charge_time_from_soc(self, aircraft: object, init_soc:int, target_soc: int) -> (float, float) :
        """
        Calculates the charge time required to reach the target soc level.

        Parameters
        ----------
        init_soc: int
            Initial soc level of the aircraft.
        target_soc: int
            Target soc level of the aircraft.

        Returns
        -------
        float
            Charge time required to reach the target soc level.
        """
        if check_magnitude_order(init_soc, target_soc):
            df = aircraft.system_manager.aircraft_battery_models
            init_soc_time = df[df.soc == init_soc]['time_sec'].values[0]
            target_soc_time = df[df.soc == target_soc]['time_sec'].values[0]
            return sec_to_ms(target_soc_time - init_soc_time)
        else:
            aircraft.logger.error(f"Final soc level ({target_soc}) is greater than or equal to 100).")
            print(f"Charging truncation: Aircraft {aircraft.tail_number} at {aircraft.current_vertiport_id} will not charge. Initial soc: {init_soc}. Final soc level ({target_soc}). At time {aircraft.env.now}.")
            aircraft.system_manager.truncation_penalty += 1
            aircraft.system_manager.trigger_truncation_event(event_name="overcharge_truncation_event", id=aircraft.tail_number)
            return 0

    def charge_aircraft(self, aircraft: object, parking_space: object, shared_charger: bool = False) -> None:
        """
        If the charger configuration is fixed and share-limited, all of the chargers that a parking pad has access
        should be requested. We use chargers as a resource instead of simpy.Store or simpy.Filterstore because it's
        easier to follow which resource is being used by which aircraft.
        """
        aircraft.status = AircraftStatus.CHARGE
        old_soc = aircraft.soc        

        # If the chargers are shared, request all of the chargers that the parking pad has access to.
        if shared_charger:
            used_charger_resource, selected_charging_request = self.handle_shared_charger(aircraft=aircraft,
                                                                                          parking_space=parking_space)
        else:
            used_charger_resource, selected_charging_request = self.request_charger(aircraft=aircraft)
            yield selected_charging_request

        self.start_charging_process(aircraft=aircraft)
        if not aircraft.charged_during_turnaround:
            idle_time = aircraft.env.now - aircraft.arrival_time
        else:
            idle_time = aircraft.env.now - aircraft.charging_end_time
        aircraft.idle_time += idle_time
        aircraft.save_process_time(event='idle', process_time=idle_time)
        # aircraft.logger.debug(f'Saved idle time of {miliseconds_to_hms(idle_time)} between arrival and charging for aircraft {aircraft.tail_number} at {aircraft.location}.')

        aircraft.detailed_status = 'charging'
        charge_time = self.get_charging_time(aircraft=aircraft)

        finish_time = aircraft.env.now + charge_time

        # Add charging process to the heap to keep track of the next available time for the chargers.
        self.add_to_active_charging_process(aircraft=aircraft, finish_time=finish_time)

        yield aircraft.env.timeout(round(charge_time))
        aircraft.save_process_time(event='charge', process_time=charge_time)
        aircraft.detailed_status = 'idle'
        self.end_charging_process(aircraft=aircraft)

        # Remove the charging process from the heap since the charging is completed.
        self.remove_from_active_charging_process(aircraft=aircraft, finish_time=finish_time)

        # print(f"Finished Charging: Aircraft {aircraft.tail_number} at {aircraft.current_vertiport_id}. At time {miliseconds_to_hms(aircraft.env.now)}.")
        self.soc_charge_update(aircraft=aircraft, initial_soc=old_soc, charge_time=charge_time)

        # Release the charger resource that is used by the aircraft and the charging request
        self.release_charger_resource(used_charger_resource, selected_charging_request)
        # Set the charged_during_turnaround to True to avoid charging the aircraft again during the turnaround
        aircraft.charged_during_turnaround = True
        aircraft.logger.debug(f"Finished Charging: Aircraft {aircraft.tail_number} at {aircraft.current_vertiport_id}. Previous SoC: {round(old_soc, 2)}, New SoC: {careful_round(aircraft.soc, 2)}")

    def add_to_active_charging_process(self, aircraft: object, finish_time: float) -> None:
        """
        Adds the finish, start and duration aircraft to the active charging process of the vertiport.
        """
        vertiport = aircraft.system_manager.vertiports[aircraft.current_vertiport_id]
        # Push the finish time to the heap
        heapq.heappush(vertiport.active_charging_processes, finish_time)

    def remove_from_active_charging_process(self, aircraft: object, finish_time: float) -> None:
        """
        Removes the finish, start and duration aircraft from the active charging process of the vertiport.
        """
        vertiport = aircraft.system_manager.vertiports[aircraft.current_vertiport_id]
        vertiport.active_charging_processes.remove(finish_time)
        heapq.heapify(vertiport.active_charging_processes)