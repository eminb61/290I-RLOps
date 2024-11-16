from typing import Dict
from ..utils.units import miles_to_m, ms_to_sec

def time_to_arrival_estimator(aircraft: object) -> float:
    """
    Estimates the time to arrival at the destination vertiport for the given aircraft.

    The function calculates the estimated time to arrival (ETA) based on the aircraft's current
    location and state. It handles various scenarios, including when the aircraft is at the origin
    or destination vertiport, in the air, or at an intermediate vertiport node.

    Parameters:
        aircraft (object): The aircraft for which to estimate the time to arrival.

    Returns:
        float: The estimated time to arrival in seconds.
    """
    # Return 0 if the destination is not set or if already at the destination
    if aircraft.destination_vertiport_id is None or \
       aircraft.origin_vertiport_id == aircraft.destination_vertiport_id:
        return 0

    # Retrieve or initialize the average flight time
    average_flight_time = aircraft.system_manager.get_average_flight_time(aircraft.flight_direction)
    average_flight_time_initialized = average_flight_time is not None

    if not average_flight_time_initialized:
        # Compute the initial average flight time in seconds
        average_flight_time = aircraft.system_manager.initial_flight_duration_estimate(
            aircraft.origin_vertiport_id,
            aircraft.destination_vertiport_id,
            aircraft.aircraft_params['cruise_speed']
        )
        aircraft.system_manager.set_average_flight_time(aircraft.flight_direction, average_flight_time)
        aircraft.system_manager.increase_flight_count(aircraft.flight_direction)

    # Convert taxi duration to seconds once
    taxi_duration_sec = ms_to_sec(aircraft.taxi_duration)

    # Check if the aircraft is at a vertiport node location
    for vertiport_id, vertiport in aircraft.system_manager.vertiports.items():
        if vertiport.is_node_vertiport_location(aircraft.location):
            if vertiport_id == aircraft.destination_vertiport_id:
                # Aircraft is at the destination vertiport
                if vertiport.is_parking_pad_location(aircraft.location):
                    return 0
                else:
                    # Time to taxi to the parking pad
                    return taxi_duration_sec
            elif vertiport_id == aircraft.origin_vertiport_id:
                # Aircraft is at the origin vertiport
                if vertiport.is_parking_pad_location(aircraft.location):
                    # Total ETA: taxi-out time + average flight time + taxi-in time at destination
                    return taxi_duration_sec + average_flight_time + taxi_duration_sec
                else:
                    # At takeoff pad; compute takeoff time without taxi duration
                    return average_flight_time + taxi_duration_sec

    if aircraft.location is None:
        attributes = vars(aircraft)
        for attribute, value in attributes.items():
            print(f"{attribute}: {value}")
        print("Time: ", aircraft.env.now)

    # Aircraft is in flight; calculate ETA based on waypoint progress
    waypoint_rank = aircraft.system_manager.airspace.get_waypoint_rank(
        aircraft.location, aircraft.flight_direction
    )
    flight_length = aircraft.system_manager.airspace.get_flight_length(
        aircraft.flight_direction
    )

    estimated_time_to_arrival = average_flight_time * (flight_length - waypoint_rank) / flight_length

    # Reset average flight time if it was not initialized before
    if not average_flight_time_initialized:
        aircraft.system_manager.set_average_flight_time(aircraft.flight_direction, None)
        aircraft.system_manager.decrease_flight_count(aircraft.flight_direction)

    # Total ETA includes the estimated time to arrival and taxi time at the destination
    return estimated_time_to_arrival + taxi_duration_sec

