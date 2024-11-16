from collections import defaultdict
import numpy as np
import pandas as pd
from .units import ms_to_min

def calculate_passenger_trip_time_stats(passenger_trip_time_tracker, flight_directions, logger, print_metrics=False):
    trip_times = defaultdict(lambda: defaultdict(dict))
    for vertiport_id, data in passenger_trip_time_tracker.items():
        waiting_times = np.array(list(data.values()))
        trip_times[vertiport_id]['passenger_trip_time']['mean'] = ms_to_min(np.mean(waiting_times))
        trip_times[vertiport_id]['passenger_trip_time']['median'] = ms_to_min(np.median(waiting_times))
        trip_times[vertiport_id]['passenger_trip_time']['std'] = ms_to_min(np.std(waiting_times))
        trip_times[vertiport_id]['passenger_trip_time']['max'] = ms_to_min(np.max(waiting_times))
        trip_times[vertiport_id]['passenger_trip_time']['min'] = ms_to_min(np.min(waiting_times))
    try:
        logger.info("Passenger trip time stats (mins):")
        logger.info("----------------------------------")
        df = pd.DataFrame(
            {
                'mean': trip_times[vertiport_id]['passenger_trip_time']['mean'],
                'median': trip_times[vertiport_id]['passenger_trip_time']['median'],
                'std': trip_times[vertiport_id]['passenger_trip_time']['std'],
                'max': trip_times[vertiport_id]['passenger_trip_time']['max'],
                'min': trip_times[vertiport_id]['passenger_trip_time']['min']
            } for vertiport_id in trip_times.keys()
        )
        df.index = trip_times.keys()
        logger.info(df)
        logger.info("")
        
        if print_metrics:
            print("Passenger trip time stats (mins):")
            print("----------------------------------")
            print(df)
            print("----------------------------------")
            print("")         
    except:
        pass

def calculate_passenger_waiting_time_stats(passenger_waiting_time_tracker, logger, print_metrics):
    waiting_times_dict = defaultdict(lambda: defaultdict(dict))
    for vertiport_id, data in passenger_waiting_time_tracker.items():
        waiting_times = np.array(list(data.values()))
        waiting_times_dict[vertiport_id]['passenger_waiting_time']['mean'] = ms_to_min(np.mean(waiting_times))
        waiting_times_dict[vertiport_id]['passenger_waiting_time']['median'] = ms_to_min(np.median(waiting_times))
        waiting_times_dict[vertiport_id]['passenger_waiting_time']['std'] = ms_to_min(np.std(waiting_times))
        waiting_times_dict[vertiport_id]['passenger_waiting_time']['max'] = ms_to_min(np.max(waiting_times))
        waiting_times_dict[vertiport_id]['passenger_waiting_time']['min'] = ms_to_min(np.min(waiting_times))
    
    try:
        logger.info("Passenger waiting time stats (mins):")
        logger.info("-------------------------------------")
        df = pd.DataFrame(
            {
                'Vertiport ID': vertiport_id,
                'mean': waiting_times_dict[vertiport_id]['passenger_waiting_time']['mean'],
                'median': waiting_times_dict[vertiport_id]['passenger_waiting_time']['median'],
                'std': waiting_times_dict[vertiport_id]['passenger_waiting_time']['std'],
                'max': waiting_times_dict[vertiport_id]['passenger_waiting_time']['max'],
                'min': waiting_times_dict[vertiport_id]['passenger_waiting_time']['min']
            } for vertiport_id in waiting_times_dict.keys()
        )
        logger.info(df)
        logger.info("-------------------------------------")
        logger.info("")

        if print_metrics:
            print("Passenger waiting time stats (mins):")
            print("--------------------------------------")
            print(df)
            print("--------------------------------------")
            print("")
    except:
        pass


def log_and_print_spilled_passengers(spilled_passengers, logger, print_metrics):
    try:
        # Create DataFrame
        df = pd.DataFrame(list(spilled_passengers.items()), columns=['Flight Dir', 'Spilled Passengers'])
        
        # Calculate total spilled passengers
        total_spilled = df['Spilled Passengers'].sum()
        
        # Append a new row for the total
        total_row = pd.DataFrame({'Flight Dir': ['Total'], 'Spilled Passengers': [total_spilled]})
        df = pd.concat([df, total_row], ignore_index=True)
        
        logger.info("Spilled passengers:")
        logger.info("-------------------")
        logger.info(df)
        logger.info("-------------------")
        logger.info("")
    
        if print_metrics:
            print("Spilled passengers:")
            print("-------------------")
            print(df)
            print("-------------------")
            print("")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

