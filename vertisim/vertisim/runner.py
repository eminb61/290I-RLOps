import argparse
from multiprocessing import Pool, Process, Manager
from queue import Empty
from .vertisim import VertiSim
import json
import simpy
import sqlite3
from collections import defaultdict
from .utils.helpers import read_config_file
import argparse
import glob
import os
from queue import Empty
import time
import re
import pandas as pd
from tqdm import tqdm

# Constants for database
QUEUE_TIMEOUT = 1  # Timeout in seconds for reading from queue

def runner(config, passenger_schedule_file_path=None):
    if passenger_schedule_file_path is not None:
        config['network_and_demand_params']['passenger_schedule_file_path'] = passenger_schedule_file_path
        print(f"Running simulation for pax file: {passenger_schedule_file_path}")
    env = simpy.Environment()
    vertisim = VertiSim(env=env, config=config)
    vertisim.run()

    # if results and results_list is not None:
    #     results_list.append(results)

def run_simulation_with_config(config_path, base_config, results_list):
    config = base_config.copy()
    config['network_and_demand_params']['passenger_schedule_file_path'] = config_path
    config['sim_params']['logging'] = False
    print(f"Running simulation with config: {config_path}")
    runner(config, results_list)

def db_writer(queue, db_file):
    """Process that writes data to the SQLite database from the queue."""
    conn = create_connection(db_file)
    cursor = conn.cursor()

    while True:
        try:
            # Get data from the queue
            data = queue.get(timeout=QUEUE_TIMEOUT)
        except Empty:
            continue  # Continue if no data is available in the queue
        if data is None:
            break  # Exit if sentinel value is received

        # Unpack the result dictionary from the queue
        try:
            save_results_to_db(cursor, data)
            conn.commit()
        except sqlite3.Error as err:
            print(f"Error writing to database: {err}")

    conn.close()
    print("Database writer process has shut down.")

def create_connection(db_file: str, timeout: int = 30):
    """Create a database connection to the SQLite database specified by db_file with the specified timeout."""
    try:
        conn = sqlite3.connect(db_file, timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging for better concurrency
        return conn
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
    
def save_results_to_db(cursor, results):
    """Helper function to save results to the database."""
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Simulation (
                SimulationID VARCHAR(8) PRIMARY KEY,
                SimulatedDate TEXT,
                Algorithm TEXT,
                NumAircraft INTEGER,
                NSpilled INTEGER,
                NFlights INTEGER,
                NumPax INTEGER,
                Reward REAL
            );
        """)
        # Insert or update the simulation results in the database
        cursor.execute("""
            INSERT INTO Simulation (SimulationID, SimulatedDate, Algorithm, NumAircraft, NSpilled, NFlights, NumPax, Reward)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(SimulationID) DO UPDATE SET
            SimulatedDate=excluded.SimulatedDate,
            Algorithm=excluded.Algorithm,
            NumAircraft=excluded.NumAircraft,
            NSpilled=excluded.NSpilled,
            NFlights=excluded.NFlights,
            NumPax=excluded.NumPax,
            Reward=excluded.Reward;
        """, (
            results['SimulationID'],
            results['SimulatedDate'],
            results['Algorithm'],
            results['NumAircraft'],
            results['NSpilled'],
            results['NFlights'],
            results['NumPax'],
            results['Reward']
        ))
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

def create_multi_process_runner(env_config, args):

    if args.batch_run:
        passenger_schedule_file = args.passenger_schedule_file
        csv_files = glob.glob(os.path.join(passenger_schedule_file, "*.csv"))

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in folder: {passenger_schedule_file}")

        print(f"Running simulations in batch mode on {args.n_cores} cores")
        
        # Create a pool of workers and run the simulations in parallel
        with Pool(processes=args.n_cores) as pool:
            # Wrap the starmap with tqdm for progress
            list(tqdm(pool.starmap(runner, [(env_config, passenger_schedule_file_path) for passenger_schedule_file_path in csv_files]), 
                      total=len(csv_files), desc="Simulations Progress", unit="simulation"))


        # Signal the database writer process to shut down
        # queue.put(None)
        # db_process.join()
        print("All simulations completed, and database writer has shut down.")
    else:
        runner(env_config)

    end_time = time.time()  # End time of the code execution
    run_time = end_time - start_time  # Calculate the code run time
    print(f"Code run time: {round(run_time/60, 1)} minutes")

def parse_args():
    parser = argparse.ArgumentParser(description='Run VertiSim with a configuration file')
    parser.add_argument('-ec', '--env_config', type=str, help='Path to the config.json file', default=None)
    parser.add_argument('-br', '--batch_run', action='store_true', help='Run the simulation in batch')
    parser.add_argument('-nc', '--n_cores', type=int, help='Number of cores to run the simulation on', default=1)
    parser.add_argument('-paxp', '--passenger_schedule_file', type=str, help='Path to the passenger schedule file', default=None)
    parser.add_argument('-sp', '--sweep_path', type=str, help='Run the simulation in sweep mode', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    start_time = time.time()  # Start time of the code execution
    args = parse_args()

    if args.sweep_path:
        # Read all the configuration files in the sweep path
        config_files = glob.glob(os.path.join(args.sweep_path, "*.json"))
        if len(config_files) == 0:
            raise ValueError(f"No JSON files found in folder: {args.sweep_path}")
        # Extract the hub, spoke and demand configuration from the file name. Filenames: config_ondemand_1_hub_8_spoke_40_a_2000_d.json
        pattern = r"config_ondemand_(\d+)_hub_(\d+)_spoke_\d+_a_(\d+)_d"
        for config_file in config_files:
            file_name = os.path.basename(config_file)
            match = re.match(pattern, file_name)

            if match:
                hub = int(match.group(1))
                spoke = int(match.group(2))
                demand = int(match.group(3))
                args.passenger_schedule_file = f"vertisim/vertisim/input/passenger/spill_demand/{hub}_hub_{spoke}_spoke_{demand}"
                config = read_config_file(config_file)
                print(f"Running simulation with config: {config_file}")
                create_multi_process_runner(config, args)
            else:
                print(f"Filename {file_name} does not match the pattern {pattern}")
        
    elif args.batch_run:
        env_config = read_config_file(args.env_config)
        create_multi_process_runner(env_config, args)


    elif args.env_config:
        env_config = read_config_file(args.env_config)
        env_config['sim_params']['algorithm'] = "VertiSimHeuristics"
        create_multi_process_runner(env_config, args)
    else:
        raise ValueError("Please provide a configuration file")

