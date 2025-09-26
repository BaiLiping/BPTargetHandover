# ------------------------------------------------------------------------
# BP Target Handover
# Copyright (c) 2025 Liping Bai. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import numpy as np
import pickle
import copy
from TrackerBP import TrackerBP
from Utils import set_parameters, track_formation  # Ensure track_formation is imported.
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager

def process_trial_distributed(mc, parameters, data, shared_results):
    """
    Process one Monte Carlo trial for the distributed experiment.
    Runs separate trackers for BS1 (sensor 1) and BS2 (sensor 2),
    performs track formation and computes Track-GOSPA metrics for each sensor,
    stores the gamma snapshots for debugging, saves the trial results to file,
    and stores selected metrics in shared_results.
    """
    results_folder = "/Users/lipingb/Desktop/distributed_experiment"
    os.makedirs(results_folder, exist_ok=True)
    
    n_steps = parameters['n_steps']
    # Extract trial data.
    trajectories = data['all_true_tracks'][mc]
    measurements = data['all_measurements'][mc]
    measurement_flags = data['all_measurement_flags'][mc]
    
    # Containers for storing per-step estimates, cardinality, and gamma snapshots.
    estimates_sensor1 = [None] * n_steps
    estimates_sensor2 = [None] * n_steps
    estimated_cardinality_sensor1 = np.zeros(n_steps)
    estimated_cardinality_sensor2 = np.zeros(n_steps)
    gamma_sensor1 = [None] * n_steps
    gamma_sensor2 = [None] * n_steps
    
    # Prepare sensor-specific parameter sets.
    parameters_sensor1 = copy.deepcopy(parameters)
    parameters_sensor1['num_sensors'] = 1
    parameters_sensor2 = copy.deepcopy(parameters)
    parameters_sensor2['num_sensors'] = 1
    
    # Initialize distributed trackers for BS1 and BS2.
    distributed_tracker1 = TrackerBP(parameters_sensor1)
    distributed_tracker2 = TrackerBP(parameters_sensor2)
    
    # Tracking loop: update each distributed tracker using its sensor's measurements.
    for step in range(n_steps):
        # --- BS1 Tracker Update ---
        sensor_index = 0
        current_measurements_sensor1 = measurements[step][sensor_index]
        distributed_tracker1.compute_alpha()
        sensor_measurements_1 = np.array(current_measurements_sensor1)
        distributed_tracker1.compute_xi_sigma(sensor_measurements_1, sensor_index)
        distributed_tracker1.compute_beta(sensor_measurements_1, sensor_index)
        distributed_tracker1.compute_kappa_iota()
        distributed_tracker1.compute_gamma()
        # Store the gamma snapshot for BS1.
        gamma_sensor1[step] = deepcopy(distributed_tracker1.gamma)
        distributed_tracker1.prune()
        estimates_sensor1[step] = deepcopy(distributed_tracker1.estimate_state())
        estimated_cardinality_sensor1[step] = distributed_tracker1.estimate_cardinality()
        
        # --- BS2 Tracker Update ---
        sensor_index = 1
        current_measurements_sensor2 = measurements[step][sensor_index]
        distributed_tracker2.compute_alpha()
        sensor_measurements_2 = np.array(current_measurements_sensor2)
        distributed_tracker2.compute_xi_sigma(sensor_measurements_2, sensor_index)
        distributed_tracker2.compute_beta(sensor_measurements_2, sensor_index)
        distributed_tracker2.compute_kappa_iota()
        distributed_tracker2.compute_gamma()
        # Store the gamma snapshot for BS2.
        gamma_sensor2[step] = deepcopy(distributed_tracker2.gamma)
        distributed_tracker2.prune()
        estimates_sensor2[step] = deepcopy(distributed_tracker2.estimate_state())
        estimated_cardinality_sensor2[step] = distributed_tracker2.estimate_cardinality()
    
    # --- Track Formation and Track-GOSPA Metrics Computation for BS1 ---
    detection_dict_sensor1, true_labels_sensor1, clutter_sensor1, \
    missed_detections_sensor1, complete_tracks_sensor1, gospa_track_sensor1 = \
        track_formation(estimates_sensor1, n_steps)
    

    
    # --- Track Formation and Track-GOSPA Metrics Computation for BS2 ---
    detection_dict_sensor2, true_labels_sensor2, clutter_sensor2, \
    missed_detections_sensor2, complete_tracks_sensor2, gospa_track_sensor2 = \
        track_formation(estimates_sensor2, n_steps)

    
    # Package trial results including gamma snapshots.
    trial_results = {
        'true_tracks': trajectories,
        'measurements': measurements,
        'measurement_flags': measurement_flags,
        'estimates_sensor1': estimates_sensor1,
        'estimates_sensor2': estimates_sensor2,
        'estimated_cardinality_sensor1': estimated_cardinality_sensor1,
        'estimated_cardinality_sensor2': estimated_cardinality_sensor2,
        'detection_dict_sensor1': detection_dict_sensor1,
        'true_labels_sensor1': true_labels_sensor1,
        'clutter_sensor1': clutter_sensor1,
        'missed_detections_sensor1': missed_detections_sensor1,
        'complete_tracks_sensor1': complete_tracks_sensor1,
        'gospa_track_sensor1': gospa_track_sensor1,
        'detection_dict_sensor2': detection_dict_sensor2,
        'true_labels_sensor2': true_labels_sensor2,
        'clutter_sensor2': clutter_sensor2,
        'missed_detections_sensor2': missed_detections_sensor2,
        'complete_tracks_sensor2': complete_tracks_sensor2,
        'gospa_track_sensor2': gospa_track_sensor2,
        'parameters': parameters
    }
    
    trial_filename = os.path.join(results_folder, f"trial_{mc:04d}.pkl")
    with open(trial_filename, 'wb') as f:
        pickle.dump(trial_results, f)
    print(f"Distributed trial {mc+1} results saved to {trial_filename}")

def run_distributed_mc_parallel():
    """
    Loads pre-generated data and parameters, then processes all Monte Carlo trials in parallel.
    After processing, additional post-processing or averaging of metrics can be performed.
    """
    parameters = set_parameters()  # Global parameters.
    results_folder = "/Users/lipingb/Desktop/distributed_experiment"
    os.makedirs(results_folder, exist_ok=True)
    
    # Load pre-generated data.
    with open('/Users/lipingb/Desktop/data_generation.pkl', 'rb') as f:
        data = pickle.load(f)
    
    num_mc = parameters['num_mc']
    n_steps = parameters['n_steps']
    
    # Create a Manager list to store results from each trial.
    with Manager() as manager:
        shared_results = manager.list([None] * num_mc)
        
        # Process the trials in parallel using 12 processes.
        with Pool(processes=8) as pool:
            args = [(mc, parameters, data, shared_results) for mc in range(num_mc)]
            pool.starmap(process_trial_distributed, args)
        
        # Convert the Manager list to a regular list for further processing if needed.
        all_trials_results = list(shared_results)
    
    print("Distributed tracking experiment completed.")

if __name__ == '__main__':
    # Run the distributed Monte Carlo trials in parallel.
    run_distributed_mc_parallel()
    # Visualize the distributed tracking results.
    #visualize_distributed_result()
