# ------------------------------------------------------------------------
# BP Target Handover
# Copyright (c) 2025 Liping Bai. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import pickle
from TrackerBP_centralized import TrackerBP
from Utils import set_parameters, track_formation
from copy import deepcopy
import os
import imageio
from PIL import Image
from matplotlib.lines import Line2D  # For custom legend handles
import random
from generate_data import generate_data
from scipy.optimize import linear_sum_assignment
from visualize_centralized_result import visualize_centralized_result
from multiprocessing import Pool, Manager

def process_trial(mc, parameters, data, shared_results):
    """
    Process one Monte Carlo trial.
    This function runs the centralized tracker for one trial,
    """
    results_folder = "/Users/lipingb/Desktop/centralized_experiement"
    os.makedirs(results_folder, exist_ok=True)
    
    n_steps = parameters['n_steps']
    # Extract trial data.
    trajectories = data['all_true_tracks'][mc]  # Expected: list of true track sets per time step.
    measurements = data['all_measurements'][mc]
    measurement_flags = data['all_measurement_flags'][mc]
    
    # Containers for storing estimates and cardinality over time.
    estimates = [None] * n_steps
    estimated_cardinality = np.zeros(n_steps)
    
    # Initialize the centralized tracker.
    centralized_tracker = TrackerBP(parameters)
    
    # Run the tracker over all time steps.
    for step in range(n_steps):
        current_measurements = measurements[step]
        centralized_tracker.compute_alpha()
        for sensor_index in range(parameters['num_sensors']):
            sensor_measurements = np.array(current_measurements[sensor_index])
            centralized_tracker.compute_xi_sigma(sensor_measurements, sensor_index)
            centralized_tracker.compute_beta(sensor_measurements, sensor_index)
            centralized_tracker.compute_kappa_iota()
            centralized_tracker.compute_gamma()
            centralized_tracker.prune()
        estimates[step] = deepcopy(centralized_tracker.estimate_state())
        estimated_cardinality[step] = centralized_tracker.estimate_cardinality()
    
    # Process estimated tracks for track formation.
    detection_dict, true_labels, clutter, missed_detections, complete_tracks, gospa_track = track_formation(estimates, n_steps)
    
    
    # Package trial results.
    trial_results = {
        'measurements': measurements,
        'measurement_flags': measurement_flags,
        'true_tracks': trajectories,
        'estimates': estimates,
        'estimated_cardinality': estimated_cardinality,
        'detection_dict': detection_dict,
        'true_labels': true_labels,
        'clutter': clutter,
        'missed_detections': missed_detections,
        'complete_tracks': complete_tracks,
        'gospa_track': gospa_track,
        'parameters': parameters
    }
    trial_filename = os.path.join(results_folder, f"trial_{mc:04d}.pkl")
    with open(trial_filename, 'wb') as f:
        pickle.dump(trial_results, f)
    print(f"Trial {mc+1} results saved to {trial_filename}")


def run_centralized_mc_parallel():
    """
    Loads pre-generated data and global parameters, then processes all Monte Carlo trials
    in parallel using a Manager list to store results at each trial index.
    """
    # Set global parameters.
    parameters = set_parameters()  # Expected to return a dict with keys like 'num_mc', 'n_steps', etc.
    results_folder = "/Users/lipingb/Desktop/centralized_experiement"
    os.makedirs(results_folder, exist_ok=True)
    
    # Load pre-generated data.
    with open('/Users/lipingb/Desktop/data_generation.pkl', 'rb') as f:
        data = pickle.load(f)
    
    num_mc = parameters['num_mc']
    n_steps = parameters['n_steps']
    
    # Create a Manager list to store results from each trial.
    with Manager() as manager:
        shared_results = manager.list([None] * num_mc)
        
        # Create a pool with 12 processes.
        with Pool(processes=12) as pool:
            args = [(mc, parameters, data, shared_results) for mc in range(num_mc)]
            pool.starmap(process_trial, args)

    plt.show()    
    print('Tracking experiment completed.')

if __name__ == '__main__':
    # Generate data first.
    # Run the Monte Carlo trials in parallel using shared memory for results.
    run_centralized_mc_parallel()
    # Visualize the centralized tracking result.
    #visualize_centralized_result()
