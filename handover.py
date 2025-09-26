# ------------------------------------------------------------------------
# BP Target Handover
# Copyright (c) 2025 Liping Bai. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
import os
import pickle
import copy
import numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

from TrackerBP_handover import TrackerBP, BernoulliMixture 
from Utils import set_parameters, track_formation
from trackGOSPA import TrackGOSPAMetric

def process_trial(mc, all_true_tracks, all_measurements, all_measurement_flags,
                  base_parameters, results_folder):
    """
    Process one Monte Carlo trial (index mc) and save its results to disk.
    """
    trajectories      = all_true_tracks[mc]
    measurements      = all_measurements[mc]
    measurement_flags = all_measurement_flags[mc]
    n_steps           = base_parameters['n_steps']

    # --- Prepare sensor‐specific parameter sets ---
    parameters_sensor1 = deepcopy(base_parameters)
    parameters_sensor1.update({'num_sensors': 1, 'detection_threshold': 0.5})
    parameters_sensor2 = deepcopy(base_parameters)
    parameters_sensor2.update({'num_sensors': 1, 'detection_threshold': 0.5})

    # --- Storage for estimates & cardinalities ---
    estimates_sensor1 = [None] * n_steps
    estimates_sensor2 = [None] * n_steps
    card1 = np.zeros(n_steps)
    card2 = np.zeros(n_steps)

    # --- Initialize trackers & handover mixtures ---
    t1 = TrackerBP(parameters_sensor1)
    t2 = TrackerBP(parameters_sensor2)
    t3 = TrackerBP(parameters_sensor2)

    hand12 = BernoulliMixture(
        states    = np.zeros((4, parameters_sensor1['num_particles'], 0)),
        existence = [],
        label     = []
    )

    hand21 = BernoulliMixture(
        states    = np.zeros((4, parameters_sensor2['num_particles'], 0)),
        existence = [],
        label     = []
    )
    
    for k in range(n_steps):
        # in order to simulate the parallel processing of the two sensors
        # we have to update the trackers in a round-robin fashion
        hand12_meas = np.zeros((2, 0))
        hand21_meas = np.zeros((2, 0))
        # --- Sensor 1 predict at step=k ---
        sensor_index = 0
        handover_index = 1
        threshold = 0.5
        meas1 = np.atleast_1d(measurements[k][sensor_index])
        hand12 = t1.compute_alpha(handover_index, sensor_index, threshold)
        # --- Sensor 2 predict at step=k ---
        sensor_index = 1
        handover_index = 0
        threshold = 0.5
        meas2 = np.atleast_1d(measurements[k][sensor_index])
        hand21 = t2.compute_alpha(handover_index, sensor_index, threshold)
        # incorporate the handover
        handover_index = 1
        t1.incorporate_handover(hand21, handover_index)
        handover_index = 0
        t2.incorporate_handover(hand12, handover_index)
        # --- Sensor 1 measurement evaluation at step=k ---
        sensor_index = 0
        handover_index = 1
        meas1 = np.atleast_1d(measurements[k][sensor_index])
        t1.compute_xi_sigma(meas1, sensor_index)
        hand12_meas = t1.compute_beta_handover_measurements(meas1, sensor_index)
        t1.compute_kappa_iota()
        t1.compute_gamma()
        # --- Sensor 2 measurement evaluation at step=k ---
        sensor_index = 1
        handover_index = 0
        meas2 = np.atleast_1d(measurements[k][sensor_index])
        t2.compute_xi_sigma(meas2, sensor_index)
        hand21_meas = t2.compute_beta_handover_measurements(meas2, sensor_index)
        t2.compute_kappa_iota()
        t2.compute_gamma()
        # --- Sensor 1 update at step=k ---
        sensor_index = 0
        handover_index = 1
        if hand21_meas.shape[1] > 0:
            t1.compute_xi_sigma(hand21_meas, handover_index)
            t1.compute_beta(hand21_meas, handover_index)
            t1.compute_kappa_iota()
            t1.compute_gamma()
        t1.gamma.label = np.atleast_1d(t1.gamma.label)
        t1.prune()
        estimates_sensor1[k] = deepcopy(t1.estimate_state())
        card1[k] = t1.estimate_cardinality()
        # --- Sensor 2 update at step=k ---
        sensor_index = 1
        handover_index = 0
        if hand12_meas.shape[1] > 0:
            t2.compute_xi_sigma(hand12_meas, handover_index)
            t2.compute_beta(hand12_meas, handover_index)
            t2.compute_kappa_iota()
            t2.compute_gamma()
        t2.gamma.label = np.atleast_1d(t2.gamma.label)
        t2.prune()
        estimates_sensor2[k] = deepcopy(t2.estimate_state())
        card2[k] = t2.estimate_cardinality()
        # --- Sensor 3 update at step=k ---
        #t3.gamma.copy_from(t2.gamma)
        # --- Sensor 3 predict at step=k ---
        #sensor_index = 1
        #handover_index = 0
        #if k < n_steps - 1:
        #    meas2 = np.atleast_1d(measurements[k+1][sensor_index])
        #    hand21_next = t3.compute_alpha(hand12, handover_index, sensor_index)

    # --- Track formation & Track-GOSPA ---
    det1, tl1, clutter1, miss1, comp1, gospa_track1 = \
        track_formation(estimates_sensor1, n_steps)
    det2, tl2, clutter2, miss2, comp2, gospa_track2 = \
        track_formation(estimates_sensor2, n_steps)


    # --- Save results ---
    trial_data = {
        'true_tracks': trajectories,
        'measurements': measurements,
        'measurement_flags': measurement_flags,
        'estimates_sensor1': estimates_sensor1,
        'estimates_sensor2': estimates_sensor2,
        'estimated_cardinality_sensor1': card1,
        'estimated_cardinality_sensor2': card2,
        'detection_dict_sensor1': det1,
        'true_labels_sensor1': tl1,
        'clutter_sensor1': clutter1,
        'missed_detections_sensor1': miss1,
        'complete_tracks_sensor1': comp1,
        'gospa_track_sensor1': gospa_track1,
        'detection_dict_sensor2': det2,
        'true_labels_sensor2': tl2,
        'clutter_sensor2': clutter2,
        'missed_detections_sensor2': miss2,
        'complete_tracks_sensor2': comp2,
        'gospa_track_sensor2': gospa_track2,
        'parameters': parameters_sensor1
    }

    out_path = os.path.join(results_folder, f"trial_{mc:04d}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(trial_data, f)

    print(f"[{os.getpid()}] Trial {mc} → {out_path}")

def run_handover_multithreaded():
    base_parameters = set_parameters()
    results_folder  = "/Users/lipingb/Desktop/handover_experiment"
    os.makedirs(results_folder, exist_ok=True)

    # Load all Monte Carlo data once
    with open('/Users/lipingb/Desktop/data_generation.pkl', 'rb') as f:
        data = pickle.load(f)
    all_true_tracks        = data['all_true_tracks']
    all_measurements       = data['all_measurements']
    all_measurement_flags  = data['all_measurement_flags']

    num_mc = base_parameters['num_mc']

    # Spin up one process per core, submit all trials
    with ProcessPoolExecutor(max_workers=12) as exe:
        futures = [
            exe.submit(process_trial, mc,
                       all_true_tracks, all_measurements, all_measurement_flags,
                       base_parameters, results_folder)
            for mc in range(num_mc)
        ]
        # Optional: re-raise any exceptions here
        for fut in as_completed(futures):
            fut.result()


if __name__ == '__main__':
    run_handover_multithreaded()
