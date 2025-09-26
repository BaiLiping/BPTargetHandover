# ------------------------------------------------------------------------
# BP Target Handover
# Copyright (c) 2025 Liping Bai. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import numpy as np
# Create a custom color map for estimated tracks.
custom_colors = [
        '#FF00FF', '#1E90FF', '#FFA500', '#00FF00', '#FF0000',
        '#8A2BE2', '#EE82EE', '#FF1493', '#7B68EE', '#00CED1',
        '#DC143C', '#FF69B4', '#00FFFF', '#FFFF00', '#0000FF',
        '#008000', '#800080', '#808000', '#FFA07A', '#7FFF00',
        '#9932CC', '#FF6347', '#FFD700', '#ADFF2F', '#4B0082',
        '#00BFFF', '#B22222', '#8B4513', '#2E8B57', '#4682B4',
        '#D2691E', '#DA70D6', '#9ACD32', '#F08080', '#BA55D3',
        '#66CDAA'
    ]



def track_formation(estimatedTracks, n_steps):
    """
    Processes the estimated tracks over time to build a detection dictionary,
    determine true tracks versus false alarms, compute missed detections, and 
    combine the detection and missed detection events into a complete track for 
    each true track label.
    
    Parameters:
        estimatedTracks (list): List (length n_steps) of estimated track dicts.
        n_steps (int): Total number of time steps.
    
    Returns:
        detection_dict (dict): Maps track labels to a list of (time, position) tuples for detections.
        true_labels (set): Labels with at least 10 detections.
        fa_labels (set): Labels with fewer than 10 detections.
        missed_detections (dict): For each true track label, a list of missed detection tuples.
        complete_tracks (dict): For each true track label, a sorted list of all events in a uniform format.
                                Each event is a tuple: (time, [x,y], event_type)
    """
    detection_dict = {}
    all_labels = set()
    gospa_tracks = [[] for i in range(n_steps)]

    for t in range(n_steps):
        estimatedTrack = estimatedTracks[t]
        if estimatedTrack is not None and 'state' in estimatedTrack:
            numEstTracks = estimatedTrack['state'].shape[1]
            labels = estimatedTrack.get('label', None)
            if labels is None:
                labels = list(range(numEstTracks))
            else:
                if isinstance(labels, list):
                    if len(labels) != numEstTracks:
                        labels = [labels[0]] * numEstTracks
                else:
                    labels = list(np.array(labels).flatten())
                    if len(labels) != numEstTracks:
                        labels = [labels[0]] * numEstTracks
            for i in range(numEstTracks):
                track_label = labels[i]
                if isinstance(track_label, np.ndarray):
                    track_label = track_label.item() if track_label.size == 1 else tuple(track_label.tolist())
                all_labels.add(track_label)
                pos = estimatedTrack['state'][0:2, i]
                existence = estimatedTrack['existence'][i]
                detection_dict.setdefault(track_label, []).append((t, pos, existence))
    
    # Determine true tracks (>= 10 detections) vs false alarms.
    true_labels = {label for label, events in detection_dict.items() if len(events) >= 10}
    fa_labels = {label for label, events in detection_dict.items() if len(events) < 10}

    # Compute missed detections for true tracks.
    missed_detections = {}
    for label in true_labels:
        events = detection_dict.get(label, [])
        events.sort(key=lambda x: x[0])
        detection_times = [e[0] for e in events]
        missed_detections[label] = []
        for t in range(n_steps):
            if t not in detection_times:
                currentEstTrack = estimatedTracks[t]
                gamma_obj = currentEstTrack['gamma']
                gamma_states = np.array(gamma_obj.states)
                gamma_existence = np.array(gamma_obj.existence)
                matched_idx = np.where(gamma_obj.label == label)[0]
                pos = np.mean(gamma_states[0:2, :, matched_idx], axis=1)
                existence_val = gamma_existence[matched_idx]   
                missed_detections[label].append((t, pos, existence_val))

    clutters = {}
    for label in fa_labels:
        clutter_events = detection_dict.get(label, [])
        clutter_events.sort(key=lambda x: x[0])
        clutter_events_mod = []
        for t, pos, existence in clutter_events:
            pos_list = pos.tolist() if isinstance(pos, np.ndarray) else pos
            clutter_events_mod.append((t, pos_list, existence, 'detection'))
        clutters[label] = clutter_events_mod

    
    # Combine detection and missed detection events into complete tracks.
    complete_tracks = {}
    for label in true_labels:
        detection_events = detection_dict.get(label, [])
        detection_events.sort(key=lambda x: x[0])
        detection_events_mod = []
        for t, pos, existence in detection_events:
            pos_list = pos.tolist() if isinstance(pos, np.ndarray) else pos
            detection_events_mod.append((t, pos_list, existence, 'detection'))
        missed_events = missed_detections.get(label, [])
        missed_events.sort(key=lambda x: x[0])
        missed_events_mod = []
        for t, pos, existence_val in missed_events:
            pos_list = pos.flatten().tolist()
            # Safely access the first element of existence_val if available.
            if (existence_val is not None and 
                isinstance(existence_val, np.ndarray) and 
                existence_val.size > 0):
                existence_value = existence_val[0]
            else:
                existence_value = None

            missed_events_mod.append((t, pos_list, existence_value, 'missed'))
            
        # Merge events and sort by time.
        all_events = detection_events_mod + missed_events_mod
        all_events.sort(key=lambda x: x[0])
        complete_tracks[label] = all_events

    for label in all_labels:
        events = detection_dict.get(label, [])
        events.sort(key=lambda x: x[0])
        for t, pos, existence in events:
            incidence = {}
            incidence['position'] = pos
            incidence['id'] = str(label)
            incidence['existence'] = existence
            gospa_tracks[t].append(incidence)

    return detection_dict, true_labels, clutters, missed_detections, complete_tracks, gospa_tracks


def set_parameters():
    """
    Initializes and returns the parameters for data generation and tracking.

    Returns:
        parameters (dict): Dictionary containing all tracking and data generation parameters.
    """
    # Data generation parameters
    p_d = 0.9
    p_s = 0.95
    clutter_lambda = 5
    sensing_range = 120   # Sensing range in meters
    sigma_range = 1     # Std dev for range measurements (meters)
    sigma_bearing = 1     # Std dev for bearing measurements (degrees)
    sigma_v = 0.05       # Process noise for trajectory

    # Base station positions
    bs1_x, bs1_y = 0, 0
    bs2_x, bs2_y = 150, 0

    # Initial target positions and meeting point
    ue1_x, ue1_y = -30, 50
    ue2_x, ue2_y = -70, -50
    meeting_point = np.array([80, 0])
    
    # Number of time steps for the experiment
    n_steps = 100

    # Tracking parameters stored in a dictionary
    parameters = {
        'mu_n' : 0.01,
        'd_t': 1, 
        'num_mc': 100,
        'n_steps': n_steps,
        'sigma_v': sigma_v,
        'plottingRegion': np.array([[-150, 300], [-150, 150]]),  # Each row: [min, max] for x and y
        'velocity_noise': np.diag([1**2, 1**2]),
        # Initial EKF covariance for births (Cartesian 4x4): diag([pos_x, pos_y, vel_x, vel_y])
        'init_cov': np.diag([100.0, 100.0, 10.0, 10.0]),
        'nun_steps': np.ones(n_steps),
        'p_s': p_s,
        'num_sensors': 2,
        'range_variance': sigma_range**2,
        'bearing_variance': sigma_bearing**2,
        'p_d': p_d,
        'measurement_range': sensing_range,
        'mu_c': clutter_lambda,
        'f_c': 1 / (360 * sensing_range),
        'detection_threshold': 0.5,
        'pruning_threshold': 1e-5,
        'minimumTrackLength': 1,
        'num_particles': 10000,
        # sensor_positions is a 2 x num_sensors array (each column: [bs_x; bs_y])
        'sensor_positions': np.array([[bs1_x, bs2_x],
                                     [bs1_y, bs2_y]]),
        # Including initial target positions and meeting point
        'initialTargetPositions': np.array([[ue1_x, ue1_y],
                                              [ue2_x, ue2_y]]),
        'meeting_point': meeting_point
    }
    
    return parameters

def generate_true_tracks(parameters):
    """
    Generates trajectories for all targets with the following constraints:
      - The forward-propagated trajectory (from the meeting point to the end)
        must end within the FoV of BS2.
      - The backward-propagated trajectory (from the meeting point to the start)
        must begin within the FoV of BS1.
    
    Returns:
        trajectories: 3D numpy array of shape (n_steps, 4, numTargets)
                      Each slice [:, :, i] is the trajectory of target i [x, y, vx, vy].
    """
    n_steps = parameters['n_steps']
    sigma_v = parameters['sigma_v']
    meeting_point = parameters['meeting_point']
    initial_positions = parameters['initialTargetPositions']
    numTargets = initial_positions.shape[0]
    T = 1.0  # Time step

    # State transition matrix.
    F = np.eye(4)
    F[0:2, 2:4] = T * np.eye(2)
    F_inv = np.linalg.inv(F)

    # Process noise covariance Q
    Q1 = np.array([[T**4/4, T**3/2],
                   [T**3/2, T**2]])
    Q = np.zeros((4, 4))
    Q[np.ix_([0,2], [0,2])] = Q1
    Q[np.ix_([1,3], [1,3])] = Q1
    Q = sigma_v**2 * Q

    midpoint_step = (n_steps - 1) // 2

    trajectories = np.zeros((n_steps, 4, numTargets))

    # Compute initial velocities so that each target reaches the meeting point at the midpoint.
    initial_velocities = np.zeros((numTargets, 2))
    for i in range(numTargets):
        delta = meeting_point - initial_positions[i, :]
        initial_velocities[i, :] = delta / midpoint_step

    # Set the state at the meeting point for each target.
    for i in range(numTargets):
        # Optionally, add a bit of randomness to the meeting point position.
        trajectories[midpoint_step, :, i] = np.hstack((meeting_point + np.random.randn(2),
                                                         initial_velocities[i, :]))

    # Retrieve measurement range and BS positions.
    meas_range = parameters['measurement_range']
    bs1_position = parameters['sensor_positions'][:, 0]  # BS1 position
    bs2_position = parameters['sensor_positions'][:, 1]  # BS2 position

    # Forward propagation (from meeting point to end) with BS2 FoV constraint.
    for i in range(numTargets):
        valid_forward = False
        while not valid_forward:
            temp_forward = np.zeros((n_steps - midpoint_step, 4))
            temp_forward[0, :] = trajectories[midpoint_step, :, i]
            for t in range(0, n_steps - midpoint_step - 1):
                w_k = np.random.multivariate_normal(np.zeros(4), Q)
                temp_forward[t+1, :] = F @ temp_forward[t, :] + w_k
            final_position = temp_forward[-1, 0:2]
            if np.linalg.norm(final_position - bs2_position) <= meas_range:
                valid_forward = True
                trajectories[midpoint_step:, :, i] = temp_forward

    # Backward propagation (from meeting point to start) with BS1 FoV constraint.
    for i in range(numTargets):
        valid_backward = False
        while not valid_backward:
            temp_backward = np.zeros((midpoint_step + 1, 4))
            temp_backward[-1, :] = trajectories[midpoint_step, :, i]  # starting from meeting point
            for t in range(midpoint_step - 1, -1, -1):
                w_k = np.random.multivariate_normal(np.zeros(4), Q)
                temp_backward[t, :] = F_inv @ (temp_backward[t+1, :] + w_k)
            initial_state = temp_backward[0, 0:2]
            if np.linalg.norm(initial_state - bs1_position) <= meas_range:
                valid_backward = True
                trajectories[0:midpoint_step+1, :, i] = temp_backward
    # Reverse the time order of trajectory 2 (i.e., target with index 1)
    if numTargets >= 2:
        trajectories[:, :, 1] = trajectories[::-1, :, 1]

    return trajectories

def generate_cluttered_measurements(trajectories, parameters):
    """
    Generates cluttered polar measurements for all time steps.
    
    Returns a tuple:
        measurements: list (length n_steps) of lists (length num_sensors) where each entry is a 2xM numpy array.
                      (i.e. measurements[t][sensor] is an array of shape (2, M) )
        measurement_flags: companion list with the same structure as measurements, where each entry is a 1D numpy array of length M.
                           Each element indicates:
                               0 for clutter measurement,
                               1 for target-originated measurement.
    """
    n_steps = trajectories.shape[0]
    num_sensors = parameters['sensor_positions'].shape[1]
    meas_range = parameters['measurement_range']
    meanClutter = parameters['mu_c']
    p_D = parameters['p_d']
    var_range = parameters['range_variance']
    var_bearing = parameters['bearing_variance']
    sensor_positions = parameters['sensor_positions']

    measurements = []
    measurement_flags = []
    
    for t in range(n_steps):
        meas_t = []
        flags_t = []
        # target_states: shape (numTargets, 4)
        target_states = trajectories[t, :, :].T  
        for sensor in range(num_sensors):
            bs_x, bs_y = sensor_positions[:, sensor]
            meas = []   # will hold [r, theta] pairs (each a list of 2 floats)
            flags = []  # 0 for clutter, 1 for target-originated
            
            # Generate clutter measurements.
            n_clutter = np.random.poisson(meanClutter)
            for _ in range(n_clutter):
                r_clutter = meas_range * np.sqrt(np.random.rand())
                theta_clutter = 360 * np.random.rand()
                # Explicitly cast to float.
                meas.append([float(r_clutter), float(theta_clutter)])
                flags.append(0)
                
            # Generate target-originated measurements (if within sensor coverage).
            for target_index, state in enumerate(target_states):
                # state is [x, y, vx, vy]
                x, y = float(state[0]), float(state[1])
                if check_coverage(x, y, bs_x, bs_y, meas_range):
                    if np.random.rand() <= p_D:
                        x_rel = x - bs_x
                        y_rel = y - bs_y
                        r_true = np.sqrt(x_rel**2 + y_rel**2)
                        theta_true = np.degrees(np.arctan2(y_rel, x_rel))
                        r_noisy = r_true + np.sqrt(var_range) * np.random.randn()
                        theta_noisy = theta_true + np.sqrt(var_bearing) * np.random.randn()
                        meas.append([float(r_noisy), float(theta_noisy)])
                        flags.append(1+target_index)
                        
            # Convert to numpy arrays with explicit dtypes.
            if len(meas) > 0:
                # meas is a list of [r, theta] pairs.
                # np.array(meas, dtype=float) yields an array of shape (M, 2); we then transpose to (2, M)
                meas_arr = np.array(meas, dtype=float).T  
                flags_arr = np.array(flags, dtype=int)
            else:
                meas_arr = np.empty((2, 0), dtype=float)
                flags_arr = np.empty((0,), dtype=int)
                
            meas_t.append(np.array(meas_arr))
            flags_t.append(flags_arr)
        measurements.append(meas_t)
        measurement_flags.append(flags_t)
    
    return measurements, measurement_flags

def check_coverage(x, y, bs_x, bs_y, sensing_range):
    """
    Returns True if the target at (x,y) is within the sensing range of (bs_x, bs_y).
    """
    distance = np.sqrt((x - bs_x)**2 + (y - bs_y)**2)
    return distance <= sensing_range

def compare_vector_with_matrix(matrix, vector):
    """
    Compare each column of 'matrix' with the 'vector'.
    Returns a boolean 1D array where True indicates that the column is equal to the vector.
    """
    # Ensure vector is 1D
    vector = vector.flatten()
    # Compare each column with the vector; broadcasting vector to shape (n, m)
    return np.all(matrix == vector[:, None], axis=0)
