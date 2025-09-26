import numpy as np
import pickle
from Utils import set_parameters, generate_true_tracks, generate_cluttered_measurements

def generate_data():
    """
    Generates true tracks and cluttered measurements (with accompanying measurement flags)
    for multiple Monte Carlo trials, and saves them into a pickle file.
    """
    # Set experiment parameters
    parameters = set_parameters()
    parameters['p_d'] = 1 
    num_mc = parameters['num_mc']
    
    # Preallocate lists for storing data for each trial
    all_true_tracks = [None] * num_mc
    all_measurements = [None] * num_mc
    all_measurement_flags = [None] * num_mc

    for mc in range(num_mc):
        print(f"Processing trial {mc+1}/{num_mc}")

        # Perturb the initial positions and meeting point for diversity across trials
        init_positions = parameters['initialTargetPositions'] + np.random.randn(2, 2)
        parameters['initialTargetPositions'] = init_positions
        parameters['meeting_point'] = parameters['meeting_point'] + np.random.randn(2)
        
        # Generate the true trajectories for all targets
        trajectories = generate_true_tracks(parameters)
        
        # Generate cluttered measurements and corresponding flags from the true trajectories.
        measurements, measurement_flags = generate_cluttered_measurements(trajectories, parameters)
        
        # Store data for this Monte Carlo trial
        all_true_tracks[mc] = trajectories
        all_measurements[mc] = measurements
        all_measurement_flags[mc] = measurement_flags

    # Create a dictionary to hold all the generated data
    data = {
        'all_true_tracks': all_true_tracks,     # True trajectories for each trial
        'all_measurements': all_measurements,     # Cluttered measurements for each trial
        'all_measurement_flags': all_measurement_flags,  # Flags indicating clutter (0) vs. true (1) measurement
        'parameters': parameters                  # Experiment parameters used
    }
    
    # Save the data dictionary to a pickle file
    with open('/Users/lipingb/Desktop/data_generation_temp.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("Data generation complete. Results saved to 'data_generation.pkl'.")

if __name__ == '__main__':
    generate_data()
