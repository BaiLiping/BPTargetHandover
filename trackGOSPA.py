# ------------------------------------------------------------------------
# BP Target Handover
# Copyright (c) 2025 Liping Bai. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import numpy as np
from scipy.optimize import linear_sum_assignment

class TrackGOSPAMetric:
    def __init__(self, cutoff_distance, order, alpha=2, switching_penalty=5, has_assignment_input=False, distance_fcn=None):
        """
        Initializes the TrackGOSPAMetric.
        
        Parameters:
            cutoff_distance : float
                The maximum distance (c) used in the GOSPA calculation.
            order : float
                The order (p) used in the GOSPA metric.
            alpha : float, optional
                Alpha parameter in (0, 2]. When alpha==2, the GOSPA metric is decomposable
                into localization, missed target, and false track components.
                (Default is 2)
            switching_penalty : float, optional
                Nonzero penalty value to apply when a truth switches assignments.
                (Default is 5)
            has_assignment_input : bool, optional
                If True, the step method expects an external assignment (list of (track_id, truth_id)).
                (Default is False)
            distance_fcn : function, optional
                A custom function f(track, truth) -> scalar to compute the distance.
                If None, Euclidean distance between track["position"] and truth["position"] is used.
        """
        self.cutoff_distance = cutoff_distance
        self.order = order
        self.alpha = alpha
        self.switching_penalty = switching_penalty
        self.has_assignment_input = has_assignment_input
        self.distance_fcn = distance_fcn
        self.last_assignment = None  # Dictionary mapping truth id to track id

    def get_distance_matrix(self, tracks, truths):
        """
        Computes the distance matrix between each track and truth.
        Each element is calculated as:
            cost = min( (d(track, truth))^p, cutoff_distance^p )
        where p = order.
        
        Parameters:
            tracks : list of dict
            truths : list of dict
            
        Returns:
            d_matrix : NumPy array of shape (num_tracks, num_truths)
        """
        p = self.order
        c = self.cutoff_distance
        c_p = c ** p
        num_tracks = len(tracks)
        num_truths = len(truths)
        d_matrix = np.zeros((num_tracks, num_truths))
        
        for i, track in enumerate(tracks):
            for j, truth in enumerate(truths):
                if self.distance_fcn is not None:
                    d = self.distance_fcn(track, truth)
                else:
                    # Assume that both track and truth have a 'position' key.
                    pos_track = np.array(track["position"])
                    pos_truth = np.array(truth["position"])
                    d = np.linalg.norm(pos_track - pos_truth)
                # Use the p-th power of distance with a cutoff
                d_matrix[i, j] = min(d ** p, c_p)
        return d_matrix

    def get_current_associations(self, d_matrix, tracks, truths, assignment_input=None):
        """
        Returns the current assignment between tracks and truths.
        
        If has_assignment_input is True and an assignment is provided,
        the assignment is converted from identifiers to indices.
        Otherwise, the assignment is computed using the Hungarian algorithm
        (global nearest neighbor) with a cost threshold of (cutoff_distance^p)/2.
        
        Parameters:
            d_matrix : NumPy array
                The cost matrix.
            tracks : list of dict
            truths : list of dict
            assignment_input : list of (track_id, truth_id), optional
                
        Returns:
            indexed_assignments : list of tuples (track_index, truth_index)
        """
        p = self.order
        c = self.cutoff_distance
        c_p = c ** p

        if self.has_assignment_input and assignment_input is not None:
            # Convert identifier assignments to indices.
            track_ids = [track.get("id", i) for i, track in enumerate(tracks)]
            truth_ids = [truth.get("id", j) for j, truth in enumerate(truths)]
            track_id_to_index = {tid: i for i, tid in enumerate(track_ids)}
            truth_id_to_index = {tid: j for j, tid in enumerate(truth_ids)}
            indexed_assignments = []
            for t_id, tr_id in assignment_input:
                if t_id in track_id_to_index and tr_id in truth_id_to_index:
                    indexed_assignments.append((track_id_to_index[t_id], truth_id_to_index[tr_id]))
            return indexed_assignments
        else:
            # Compute assignment using the Hungarian algorithm.
            row_ind, col_ind = linear_sum_assignment(d_matrix)
            assignments = []
            # Only accept assignments with cost below the threshold.
            for i, j in zip(row_ind, col_ind):
                if d_matrix[i, j] < (c_p / 2):
                    assignments.append((i, j))
            return assignments

    def compute_switching_penalty(self, assignments, tracks, truths):
        """
        Computes the switching penalty by comparing the current assignment to the previous one.
        An assignment is represented as a mapping (dictionary) from truth IDs to track IDs,
        where a value of 0 indicates that a truth is unassigned.
        
        For each truth present in both the current and previous assignments:
          - A full switch (both assignments nonzero and different) counts fully.
          - A half switch (one assignment is zero) counts as 0.5.
        
        The switching penalty is then:
            switching_penalty * (num_full_switches + 0.5*num_half_switches)^(1/order)
        
        Parameters:
            assignments : list of (track_index, truth_index)
            tracks : list of dict
            truths : list of dict
            
        Returns:
            switching_penalty_value : float
        """
        p = self.order
        
        # Build the current assignment dictionary: truth_id -> track_id (0 if unassigned)
        curr_assignment = {}
        for j, truth in enumerate(truths):
            # Use provided 'id' if available; otherwise, use the index.
            truth_id = truth.get("id", j)
            curr_assignment[truth_id] = 0  # start as unassigned
        for (i, j) in assignments:
            track = tracks[i]
            truth = truths[j]
            track_id = track.get("id", i)
            truth_id = truth.get("id", j)
            curr_assignment[truth_id] = track_id

        if self.last_assignment is None:
            switching_penalty_value = 0
        else:
            full_switches = 0
            half_switches = 0
            common_truths = set(curr_assignment.keys()).intersection(self.last_assignment.keys())
            for tid in common_truths:
                prev_track = self.last_assignment.get(tid, 0)
                curr_track = curr_assignment.get(tid, 0)
                if prev_track != curr_track:
                    if prev_track and curr_track:  # both nonzero → full switch
                        full_switches += 1
                    else:
                        half_switches += 1
            switching_penalty_value = self.switching_penalty * ((full_switches + 0.5 * half_switches) ** (1 / p))
        # Update the stored assignment for next time.
        self.last_assignment = curr_assignment
        return switching_penalty_value

    def step(self, tracks, truths, assignment_input=None):
        """
        Computes the GOSPA metric.
        
        Parameters:
            tracks : list of dict
                Each dictionary must have a 'position' key (and optionally 'id').
            truths : list of dict
                Each dictionary must have a 'position' key (and optionally 'id').
            assignment_input : list of (track_id, truth_id), optional
                Only used if has_assignment_input is True.
        
        Returns:
            If alpha == 2, a tuple:
                (lgospa, gospa, switching, localization, miss_truth, false_tracks)
            Otherwise, a tuple:
                (lgospa, gospa, switching)
        """
        p = self.order
        alpha = self.alpha
        c = self.cutoff_distance
        c_p = c ** p

        num_tracks = len(tracks)
        num_truths = len(truths)
        n = max(num_tracks, num_truths)
        m = min(num_tracks, num_truths)

        d_matrix = self.get_distance_matrix(tracks, truths)
        indexed_assignments = self.get_current_associations(d_matrix, tracks, truths, assignment_input)

        assigned_costs = [d_matrix[i, j] for (i, j) in indexed_assignments]
        loc_ospa = (sum(assigned_costs)) ** (1 / p) if assigned_costs else 0

        num_assignments = len(indexed_assignments)
        mis_m = (c_p / alpha * ((alpha - 1) * (m - num_assignments))) ** (1 / p) if (m - num_assignments) > 0 else 0
        mis_n = (c_p / alpha * (n - num_assignments)) ** (1 / p) if (n - num_assignments) > 0 else 0

        gospa_p = (loc_ospa ** p + mis_m ** p + mis_n ** p)
        gospa = gospa_p ** (1 / p)

        switching = self.compute_switching_penalty(indexed_assignments, tracks, truths)
        lgospa = (gospa_p + switching ** p) ** (1 / p)

        # If alpha is 2, also return the component breakdown.
        if alpha == 2:
            # Following the MATLAB logic:
            # If the smaller set corresponds to truths then mis_m is the missed truth component.
            if m == num_truths:
                miss_truth = mis_m
                false_tracks = mis_n
            else:
                miss_truth = mis_n
                false_tracks = mis_m
            return lgospa, gospa, switching, loc_ospa, miss_truth, false_tracks
        else:
            return lgospa, gospa, switching
