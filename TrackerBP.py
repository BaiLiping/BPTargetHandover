# ------------------------------------------------------------------------
# BP Target Handover
# Copyright (c) 2025 Liping Bai. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import numpy as np
from scipy.linalg import cholesky
from copy import deepcopy
from typing import Optional, Any
from scipy.stats import multivariate_normal

class BernoulliMixture:
    """
    Holds the states, existence probabilities, and labels for a multi-Bernoulli mixture.
    """
    def __init__(self,
                 states: Optional[np.ndarray] = None,
                 existence: Optional[Any] = None,
                 label: Optional[Any] = None) -> None:
        self.states = states
        # Force existence and label to be lists.
        self.existence = list(existence) if existence is not None else []
        self.label = list(label) if label is not None else []

    def copy_from(self, other: "BernoulliMixture") -> None:
        """Deep copy from another BernoulliMixture."""
        self.states = deepcopy(other.states)
        self.existence = deepcopy(other.existence)
        self.label = deepcopy(other.label)

    def size(self) -> int:
        """
        Returns total number of Bernoulli components (assumed to be along the 3rd axis of states).
        """
        if self.states is None or self.states.size == 0:
            return 0
        return self.states.shape[2]

class TrackerBP:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters

        # Extract parameters with consistent names.
        self.mu_n = parameters['mu_n']
        self.mu_c = parameters['mu_c']
        self.f_c = parameters['f_c']
        self.d_t = parameters['d_t']
        self.sensingRange = parameters['measurement_range']
        self.num_particles = parameters['num_particles']
        self.process_noise = parameters['sigma_v']
        self.p_s = parameters['p_s']
        self.num_sensors = parameters['num_sensors']
        self.detection_threshold = parameters['detection_threshold']
        self.pruning_threshold = parameters['pruning_threshold']
        self.num_steps = parameters['nun_steps']
        self.p_d = parameters['p_d']
        self.clutter_intensity = self.mu_c * self.f_c
        # Surveillance region computed using sensingRange.
        self.surveillanceRegion = 2 * np.pi * self.sensingRange ** 2
        self.birth_intensity = self.mu_n / self.surveillanceRegion
        # Sensor positions assumed to be a 2 x num_sensors array.
        self.sensor_positions = parameters['sensor_positions']
        self.var_range = parameters['range_variance']
        self.var_bearing = parameters['bearing_variance']

        # Initialize particles (using default sensor position if not provided).
        self.new_particles = self.initiate_particles([0, 0])
        self.likelihood_table = np.zeros((1, 0, self.num_particles))

        # State transition and noise matrices.
        self.F = np.array([[1, 0, self.d_t, 0],
                           [0, 1, 0, self.d_t],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.Q = np.array([[self.d_t ** 4 / 4, 0, self.d_t ** 3 / 2, 0],
                           [0, self.d_t ** 4 / 4, 0, self.d_t ** 3 / 2],
                           [self.d_t ** 3 / 2, 0, self.d_t ** 2, 0],
                           [0, self.d_t ** 3 / 2, 0, self.d_t ** 2]])

        # Initialize Bernoulli mixtures with existence and label as lists.
        self.alpha = BernoulliMixture(
            states=np.zeros((4, self.num_particles, 0)),
            existence=[],
            label=[]
        )
        self.varsigma = BernoulliMixture(
            states=np.zeros((4, self.num_particles, 0)),
            existence=[],
            label=[]
        )
        self.gamma = BernoulliMixture(
            states=np.zeros((4, self.num_particles, 0)),
            existence=[],
            label=[]
        )

        # Data Association messages (initialized as empty arrays).
        self.xi = np.array([])
        self.beta = np.array([])
        self.nu = np.array([])
        self.phi = np.array([])
        self.kappa = np.array([])
        self.iota = np.array([])

    def compute_alpha(self) -> None:
        """
        Predicts the target states (alpha) from the gamma mixture.
        """
        num_targets = self.gamma.states.shape[2]
        self.alpha.copy_from(self.gamma)

        for target in range(num_targets):
            # Process noise: sample process noise per particle and add to predicted states.
            predicted_states = self.F @ self.alpha.states[:, :, target]
            noise = np.random.multivariate_normal(
                np.zeros(predicted_states.shape[0]), self.Q, size=self.num_particles
            ).T
            self.alpha.states[:, :, target] = predicted_states + noise
            self.alpha.existence[target] = self.p_s * deepcopy(self.alpha.existence[target])
        self.gamma.copy_from(self.alpha)

    def compute_xi_sigma(self,
                         sensor_measurements: np.ndarray,
                         sensor_index: int) -> None:
        """
        Computes the xi message and updates the varsigma mixture using new sensor measurements.
        """
        num_measurements = sensor_measurements.shape[1] if sensor_measurements.size else 0
        # Access sensor position as a 2D column vector.
        sensor_position = self.sensor_positions[:, sensor_index]
        clutter_intensity = self.mu_c * self.f_c
        # Use measurement_range parameter if available; otherwise, use sensingRange.
        measurement_range = self.parameters.get('measurement_range', self.sensingRange)

        # Initialize new particles from the given sensor position.
        self.new_particles = self.initiate_particles(sensor_position)

        if num_measurements > 0:
            measurements_likelihood = self.calculate_likelihood_for_new_measurements(sensor_position,
                                                                                    sensor_measurements)
        else:
            measurements_likelihood = np.array([])

        # Allocate space for varsigma mixture and xi message.
        self.varsigma.states = np.empty((4, self.num_particles, num_measurements))
        self.varsigma.label = []
        self.varsigma.existence = []
        self.xi = np.empty((num_measurements,))

        for meas in range(num_measurements):
            self.varsigma.states[:, :, meas] = self.sample_from_likelihood(sensor_measurements[:, meas],
                                                                            sensor_position)
            current_max_label = max(self.gamma.label) if self.gamma.label else 0
            self.varsigma.label.append(current_max_label + meas + 1)
            self.varsigma.existence.append(measurements_likelihood[meas] *
                                             (self.mu_n) / clutter_intensity)
            self.xi[meas] = 1 + deepcopy(self.varsigma.existence[-1])

    def compute_beta(self,
                     sensor_measurements: np.ndarray,
                     sensor_index: int) -> None:
        """
        Evaluates measurement likelihood factors and computes the beta messages.
        """
        var_range = self.var_range
        var_bearing = self.var_bearing
        clutter_intensity = self.f_c * self.mu_c
        detection_probability = self.p_d
        sensor_positions = self.sensor_positions  # 2 x num_sensors
        num_particles = self.num_particles

        num_measurements = sensor_measurements.shape[1] if sensor_measurements.size else 0
        num_targets = self.gamma.states.shape[2] if self.gamma.states.size else 0

        # Initialize beta and likelihood table.
        self.beta = np.zeros((num_measurements + 1, num_targets))
        self.likelihood_table = np.zeros((num_measurements + 1, num_targets, num_particles))

        if num_targets > 0:
            # For missed detections.
            self.likelihood_table[0, :, :] = 1 - detection_probability

            for target in range(num_targets):
                # Compute predicted range and bearing for each particle.
                diff_x = self.gamma.states[0, :, target] - sensor_positions[0, sensor_index]
                diff_y = self.gamma.states[1, :, target] - sensor_positions[1, sensor_index]
                predicted_range = np.sqrt(diff_x ** 2 + diff_y ** 2)
                predicted_bearing = np.degrees(np.arctan2(diff_y, diff_x))

                for m in range(num_measurements):
                    bearing_error = self.wrap_to_180(sensor_measurements[1, m] - predicted_bearing)
                    # Compute likelihood for each particle.
                    self.likelihood_table[m + 1, target, :] = (
                        (1 / (2 * np.pi * np.sqrt(var_bearing * var_range))) *
                        (detection_probability / clutter_intensity) *
                        np.exp(-0.5 * ((sensor_measurements[0, m] - predicted_range) ** 2 / var_range)) *
                        np.exp(-0.5 * (bearing_error ** 2 / var_bearing))
                    )

            # v-factors for missed detections.
            v_factors = np.zeros((num_measurements + 1, num_targets))
            v_factors[0, :] = 1
            # Convert gamma.existence to numpy array for broadcasting.
            existence = np.tile(np.array(self.gamma.existence), (num_measurements + 1, 1))
            self.beta = existence * np.mean(self.likelihood_table, axis=2) + (1 - existence) * v_factors

    # Original compute_kappa_iota implementation (commented out)
    '''
    def compute_kappa_iota(self):
        """
        Performs iterative belief propagation for data association.
        """
        num_measurements = self.beta.shape[0] - 1
        num_targets = self.beta.shape[1]
        if num_targets == 0 or num_measurements == 0:
            self.kappa = self.beta
            self.iota = self.xi
            return

        # Initialize kappa as ones.
        self.kappa = np.ones((num_measurements, num_targets))
        for iteration in range(100):
            previous_kappa = deepcopy(self.kappa)
            product = self.kappa * self.beta[1:, :]
            likelihood_sum = self.beta[0, :] + np.sum(product, axis=0)
            denominator = np.tile(likelihood_sum, (num_measurements, 1)) - product
            messages = self.beta[1:, :] / (denominator + 1e-12)
            sum_messages = self.xi + np.sum(messages, axis=1)
            self.kappa = 1 / (np.tile(sum_messages[:, None], (1, num_targets)) - messages + 1e-12)
            # Check for convergence in log space.
            distance = np.max(np.abs(np.log(self.kappa + 1e-12) - np.log(previous_kappa + 1e-12)))
            if distance < 1e-5:
                break

        # Combine messages with a column of ones.
        self.iota = np.hstack((np.ones((num_measurements, 1)), messages))
        row_sums = np.sum(self.iota, axis=1, keepdims=True)
        self.iota = self.iota / (row_sums + 1e-12)
        # Extract the first column as the final iota values.
        self.iota = self.iota[:, 0]
    '''

    def compute_kappa_iota(self,
                           beta=None,
                           xi=None,
                           edge_mask=None,
                           max_iter=200,
                           tol=1e-8,
                           eps=1e-12,
                           return_full_posteriors=False):
        """
        SPADA message passing (sum–product on the bipartite DA subgraph) with a sparse edge mask.

        Inputs
        ------
        beta : (M+1, N) array or None
            Association weights β_j(0) and β_j(m):
              beta[0, j]   = β_j(0)          # miss-detection / no-meas for target j
              beta[m, j]   = β_j(m), m>=1    # weight that target j explains measurement m
            If None, uses self.beta (for backward compatibility with your original code).

        xi : (M,) or (M, N+1) array or None
            Measurement-side weights ξ_m(0) and optionally ξ_m(i):
              - If shape is (M,), we assume ξ_m(i>0) = 1 (common single-sensor case).
              - If shape is (M, N+1), we use xi[:,0] = ξ_m(0), xi[:,1:] = ξ_m(i).
            If None, uses self.xi (back-compat).

        edge_mask : (M, N) bool or None
            True iff edge Ψ^{i,m} exists. If None, tries self.edge_mask, else assumes fully-connected.

        max_iter, tol, eps : standard iteration/stop/numerical guards.

        Outputs (stored as attributes and optionally returned)
        ---------------------------------------------
        self.kappa : (M, N)        # ν(m→i) on edges; 0 on non-edges
        self.phi   : (M, N)        # φ(i→m) on edges; 0 on non-edges
        self.iota  : (M,)          # p̃(b_m = 0 | z)  (measurement-side posterior of 'no target')
        self.p_a   : (N, M+1)      # optional, if return_full_posteriors=True
        self.p_b   : (M, N+1)      # optional, if return_full_posteriors=True

        Notes (mapping to the paper / your figure)
        ------------------------------------------
        - This is the vectorized form of the SPADA ratios:
            φ_{i→m} = β_i(m) / [ β_i(0) + Σ_{m'≠m} β_i(m') ν_{m'→i} ]
            ν_{m→i} = ξ_m(i) / [ ξ_m(0) + Σ_{i'≠i} ξ_m(i') φ_{i'→m} ].
        - The **edge mask** enforces the graph in your drawing: only existing Ψ^{i,m}
          contribute to sums/updates. Non-edges never send/receive messages.
        - We form p̃(b_m) with ξ_m(0) and ξ_m(i) φ_{i→m} (fixes the “ones-column” issue).
        - If you want the target-side posteriors too, set return_full_posteriors=True.
        """
        # ---- Resolve inputs / legacy attributes --------------------------------
        if beta is None:
            if not hasattr(self, "beta"):
                raise ValueError("compute_kappa_iota: provide `beta` or set self.beta before calling.")
            beta = self.beta
        if xi is None:
            if not hasattr(self, "xi"):
                raise ValueError("compute_kappa_iota: provide `xi` or set self.xi before calling.")
            xi = self.xi
        beta = np.asarray(beta, dtype=float)
        xi   = np.asarray(xi,   dtype=float)

        M = beta.shape[0] - 1   # number of measurements
        N = beta.shape[1]       # number of targets
        if M < 0 or N == 0:
            # No associations possible
            self.kappa = np.zeros((max(M,0), N))
            self.phi   = np.zeros((max(M,0), N))
            self.iota  = np.ones((max(M,0),))
            if return_full_posteriors:
                self.p_a = np.zeros((N, max(M,0)+1))
                self.p_b = np.hstack([self.iota[:,None], np.zeros((max(M,0), N))])
                return self.p_a, self.p_b
            return self.kappa, self.iota

        beta0 = beta[0, :]        # (N,)
        beta_mi = beta[1:, :]     # (M, N)

        # ξ format: (M,) => ξ_m(0) supplied, ξ_m(i>0)=1 ; (M, N+1) => full
        if xi.ndim == 1:
            xi0  = xi.reshape(M)              # (M,)
            xiij = np.ones((M, N), float)     # (M, N)
        else:
            if xi.shape != (M, N+1):
                raise ValueError(f"`xi` must be (M,) or (M, N+1); got {xi.shape}")
            xi0  = xi[:, 0]
            xiij = xi[:, 1:]

        # Edge mask
        if edge_mask is None:
            if hasattr(self, "edge_mask") and self.edge_mask is not None:
                edge_mask = np.asarray(self.edge_mask, dtype=bool)
            else:
                edge_mask = np.ones((M, N), dtype=bool)
        else:
            edge_mask = np.asarray(edge_mask, dtype=bool)
        if edge_mask.shape != (M, N):
            raise ValueError(f"`edge_mask` must be shape (M, N) = {(M, N)}; got {edge_mask.shape}")

        # ---- Initialize messages on existing edges -----------------------------
        # φ^{(0)}(i→m) = β_i(m)/β_i(0)  (only where edge exists)
        phi   = np.zeros((M, N), float)
        valid_cols = (beta0 > 0)
        # Broadcast β_i(0) along M and then mask
        denom0 = beta0[None, :] + eps
        # Broadcast along M to match (M,N) before masked indexing
        denom0_full = np.broadcast_to(denom0, (M, N))
        mask0 = edge_mask & valid_cols[None, :]
        phi[mask0] = beta_mi[mask0] / denom0_full[mask0]

        # ν messages κ ≡ ν(m→i); start with ones on edges, 0 on non-edges
        kappa = np.zeros((M, N), float)
        kappa[edge_mask] = 1.0

        # ---- SPADA iterations (vectorized eqs. (30)–(31)) ---------------------
        if not edge_mask.any():
            # No edges ⇒ every measurement stays unclaimed
            self.kappa = kappa
            self.phi   = phi
            self.iota  = np.ones((M,), float)
            if return_full_posteriors:
                self.p_b = np.hstack([self.iota[:, None], np.zeros((M, N))])
                self.p_a = np.zeros((N, M+1))
                self.p_a[:, 0] = 1.0
                return self.p_a, self.p_b
            return self.kappa, self.iota

        for _ in range(max_iter):
            prev = kappa.copy()

            # --- φ update: φ_{i→m} = β_i(m) / ( β_i(0) + Σ_{m'≠m} β_i(m') ν_{m'→i} )
            prod = beta_mi * kappa                             # (M, N)
            sum_over_m = beta0 + np.sum(prod, axis=0)          # (N,)
            denom_phi = sum_over_m[None, :] - prod             # (M, N)
            # Only compute on edges; zero elsewhere
            phi_new = np.zeros_like(phi)
            mask_phi = edge_mask & (denom_phi > 0)
            phi_new[mask_phi] = beta_mi[mask_phi] / (denom_phi[mask_phi] + eps)
            phi = phi_new

            # --- ν update: ν_{m→i} = ξ_m(i) / ( ξ_m(0) + Σ_{i'≠i} ξ_m(i') φ_{i'→m} )
            sum_over_i = np.sum(xiij * phi * edge_mask, axis=1, keepdims=True)   # (M,1)
            denom_nu = xi0[:, None] + sum_over_i - (xiij * phi)                  # (M,N)
            kappa_new = np.zeros_like(kappa)
            mask_nu = edge_mask & (denom_nu > 0)
            kappa_new[mask_nu] = xiij[mask_nu] / (denom_nu[mask_nu] + eps)
            kappa = kappa_new

            # Convergence: log-space distance on existing edges
            delta = float(np.max(np.abs(np.log(kappa[edge_mask] + eps) - np.log(prev[edge_mask] + eps)))) if edge_mask.any() else 0.0
            if delta < tol:
                break

        # ---- Measurement posterior p̃(b_m)  (fixes earlier 'ones' bug) --------
        # Unnormalized scores: s0 = ξ_m(0),  si[:,i] = ξ_m(i) φ_{i→m} on edges
        s0 = xi0.copy()                           # (M,)
        si = (xiij * phi) * edge_mask            # (M, N)
        Zm = s0 + np.sum(si, axis=1)             # (M,)
        iota = s0 / (Zm + eps)                   # p̃(b_m = 0 | z)

        # Save API-compatible fields
        self.kappa = kappa
        self.phi   = phi
        self.iota  = iota

        if not return_full_posteriors:
            return self.kappa, self.iota

        # ---- (Optional) full posteriors over a^j and b^m ----------------------
        # Target-side: p̃(a^j = m) ∝ β_j(m) ν_{m→j},  p̃(a^j = 0) ∝ β_j(0)
        num_m = beta_mi * kappa                 # (M, N)
        Zj = beta0 + np.sum(num_m, axis=0)      # (N,)
        p_a = np.zeros((N, M+1), float)
        p_a[:, 0]  = beta0 / (Zj + eps)
        p_a[:, 1:] = (num_m / (Zj[None, :] + eps)).T

        # Measurement-side: p̃(b^m = 0) and p̃(b^m = i) with masking
        p_b = np.zeros((M, N+1), float)
        p_b[:, 0]  = iota
        p_b[:, 1:] = si / (Zm[:, None] + eps)
        # Zero impossible edges and renormalize each row of p_b
        p_b[:, 1:][~edge_mask] = 0.0
        p_b /= (np.sum(p_b, axis=1, keepdims=True) + eps)

        self.p_a = p_a
        self.p_b = p_b
        return p_a, p_b

    def compute_gamma(self):
        """
        Updates legacy potential tracks and merges them with new tracks.
        """
        num_measurements = self.likelihood_table.shape[0] - 1
        num_particles = self.likelihood_table.shape[2]
        num_targets = self.likelihood_table.shape[1]
  
        for target in range(num_targets):
            # missed detection
            weights = deepcopy(self.likelihood_table[0, target, :])
            # Sum over all measurements.
            for m in range(num_measurements):
                weights += self.kappa[m, target] * self.likelihood_table[m+1, target, :]
            # Sum of the weights of all particles.
            sum_weights = np.sum(weights)
            isAlive = self.gamma.existence[target] * sum_weights / num_particles
            isDead = 1 - self.gamma.existence[target]
            self.gamma.existence[target] = isAlive / (isAlive + isDead + 1e-12)
            if sum_weights > 0:
                norm_weights = weights / sum_weights
                indices = TrackerBP.resample(norm_weights, num_particles)
                self.gamma.states[:, :, target] = deepcopy(self.gamma.states[:, indices, target])
    
        # Merge legacy tracks with new tracks.
        if self.gamma.states.size == 0:
            merged_states = self.varsigma.states
        else:
            merged_states = np.concatenate((self.gamma.states, self.varsigma.states), axis=2)
        self.gamma.states = merged_states
    
        # Update existence for new tracks.
        new_existences = (self.iota * np.array(self.varsigma.existence)) / (self.iota * np.array(self.varsigma.existence) + 1 + 1e-12)
        if not self.gamma.existence:
            merged_existences = new_existences.tolist()
        else:
            merged_existences = self.gamma.existence + new_existences.tolist()
        self.gamma.existence = merged_existences
    
        # Merge labels.
        if not self.gamma.label:
            merged_labels = self.varsigma.label
        else:
            merged_labels = self.gamma.label + self.varsigma.label
        self.gamma.label = merged_labels

    def prune(self):
        """
        Prunes tracks with low existence probability.
        """
        if self.gamma.size() > 0:
            valid_idx = np.array([e >= self.pruning_threshold for e in self.gamma.existence])
            if valid_idx.any():
                self.gamma.states = self.gamma.states[:, :, valid_idx]
                self.gamma.label = np.array(self.gamma.label)[valid_idx].tolist()
                self.gamma.existence = np.array(self.gamma.existence)[valid_idx].tolist()
            else:
                self.gamma.states = np.empty((4, self.num_particles, 0))
                self.gamma.label = []
                self.gamma.existence = []

    def estimate_state(self):
        """
        Estimates the state for targets with existence probability above the detection threshold.
        """
        estimates = {}
        detected_states = []
        detected_labels = []
        detected_existence = []

        if self.gamma.states.shape[2] > 0:
            numTargets = self.gamma.states.shape[2]
            for target in range(numTargets):
                if self.gamma.existence[target] > self.detection_threshold:
                    state_mean = np.mean(self.gamma.states[:, :, target], axis=1)
                    detected_states.append(state_mean)
                    detected_labels.append(self.gamma.label[target])
                    detected_existence.append(self.gamma.existence[target])
        if detected_states:
            estimates['state'] = np.column_stack(detected_states)
            estimates['label'] = np.array(detected_labels)
            estimates['existence'] = np.array(detected_existence)
        estimates['gamma'] = deepcopy(self.gamma)
        return estimates

    def estimate_cardinality(self):
        """
        Estimates the number of targets (cardinality) as the sum of existence probabilities.
        """
        if self.gamma.existence:
            estimated_cardinality = np.sum(self.gamma.existence)
        else:
            estimated_cardinality = 0
        return estimated_cardinality

    def calculate_likelihood_for_new_measurements(self, sensor_position: np.ndarray, sensor_measurements: np.ndarray) -> np.ndarray:
        # Ensure inputs are numpy arrays of type float
        sensor_measurements = np.asarray(sensor_measurements, dtype=float)
        sensor_position = np.asarray(sensor_position, dtype=float)
        self.new_particles = np.asarray(self.new_particles, dtype=float)
        
        measurement_range = self.parameters['measurement_range']
        var_range = float(self.parameters['range_variance'])
        var_bearing = float(self.parameters['bearing_variance'])
                
        # Compute predicted sensor readings for each particle.
        dx = self.new_particles[0, :] - sensor_position[0]
        dy = self.new_particles[1, :] - sensor_position[1]
        predicted_range = np.sqrt(dx**2 + dy**2)
        predicted_bearing = np.degrees(np.arctan2(dy, dx))
        
        # Reshape for broadcasting.
        predicted_range = predicted_range[:, np.newaxis]
        predicted_bearing = predicted_bearing[:, np.newaxis]
        
        # Compute differences.
        range_diff = sensor_measurements[0, np.newaxis, :] - predicted_range
        bearing_diff = sensor_measurements[1, np.newaxis, :] - predicted_bearing
        
        # Compute likelihood (vectorized operations).
        likelihood_particles = (1 / (2 * np.pi * np.sqrt(var_range * var_bearing))) * \
            np.exp(-0.5 * (range_diff**2 / var_range)) * \
            np.exp(-0.5 * (bearing_diff**2 / var_bearing))
        
        likelihood_measurements = np.mean(likelihood_particles, axis=0)
        return likelihood_measurements

    def sample_from_likelihood(self, sensor_measurement: np.ndarray, sensor_position: np.ndarray) -> np.ndarray:
        """
        Samples new potential target states from the measurement likelihood.
        """
        var_range = self.parameters['range_variance']
        var_bearing = self.parameters['bearing_variance']
        priorVelocityCov = self.parameters['velocity_noise']
        num_particles = self.parameters['num_particles']
    
        samples = np.zeros((4, num_particles))
        randomRange = sensor_measurement[0] + np.sqrt(var_range) * np.random.randn(num_particles)
        randomBearing = sensor_measurement[1] + np.sqrt(var_bearing) * np.random.randn(num_particles)
        samples[0, :] = sensor_position[0] + randomRange * np.cos(np.deg2rad(randomBearing))
        samples[1, :] = sensor_position[1] + randomRange * np.sin(np.deg2rad(randomBearing))
        L = cholesky(priorVelocityCov, lower=True)
        samples[2:4, :] = L @ np.random.randn(2, num_particles)
        return samples

    @staticmethod
    def resample(weights: np.ndarray, num_particles: int) -> np.ndarray:
        """
        Systematic resampling algorithm.
        """
        cumWeights = np.cumsum(weights)
        positions = (np.arange(num_particles) + np.random.uniform(0, 1)) / num_particles
        indexes = np.zeros(num_particles, dtype=int)
        i, j = 0, 0
        while i < num_particles:
            if positions[i] < cumWeights[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    @staticmethod
    def wrap_to_180(angle: Any) -> Any:
        """
        Wraps an angle (in degrees) to the interval [-180, 180].
        """
        return ((angle + 180) % 360) - 180

    def initiate_particles(self, sensor_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Initializes particles around a sensor position.
        """
        particles = np.zeros((2, self.parameters['num_particles']))
        theta = 360 * np.random.rand(self.parameters['num_particles'])
        r = self.parameters['measurement_range'] * np.sqrt(np.random.rand(self.parameters['num_particles']))
        particles[0, :] = sensor_position[0] + r * np.cos(np.deg2rad(theta))
        particles[1, :] = sensor_position[1] + r * np.sin(np.deg2rad(theta))
        return particles
