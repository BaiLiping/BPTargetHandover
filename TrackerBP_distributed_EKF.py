# ------------------------------------------------------------------------
# BP Target Handover
# Copyright (c) 2025 Liping Bai. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import numpy as np
from scipy.linalg import cholesky
from copy import deepcopy
from typing import Optional, Any

class BernoulliMixture:
    """
    Holds the states, existence probabilities, and labels for a multi-Bernoulli mixture.
    For EKF, states are represented by mean and covariance.
    """
    def __init__(self,
                 mean: Optional[list] = None,
                 covariance: Optional[list] = None,
                 existence: Optional[list] = None,
                 label: Optional[list] = None) -> None:
        self.mean = mean if mean is not None else []
        self.covariance = covariance if covariance is not None else []
        self.existence = existence if existence is not None else []
        self.label = label if label is not None else []

    def copy_from(self, other: "BernoulliMixture") -> None:
        """Deep copy from another BernoulliMixture."""
        self.mean = deepcopy(other.mean)
        self.covariance = deepcopy(other.covariance)
        self.existence = deepcopy(other.existence)
        self.label = deepcopy(other.label)

    def size(self) -> int:
        """
        Returns total number of Bernoulli components.
        """
        return len(self.mean)

class TrackerBP_EKF:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters

        # Extract parameters with consistent names.
        self.mu_n = parameters['mu_n']
        self.mu_c = parameters['mu_c']
        self.f_c = parameters['f_c']
        self.d_t = parameters['d_t']
        self.sensingRange = parameters['measurement_range']
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
        self.velocity_noise_cov = parameters['velocity_noise']
        # Optional explicit initial covariance for new tracks (4x4). If provided, overrides
        # measurement-derived initialization in init_new_track_from_measurement.
        init_cov_param = parameters.get('init_cov', None)
        if init_cov_param is not None:
            self.init_cov = np.asarray(init_cov_param, dtype=float)
            if self.init_cov.shape != (4, 4):
                raise ValueError("init_cov must be 4x4 when provided")
        else:
            self.init_cov = None

        # State transition and noise matrices.
        self.F = np.array([[1, 0, self.d_t, 0],
                           [0, 1, 0, self.d_t],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.Q = self.process_noise**2 * np.array([[self.d_t ** 4 / 4, 0, self.d_t ** 3 / 2, 0],
                                                   [0, self.d_t ** 4 / 4, 0, self.d_t ** 3 / 2],
                                                   [self.d_t ** 3 / 2, 0, self.d_t ** 2, 0],
                                                   [0, self.d_t ** 3 / 2, 0, self.d_t ** 2]])

        # Initialize Bernoulli mixtures with mean and covariance for EKF.
        self.alpha = BernoulliMixture()
        self.varsigma = BernoulliMixture()
        self.gamma = BernoulliMixture()

        # Data Association messages (initialized as empty arrays).
        self.xi = np.array([])
        self.beta = np.array([])
        self.nu = np.array([])
        self.phi = np.array([])
        self.kappa = np.array([])
        self.iota = np.array([])

        # a tag for valid target
        self.valid_target_indices = []
        self.updated_gamma_for_beta = BernoulliMixture()

    def compute_alpha(self, sensor_index: int) -> None:
        """
        Predicts the target states (alpha) from the gamma mixture using EKF prediction step.
        """
        self.alpha.copy_from(self.gamma)
        num_targets = self.alpha.size()
        base_station_position = self.sensor_positions[:, sensor_index]

        for i in range(num_targets):
            # EKF prediction for mean and covariance
            self.alpha.mean[i] = self.F @ self.alpha.mean[i]
            self.alpha.covariance[i] = self.F @ self.alpha.covariance[i] @ self.F.T + self.Q
            
            # Update existence probability based only on survival probability
            # (no range gating in EKF)
            self.alpha.existence[i] *= self.p_s
        
        self.gamma.copy_from(self.alpha)

    def compute_xi_sigma(self,
                         sensor_measurements: np.ndarray,
                         sensor_index: int) -> None:
        """
        Computes the xi message and initializes new tracks (varsigma) for new measurements.
        """
        num_measurements = sensor_measurements.shape[1] if sensor_measurements.size > 0 else 0
        sensor_position = self.sensor_positions[:, sensor_index]
        
        self.varsigma = BernoulliMixture(
            mean=[None] * num_measurements,
            covariance=[None] * num_measurements,
            existence=[0.0] * num_measurements,
            label=[0] * num_measurements
        )
        self.xi = np.zeros(num_measurements)

        if num_measurements == 0:
            return

        for m in range(num_measurements):
            mean, cov = self.init_new_track_from_measurement(sensor_measurements[:, m], sensor_position)
            self.varsigma.mean[m] = mean
            self.varsigma.covariance[m] = cov

            current_max_label = max(self.gamma.label) if self.gamma.label else 0
            self.varsigma.label[m] = current_max_label + m + 1

            # Simplified existence probability for new tracks
            self.varsigma.existence[m] = self.birth_intensity * self.p_d / self.clutter_intensity
            self.xi[m] = 1.0 + self.varsigma.existence[m]

    def compute_beta(self,
                     sensor_measurements: np.ndarray,
                     sensor_index: int) -> None:
        """
        Evaluates measurement likelihood factors and computes the beta messages for EKF.
        """
        num_measurements = sensor_measurements.shape[1] if sensor_measurements.size > 0 else 0
        sensor_pos = self.sensor_positions[:, sensor_index]
        
        # No measurement gating: consider all targets, regardless of range
        self.valid_target_indices = list(range(len(self.gamma.mean)))
        
        num_valid_targets = len(self.valid_target_indices)
        self.beta = np.zeros((num_measurements + 1, num_valid_targets))
        
        # For EKF update, we need to store updated states for each measurement association
        self.updated_gamma_for_beta = BernoulliMixture(
            mean=[[None] * num_valid_targets for _ in range(num_measurements + 1)],
            covariance=[[None] * num_valid_targets for _ in range(num_measurements + 1)]
        )

        if num_valid_targets == 0:
            return

        # Missed detection (m=0)
        self.beta[0, :] = 1 - self.p_d
        for idx, target_idx in enumerate(self.valid_target_indices):
            self.updated_gamma_for_beta.mean[0][idx] = self.gamma.mean[target_idx]
            self.updated_gamma_for_beta.covariance[0][idx] = self.gamma.covariance[target_idx]

        # Detections (m > 0)
        for m in range(num_measurements):
            measurement = sensor_measurements[:, m]
            for idx, target_idx in enumerate(self.valid_target_indices):
                mean = self.gamma.mean[target_idx]
                cov = self.gamma.covariance[target_idx]
                
                z_pred, H, S = self.get_measurement_prediction(mean, cov, sensor_pos)
                
                innovation = measurement - z_pred
                innovation[1] = self.wrap_to_180(innovation[1])
                
                try:
                    S_inv = np.linalg.inv(S)
                    det_S = np.linalg.det(S)
                    
                    likelihood = (1 / (2 * np.pi * np.sqrt(det_S))) * \
                                 np.exp(-0.5 * innovation.T @ S_inv @ innovation)
                    
                    self.beta[m + 1, idx] = self.p_d * likelihood / self.clutter_intensity
                    
                    # EKF update
                    K = cov @ H.T @ S_inv
                    updated_mean = mean + K @ innovation
                    updated_cov = (np.eye(4) - K @ H) @ cov
                    
                    self.updated_gamma_for_beta.mean[m + 1][idx] = updated_mean
                    self.updated_gamma_for_beta.covariance[m + 1][idx] = updated_cov

                except np.linalg.LinAlgError:
                    self.beta[m + 1, idx] = 0

        # Multiply by existence probability
        existence_array = np.array([self.gamma.existence[i] for i in self.valid_target_indices])
        self.beta *= existence_array

    # Original compute_kappa_iota implementation (commented out)
    '''
    def compute_kappa_iota(self):
        """
        Performs iterative belief propagation for data association.
        """
        num_measurements = self.beta.shape[0] - 1
        num_targets = self.beta.shape[1]
        if num_targets == 0 or num_measurements == 0:
            self.kappa = np.ones((num_measurements, num_targets))
            self.iota = self.xi if self.xi.size > 0 else np.array([])
            if self.iota.ndim == 1 and self.iota.size > 0:
                 self.iota = self.iota / (1 + self.iota)
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
        if num_measurements > 0:
            self.iota = np.hstack((np.ones((num_measurements, 1)), messages))
            row_sums = np.sum(self.iota, axis=1, keepdims=True)
            self.iota = self.iota / (row_sums + 1e-12)
            # Extract the first column as the final iota values.
            self.iota = self.iota[:, 0]
        else:
            self.iota = np.array([])
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
        num_measurements = self.beta.shape[0] - 1
        num_targets = len(self.valid_target_indices)

        # Update existing tracks
        for idx, target_idx in enumerate(self.valid_target_indices):
            # Posterior existence probability
            missed_detection_term = (1 - self.p_d) * self.gamma.existence[target_idx]
            
            detection_terms = 0
            if num_measurements > 0:
                detection_terms = np.sum(self.kappa[:, idx] * self.beta[1:, idx])

            # Debug print for md (missed detection term) and det (detection terms)
            try:
                lbl = self.gamma.label[target_idx] if target_idx < len(self.gamma.label) else target_idx
            except Exception:
                lbl = target_idx
            sensor_ctx = getattr(self, "_debug_sensor_id", None)
            step_ctx = getattr(self, "_debug_step", None)
            ctx_parts = []
            if sensor_ctx is not None:
                ctx_parts.append(f"s={sensor_ctx}")
            if step_ctx is not None:
                ctx_parts.append(f"t={step_ctx}")
            ctx = (" "+" ".join(ctx_parts)) if ctx_parts else ""
            try:
                det_val = float(detection_terms)
            except Exception:
                det_val = 0.0
            print(f"[EKF][compute_gamma]{ctx} label={lbl} md={missed_detection_term:.6g} det={det_val:.6g}")

            denominator = (1 - self.gamma.existence[target_idx]) + missed_detection_term + detection_terms
            self.gamma.existence[target_idx] = (missed_detection_term + detection_terms) / (denominator + 1e-12)

            # Gaussian mixture reduction for updated state
            weights = np.zeros(num_measurements + 1)
            weights[0] = missed_detection_term
            if num_measurements > 0:
                weights[1:] = self.kappa[:, idx] * self.beta[1:, idx]
            
            weights /= (np.sum(weights) + 1e-12)

            # Merge Gaussians
            merged_mean = np.zeros(4)
            for i in range(num_measurements + 1):
                if self.updated_gamma_for_beta.mean[i][idx] is not None:
                    merged_mean += weights[i] * self.updated_gamma_for_beta.mean[i][idx]
            
            merged_cov = np.zeros((4, 4))
            for i in range(num_measurements + 1):
                 if self.updated_gamma_for_beta.mean[i][idx] is not None:
                    mean_diff = self.updated_gamma_for_beta.mean[i][idx] - merged_mean
                    merged_cov += weights[i] * (self.updated_gamma_for_beta.covariance[i][idx] + np.outer(mean_diff, mean_diff))

            self.gamma.mean[target_idx] = merged_mean
            self.gamma.covariance[target_idx] = merged_cov

        # Add new tracks from varsigma
        if self.varsigma.size() > 0 and self.iota.size > 0:
            new_existences = (self.iota * np.array(self.varsigma.existence)) / (1 + self.iota * np.array(self.varsigma.existence) + 1e-12)
            self.gamma.mean.extend(self.varsigma.mean)
            self.gamma.covariance.extend(self.varsigma.covariance)
            self.gamma.existence.extend(new_existences.tolist())
            self.gamma.label.extend(self.varsigma.label)

    def prune(self):
        """
        Prunes tracks with low existence probability.
        """
        if self.gamma.size() > 0:
            valid_idx = [e >= self.pruning_threshold for e in self.gamma.existence]
            if any(valid_idx):
                self.gamma.mean = [m for m, v in zip(self.gamma.mean, valid_idx) if v]
                self.gamma.covariance = [c for c, v in zip(self.gamma.covariance, valid_idx) if v]
                self.gamma.label = [l for l, v in zip(self.gamma.label, valid_idx) if v]
                self.gamma.existence = [e for e, v in zip(self.gamma.existence, valid_idx) if v]
            else:
                self.gamma = BernoulliMixture()

    def estimate_state(self):
        """
        Estimates the state for targets with existence probability above the detection threshold.
        """
        estimates = {}
        detected_means = []
        detected_labels = []
        detected_existence = []

        if self.gamma.size() > 0:
            for i in range(self.gamma.size()):
                if self.gamma.existence[i] > self.detection_threshold:
                    detected_means.append(self.gamma.mean[i])
                    detected_labels.append(self.gamma.label[i])
                    detected_existence.append(self.gamma.existence[i])
        
        if detected_means:
            estimates['state'] = np.array(detected_means).T
            estimates['label'] = np.array(detected_labels)
            estimates['existence'] = np.array(detected_existence)
        
        estimates['gamma'] = self.gamma
        return estimates

    def estimate_cardinality(self):
        """
        Estimates the number of targets (cardinality) as the sum of existence probabilities.
        """
        if len(self.gamma.existence) > 0:
            return np.sum(self.gamma.existence)
        return 0

    def init_new_track_from_measurement(self, measurement: np.ndarray, sensor_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Initializes a new track (mean and covariance) from a measurement."""
        r = measurement[0]
        theta = np.deg2rad(measurement[1])
        
        mean = np.zeros(4)
        mean[0] = sensor_pos[0] + r * np.cos(theta)
        mean[1] = sensor_pos[1] + r * np.sin(theta)
        
        # Initial covariance: prefer explicit parameter init_cov if provided
        if self.init_cov is not None:
            cov = self.init_cov.copy()
        else:
            G = np.array([[np.cos(theta), -r * np.sin(theta)],
                          [np.sin(theta),  r * np.cos(theta)]])
            R_polar = np.diag([self.var_range, np.deg2rad(self.var_bearing)**2])
            R_cartesian = G @ R_polar @ G.T
            cov = np.zeros((4, 4))
            cov[:2, :2] = R_cartesian
            cov[2:, 2:] = self.velocity_noise_cov
        
        return mean, cov

    def get_measurement_prediction(self, mean: np.ndarray, cov: np.ndarray, sensor_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts measurement, computes Jacobian H and innovation covariance S."""
        dx = mean[0] - sensor_pos[0]
        dy = mean[1] - sensor_pos[1]
        
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        z_pred = np.array([r, np.rad2deg(theta)])
        
        H = np.zeros((2, 4))
        if r > 1e-6:
            H[0, 0] = dx / r
            H[0, 1] = dy / r
            H[1, 0] = -dy / (r**2)
            H[1, 1] = dx / (r**2)
        
        H[1,:] = np.rad2deg(H[1,:])

        R = np.diag([self.var_range, self.var_bearing])
        S = H @ cov @ H.T + R
        
        return z_pred, H, S

    @staticmethod
    def wrap_to_180(angle: Any) -> Any:
        """
        Wraps an angle (in degrees) to the interval [-180, 180].
        """
        return ((angle + 180) % 360) - 180

    # Note: EKF birth likelihood helper removed; using simplified birth existence in compute_xi_sigma.
