"""Tracker variants that record particle states for particle-cloud visualization."""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from TrackerBP import TrackerBP


class TrackerBPParticleHistory(TrackerBP):
    """Particle BP tracker that keeps a history of particle states per step.

    Snapshots are captured after each call to prune(), so the stored states
    reflect post-pruning gamma mixtures.
    """

    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.particle_history: List[Dict[str, object]] = []
        self.particle_count_history: List[int] = []
        self._step_counter = 0

    def prune(self) -> None:
        """Wrap base prune, then snapshot particle states for visualization."""
        super().prune()
        self._store_particle_snapshot()

    def _store_particle_snapshot(self) -> None:
        """Deep-copy the current gamma mixture for later visualization."""
        snapshot = {
            "step": self._step_counter,
            "states": deepcopy(self.gamma.states),
            "existence": deepcopy(self.gamma.existence),
            "label": deepcopy(self.gamma.label),
        }
        count = 0
        if snapshot["states"] is not None and snapshot["states"].size > 0:
            num_particles = snapshot["states"].shape[1]
            existence = np.asarray(snapshot["existence"], dtype=float)
            if existence.size:
                active_tracks = np.count_nonzero(existence >= self.detection_threshold)
                count = int(num_particles * active_tracks)
        self.particle_history.append(snapshot)
        self.particle_count_history.append(count)
        self._step_counter += 1

    def get_particle_clouds(self,
                            history_index: int,
                            threshold: Optional[float] = None) -> List[Tuple[Any, np.ndarray]]:
        """Return per-track particle clouds for tracks above the existence threshold.

        Returns a list of tuples: (track_label, XY_positions[N,2]).
        """
        snapshot = self.particle_history[history_index]
        states = snapshot["states"]
        if states is None or states.size == 0:
            return []

        existence = np.asarray(snapshot["existence"], dtype=float)
        if existence.size == 0:
            return []

        if threshold is None:
            threshold = self.detection_threshold

        valid = existence >= float(threshold)
        if not np.any(valid):
            return []

        filtered_states = states[:, :, valid]
        labels_raw = snapshot.get("label", [])
        if labels_raw:
            labels_array = np.asarray(labels_raw, dtype=object)[valid]
        else:
            num_tracks = filtered_states.shape[2]
            labels_array = np.arange(num_tracks)

        clouds: List[Tuple[Any, np.ndarray]] = []
        for track_idx in range(filtered_states.shape[2]):
            xy = filtered_states[0:2, :, track_idx].T
            label_value: Any = labels_array[track_idx] if labels_array.size > track_idx else track_idx
            if isinstance(label_value, np.generic):
                label_value = label_value.item()
            clouds.append((label_value, xy))
        return clouds

    def get_particle_positions(self, history_index: int, threshold: Optional[float] = None) -> np.ndarray:
        """Return flattened XY positions for tracks above the given existence threshold."""
        clouds = self.get_particle_clouds(history_index, threshold)
        if not clouds:
            return np.empty((0, 2))
        positions = [cloud[1] for cloud in clouds if cloud[1].size]
        if not positions:
            return np.empty((0, 2))
        return np.vstack(positions)
