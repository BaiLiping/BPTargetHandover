"""Generate per-frame particle cloud visualizations from a saved trial."""

import os
import pickle
from copy import deepcopy
from typing import Iterable, List, Optional, Sequence

import matplotlib

# Headless backend; saves PNGs without opening a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import imageio.v2 as imageio  # imageio v3 alias
except Exception:  # pragma: no cover
    imageio = None

from TrackerBP_with_history import TrackerBPParticleHistory


def _ensure_measurement_matrix(raw_measurements: Sequence) -> np.ndarray:
    """Return a (2, N) measurement matrix regardless of input layout."""
    arr = np.asarray(raw_measurements, dtype=float)
    if arr.size == 0:
        return np.empty((2, 0))
    if arr.ndim == 1:
        if arr.shape[0] == 0:
            return np.empty((2, 0))
        if arr.shape[0] == 2:
            return arr.reshape(2, 1)
        return arr.reshape(1, -1)
    if arr.shape[0] == 2:
        return arr
    if arr.shape[1] == 2:
        return arr.T
    raise ValueError("Unexpected measurement shape; unable to coerce to (2, N).")


def _compute_axis_limits(trackers: Sequence[TrackerBPParticleHistory], params: dict) -> List[float]:
    """Compute consistent axis limits across sensors to avoid flicker."""
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    for tracker in trackers:
        for snapshot in tracker.particle_history:
            states = snapshot["states"]
            if states is None or states.size == 0:
                continue
            xy = states[0:2, :, :].reshape(2, -1)
            x_min = min(x_min, float(np.min(xy[0])))
            x_max = max(x_max, float(np.max(xy[0])))
            y_min = min(y_min, float(np.min(xy[1])))
            y_max = max(y_max, float(np.max(xy[1])))
    if not np.isfinite([x_min, x_max, y_min, y_max]).all():
        positions = params.get("sensor_positions", np.zeros((2, 1)))
        sensing_range = params.get("measurement_range", 100.0)
        x_min = float(np.min(positions[0]) - sensing_range)
        x_max = float(np.max(positions[0]) + sensing_range)
        y_min = float(np.min(positions[1]) - sensing_range)
        y_max = float(np.max(positions[1]) + sensing_range)
    padding = 0.05 * max(x_max - x_min, y_max - y_min, 1.0)
    return [x_min - padding, x_max + padding, y_min - padding, y_max + padding]


def generate_particle_visualization(trial_path: str,
                                    output_folder: str,
                                    sensor_indices: Optional[Iterable[int]] = None,
                                    gif_path: Optional[str] = None,
                                    frame_duration: float = 0.3) -> None:
    """Replay a trial file and export particle-cloud frames for each step.

    - trial_path: path to a trial_XXXX.pkl saved by centralized/distributed/handover runs
    - output_folder: where to write per-step PNG frames
    - sensor_indices: optional subset of sensors (0-based). Defaults to all.
    - gif_path: optional GIF path; requires imageio to be installed
    - frame_duration: seconds per frame for the GIF
    """
    with open(trial_path, "rb") as f:
        trial_data = pickle.load(f)

    params = deepcopy(trial_data["parameters"])
    n_steps = int(params["n_steps"])
    measurements = trial_data["measurements"]
    num_sensors_total = int(params["num_sensors"])

    if sensor_indices is None:
        sensor_indices = list(range(num_sensors_total))
    else:
        sensor_indices = list(sensor_indices)

    trackers: List[TrackerBPParticleHistory] = []
    for _ in sensor_indices:
        trackers.append(TrackerBPParticleHistory(deepcopy(params)))

    existence_threshold = float(params.get("detection_threshold", 0.0))

    # Replay the trial to collect particle snapshots
    for step in range(n_steps):
        for idx, sensor_id in enumerate(sensor_indices):
            tracker = trackers[idx]
            tracker.compute_alpha()
            meas = _ensure_measurement_matrix(measurements[step][sensor_id])
            tracker.compute_xi_sigma(meas, sensor_id)
            tracker.compute_beta(meas, sensor_id)
            tracker.compute_kappa_iota()
            tracker.compute_gamma()
            tracker.prune()

    os.makedirs(output_folder, exist_ok=True)
    axis_limits = _compute_axis_limits(trackers, params)

    frame_paths: List[str] = []
    # Use a high-contrast, fixed color list (start with magenta)
    color_palette_list = [
        '#FF00FF', '#1E90FF', '#FFA500', '#00FF00', '#FF0000',
        '#8A2BE2', '#00CED1', '#FFD700', '#ADFF2F', '#4B0082',
        '#00BFFF', '#B22222', '#8B4513', '#2E8B57', '#4682B4',
        '#DA70D6', '#D2691E', '#9ACD32', '#FF69B4', '#20B2AA'
    ]
    track_color_maps: List[dict] = [dict() for _ in trackers]

    # Visualization helpers for field-of-view overlays
    sensor_positions = params.get("sensor_positions", None)
    sensing_range = float(params.get("measurement_range", 0.0))

    for step in range(n_steps):
        fig, axes = plt.subplots(1, len(sensor_indices), figsize=(5 * len(sensor_indices), 5), squeeze=False)
        total_particles = 0
        for ax, tracker, sensor_id, color_map in zip(axes[0], trackers, sensor_indices, track_color_maps):
            # Draw sensor field-of-view (circle) and sensor location
            if isinstance(sensor_positions, np.ndarray) and sensor_positions.ndim == 2 \
               and sensor_positions.shape[0] == 2 and sensor_id < sensor_positions.shape[1] \
               and sensing_range > 0:
                cx, cy = float(sensor_positions[0, sensor_id]), float(sensor_positions[1, sensor_id])
                circ = plt.Circle((cx, cy), sensing_range, color="0.4", fill=False,
                                  linestyle="--", linewidth=1.0, alpha=0.6)
                ax.add_patch(circ)
                ax.plot([cx], [cy], marker="x", color="0.3", markersize=6, mew=1.0)

            clouds = tracker.get_particle_clouds(step, existence_threshold)
            sensor_particle_count = 0
            if clouds:
                for label, positions in clouds:
                    if positions.size == 0:
                        continue
                    sensor_particle_count += positions.shape[0]
                    color_key = label
                    if isinstance(color_key, np.generic):
                        color_key = color_key.item()
                    if color_key not in color_map:
                        color_map[color_key] = color_palette_list[len(color_map) % len(color_palette_list)]
                    ax.scatter(positions[:, 0], positions[:, 1], s=4, alpha=0.35,
                               color=color_map[color_key], edgecolors="none")
            else:
                ax.text(0.5, 0.5, "No particles", transform=ax.transAxes, ha="center", va="center", color="0.5")
            total_particles += sensor_particle_count
            ax.set_xlim(axis_limits[0], axis_limits[1])
            # Fix y-axis to [-130, 130] as requested
            ax.set_ylim(-130.0, 130.0)
            ax.set_aspect("equal", adjustable="box")
            # Show axes and a light grid
            ax.axhline(0.0, color="0.8", linewidth=0.6, zorder=0)
            ax.axvline(0.0, color="0.8", linewidth=0.6, zorder=0)
            ax.grid(True, which="both", alpha=0.15, linestyle=":", linewidth=0.6)
            ax.set_title(f"Sensor {sensor_id + 1} – Tracks: {len(clouds)} – Particles: {sensor_particle_count}")
            for spine in ax.spines.values():
                spine.set_alpha(0.4)
        fig.suptitle(f"Particle Visualization – Step {step + 1}/{n_steps} | Total: {total_particles}")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        frame_path = os.path.join(output_folder, f"particle_visualization_step_{step:04d}.png")
        fig.savefig(frame_path, dpi=150)
        plt.close(fig)
        frame_paths.append(frame_path)

    print(f"Saved {n_steps} particle visualization frames to {output_folder}")

    if gif_path:
        if imageio is None:
            print("imageio not available; skipping GIF generation.")
        elif not frame_paths:
            print("No frames available; skipping GIF generation.")
        else:
            images = [imageio.imread(path) for path in frame_paths]
            imageio.mimsave(gif_path, images, duration=frame_duration)
            print(f"Particle visualization GIF saved to {gif_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate particle-cloud frames from a trial pickle file.")
    parser.add_argument("trial", help="Path to trial_XXXX.pkl containing measurements and parameters")
    parser.add_argument("--output", default=os.path.join("Visualization", "particle_visualization"),
                        help="Destination folder for frames")
    parser.add_argument("--sensors", nargs="*", type=int, default=None,
                        help="Optional subset of sensor indices (0-based)")
    parser.add_argument("--gif", help="Optional path for the stitched GIF", default=None)
    parser.add_argument("--frame-duration", type=float, default=0.3,
                        help="Frame duration (seconds) for the GIF")
    args = parser.parse_args()

    generate_particle_visualization(args.trial,
                                    args.output,
                                    args.sensors,
                                    args.gif,
                                    args.frame_duration)
