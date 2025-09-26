"""Visualize EKF tracking by plotting covariance ellipses instead of particles.

Supports centralized (single tracker processes all sensors per step) and
distributed/handover (one tracker per selected sensor).
"""

import os
import pickle
from copy import deepcopy
from typing import Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from TrackerBP_centralized_EKF import TrackerBP_EKF as CentralTracker
from TrackerBP_distributed_EKF import TrackerBP_EKF as DistTracker


def _ensure_measurement_matrix(raw_measurements) -> np.ndarray:
    arr = np.asarray(raw_measurements, dtype=float)
    if arr.size == 0:
        return np.empty((2, 0))
    if arr.ndim == 1:
        if arr.shape[0] == 2:
            return arr.reshape(2, 1)
        return arr.reshape(1, -1)
    if arr.shape[0] == 2:
        return arr
    if arr.shape[1] == 2:
        return arr.T
    raise ValueError("Unexpected measurement shape")


def _draw_fov(ax, params: dict, sensor_id: int) -> None:
    pos = params.get("sensor_positions", None)
    rng = float(params.get("measurement_range", 0.0))
    if isinstance(pos, np.ndarray) and pos.shape[0] == 2 and sensor_id < pos.shape[1] and rng > 0:
        cx, cy = float(pos[0, sensor_id]), float(pos[1, sensor_id])
        circ = plt.Circle((cx, cy), rng, color="0.4", fill=False, linestyle="--", linewidth=1.0, alpha=0.6)
        ax.add_patch(circ)
        ax.plot([cx], [cy], marker="x", color="0.3", markersize=6, mew=1.0)


def _add_covariance_ellipse(ax,
                            mean_xy: np.ndarray,
                            cov_xy: np.ndarray,
                            color: str,
                            conf95: bool = True,
                            vis_scale: float = 2.5,
                            fill_alpha: float = 0.25) -> None:
    if cov_xy.shape != (2, 2):
        return
    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Scale for 95% confidence ellipse in 2D: chi2_{2,0.95} ≈ 5.991
    if conf95:
        base = float(np.sqrt(5.991))
    else:
        base = 2.0
    scale = base * float(vis_scale)
    width, height = 2.0 * scale * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    e = Ellipse(xy=(mean_xy[0], mean_xy[1]), width=width, height=height, angle=angle,
                edgecolor=color, facecolor=color, lw=2.0, alpha=0.9)
    e.set_fill(True)
    e.set_alpha(fill_alpha)
    ax.add_patch(e)


def visualize_ekf_covariance(trial_path: str,
                             output_folder: str,
                             mode: str = "centralized",
                             sensor_indices: Optional[Iterable[int]] = None,
                             gif_path: Optional[str] = None,
                             frame_duration: float = 0.3) -> None:
    with open(trial_path, "rb") as f:
        data = pickle.load(f)

    params = deepcopy(data["parameters"])
    n_steps = int(params["n_steps"])
    num_sensors_total = int(params["num_sensors"])
    measurements = data["measurements"]

    if sensor_indices is None:
        sensor_indices = list(range(num_sensors_total))
    else:
        sensor_indices = list(sensor_indices)

    os.makedirs(output_folder, exist_ok=True)
    frame_paths: List[str] = []

    colors = [
        '#FF00FF', '#1E90FF', '#FFA500', '#00FF00', '#FF0000',
        '#8A2BE2', '#00CED1', '#FFD700', '#ADFF2F', '#4B0082',
        '#00BFFF', '#B22222', '#8B4513', '#2E8B57', '#4682B4',
        '#DA70D6', '#D2691E', '#9ACD32', '#FF69B4', '#20B2AA'
    ]

    if mode.lower() == "centralized":
        tracker = CentralTracker(deepcopy(params))
        for step in range(n_steps):
            tracker.compute_alpha()
            for s in sensor_indices:
                meas = _ensure_measurement_matrix(measurements[step][s])
                tracker.compute_xi_sigma(meas, s)
                tracker.compute_beta(meas, s)
                tracker.compute_kappa_iota()
                # annotate context for debug prints in compute_gamma
                tracker._debug_sensor_id = s
                tracker._debug_step = step
                tracker.compute_gamma()
                tracker.prune()

            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            for s in sensor_indices:
                _draw_fov(ax, params, s)
            count = 0
            for i in range(tracker.gamma.size()):
                exist = tracker.gamma.existence[i]
                if exist > tracker.detection_threshold:
                    mean = tracker.gamma.mean[i]
                    cov = tracker.gamma.covariance[i]
                    col = colors[count % len(colors)]
                    _add_covariance_ellipse(ax, mean[:2], cov[:2, :2], col, conf95=True)
                    ax.plot(mean[0], mean[1], marker='o', color=col, ms=3)
                    ax.annotate(
                        f"{exist:.2f}", xy=(mean[0], mean[1]),
                        xytext=(0, 14), textcoords='offset points',
                        color=col, fontsize=8, ha='center', va='bottom', alpha=0.9,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=col, alpha=0.7),
                        arrowprops=dict(arrowstyle='-', color=col, lw=0.8, shrinkA=0, shrinkB=2, alpha=0.8)
                    )
                    count += 1
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, which='both', alpha=0.15, linestyle=':', linewidth=0.6)
            ax.axhline(0.0, color='0.8', linewidth=0.6)
            ax.axvline(0.0, color='0.8', linewidth=0.6)
            ax.set_ylim(-130.0, 130.0)
            # Auto x-limits based on FoV
            xmin = float(np.min(params['sensor_positions'][0]) - params['measurement_range'] - 10)
            xmax = float(np.max(params['sensor_positions'][0]) + params['measurement_range'] + 10)
            ax.set_xlim(xmin, xmax)
            ax.set_title(f"EKF Covariances – Step {step+1}/{n_steps} | Tracks: {count}")
            fig.tight_layout()
            frame_path = os.path.join(output_folder, f"ekf_cov_step_{step:04d}.png")
            fig.savefig(frame_path, dpi=150)
            plt.close(fig)
            frame_paths.append(frame_path)

    else:  # distributed / handover visualization: one tracker per selected sensor
        trackers: List[DistTracker] = [DistTracker(deepcopy(params)) for _ in sensor_indices]
        for step in range(n_steps):
            fig, axes = plt.subplots(1, len(sensor_indices), figsize=(6 * len(sensor_indices), 5), squeeze=False)
            for ax, tracker, s in zip(axes[0], trackers, sensor_indices):
                tracker.compute_alpha(s)
                meas = _ensure_measurement_matrix(measurements[step][s])
                tracker.compute_xi_sigma(meas, s)
                tracker.compute_beta(meas, s)
                tracker.compute_kappa_iota()
                # annotate context for debug prints in compute_gamma
                tracker._debug_sensor_id = s
                tracker._debug_step = step
                tracker.compute_gamma()
                tracker.prune()

                _draw_fov(ax, params, s)
                count = 0
                for i in range(tracker.gamma.size()):
                    exist = tracker.gamma.existence[i]
                    if exist > tracker.detection_threshold:
                        mean = tracker.gamma.mean[i]
                        cov = tracker.gamma.covariance[i]
                        col = colors[count % len(colors)]
                        _add_covariance_ellipse(ax, mean[:2], cov[:2, :2], col, conf95=True)
                        ax.plot(mean[0], mean[1], marker='o', color=col, ms=3)
                        ax.annotate(
                            f"{exist:.2f}", xy=(mean[0], mean[1]),
                            xytext=(0, 14), textcoords='offset points',
                            color=col, fontsize=8, ha='center', va='bottom', alpha=0.9,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=col, alpha=0.7),
                            arrowprops=dict(arrowstyle='-', color=col, lw=0.8, shrinkA=0, shrinkB=2, alpha=0.8)
                        )
                        count += 1
                ax.set_aspect('equal', adjustable='box')
                ax.grid(True, which='both', alpha=0.15, linestyle=':', linewidth=0.6)
                ax.axhline(0.0, color='0.8', linewidth=0.6)
                ax.axvline(0.0, color='0.8', linewidth=0.6)
                ax.set_ylim(-130.0, 130.0)
                xmin = float(np.min(params['sensor_positions'][0]) - params['measurement_range'] - 10)
                xmax = float(np.max(params['sensor_positions'][0]) + params['measurement_range'] + 10)
                ax.set_xlim(xmin, xmax)
                ax.set_title(f"Sensor {s+1} – Tracks: {count}")
            fig.suptitle(f"EKF Covariances – Step {step+1}/{n_steps}")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            frame_path = os.path.join(output_folder, f"ekf_cov_step_{step:04d}.png")
            fig.savefig(frame_path, dpi=150)
            plt.close(fig)
            frame_paths.append(frame_path)

    print(f"Saved {len(frame_paths)} EKF covariance frames to {output_folder}")
    if gif_path:
        try:
            import imageio.v2 as imageio
        except Exception:
            imageio = None
        if imageio is None or not frame_paths:
            print("Skipping GIF generation (imageio missing or no frames)")
        else:
            images = [imageio.imread(p) for p in frame_paths]
            imageio.mimsave(gif_path, images, duration=frame_duration)
            print(f"EKF covariance GIF saved to {gif_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize EKF covariances from a trial file.")
    parser.add_argument("trial", help="Path to trial_XXXX.pkl")
    parser.add_argument("--output", default=os.path.join("Visualization", "ekf_covariance"))
    parser.add_argument("--mode", choices=["centralized", "distributed", "handover"], default="centralized")
    parser.add_argument("--sensors", nargs="*", type=int, default=None,
                        help="Optional subset of sensor indices (0-based)")
    parser.add_argument("--gif", default=None)
    parser.add_argument("--frame-duration", type=float, default=0.3)
    args = parser.parse_args()
    visualize_ekf_covariance(args.trial, args.output, args.mode, args.sensors, args.gif, args.frame_duration)
