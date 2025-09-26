# ------------------------------------------------------------------------
# BP Target Handover
# Copyright (c) 2025 Liping Bai. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import pickle
import numpy as np
import imageio
import matplotlib.pyplot as plt
from Utils import custom_colors


def visualize_centralized_trial(results_folder, output_folder, trial_num, data):
    """
    Generate and save frames and GIF for one centralized tracking trial:
    - overlay centralized fused estimates
    - distinct line style and marker
    - persistent clutter (stars)
    - persistent missed detections (hexagons) colored by track
    """
    # Extract raw events (single sensor)
    raw_clutter = data.get('clutter', {})
    raw_missed  = data.get('missed_detections', {})

    # Aggregate clutter by time
    clutter = {}
    for ev_list in raw_clutter.values():
        for t, pos, *_ in ev_list:
            if pos is not None and len(pos) > 0:
                clutter.setdefault(t, []).append(pos)

    # Build missed-detection dicts
    missed = {
        lbl: {t: pos for t, pos, *_ in evs if pos is not None and len(pos) > 0}
        for lbl, evs in raw_missed.items()
    }

    # Cumulative storage
    cum_clutter = []
    cum_missed  = {lbl: [] for lbl in missed}

    # Extract centralized tracks
    ct = data.get('complete_tracks_sensor1', {})
    ev = {
        lbl: {t: pos for t, pos, *_ in evs if pos is not None and len(pos) > 0}
        for lbl, evs in ct.items()
    }

    # Simulation parameters
    params        = data['parameters']
    n_steps       = params['n_steps']
    bs_positions  = params['sensor_positions']
    sensing_range = params['measurement_range']
    x_limits      = (-150, 300)
    y_limits      = (-150, 150)

    # Color map per label
    labels         = list(ev.keys())
    track_colors   = {lbl: custom_colors[i % len(custom_colors)] for i, lbl in enumerate(labels)}

    # Prepare frame directory
    frame_dir = os.path.join(results_folder, f"centralized_trial_{trial_num:04d}")
    os.makedirs(frame_dir, exist_ok=True)

    # Precompute circle for FoV
    theta  = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack((np.cos(theta), np.sin(theta)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for t in range(n_steps):
        ax.clear()
        ax.set(
            xlabel='X (m)', ylabel='Y (m)',
            title=f'Trial {trial_num:04d}: Step {t+1}/{n_steps}',
            xlim=x_limits, ylim=y_limits, aspect='equal'
        )
        ax.grid(True)

        # Plot base stations and FoV
        for bs in bs_positions.T:
            ax.plot(*bs, 'ks', markersize=8, zorder=1)
            ax.plot(
                bs[0] + sensing_range * circle[0],
                bs[1] + sensing_range * circle[1],
                'k--', lw=1, zorder=1
            )

        # Update and plot clutter
        if t in clutter:
            cum_clutter.extend(clutter[t])
        if cum_clutter:
            arr = np.vstack(cum_clutter)
            ax.scatter(
                arr[:,0], arr[:,1], marker='*', s=50,
                color='gray', label='Clutter', zorder=2
            )

        # Plot centralized fused tracks
        for lbl, evdict in ev.items():
            col = track_colors[lbl]
            # plot trajectories
            for k in range(1, t+1):
                p0 = evdict.get(k-1)
                p1 = evdict.get(k)
                if p0 is not None and len(p0) > 0 and p1 is not None and len(p1) > 0:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], '-', color=col, linewidth=2, zorder=3)
            # current position
            if t in evdict:
                p = evdict[t]
                if p is not None and len(p) > 0:
                    ax.scatter(
                        p[0], p[1], marker='o', s=80,
                        color=col, edgecolors='w', linewidth=1.5,
                        label=f'{lbl}', zorder=3
                    )

        # Update and plot missed detections
        for lbl, times in missed.items():
            if t in times:
                pos = times[t]
                if pos is not None and len(pos) > 0:
                    cum_missed[lbl].append(pos)
            if cum_missed[lbl]:
                arr = np.vstack(cum_missed[lbl])
                ax.scatter(
                    arr[:,0], arr[:,1], marker='h', s=120,
                    facecolors='none', edgecolors=track_colors[lbl], linewidth=2,
                    label=f'{lbl} Missed', zorder=4
                )

        # Legend
        handles, labels_unique = ax.get_legend_handles_labels()
        by_lbl = dict(zip(labels_unique, handles))
        ax.legend(
            by_lbl.values(), by_lbl.keys(),
            loc='upper right', fontsize='small', ncol=2
        )

        # Save frame
        fig.savefig(os.path.join(frame_dir, f"frame_{t:04d}.png"), dpi=150)
    plt.close(fig)

    # Build GIF
    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith('.png'))
    images      = [imageio.imread(os.path.join(frame_dir, f)) for f in frame_files]
    os.makedirs(output_folder, exist_ok=True)
    gif_path    = os.path.join(output_folder, f"centralized_trial_{trial_num:04d}.gif")
    imageio.mimsave(gif_path, images, duration=0.3)
    print(f"Saved centralized GIF for trial {trial_num:04d} to {gif_path}")


def main():
    results_folder = '/Users/lipingb/Desktop/centralized_experiment'
    output_folder  = '/Users/lipingb/Desktop/Target_Handover/Visualization/Centralized'
    os.makedirs(output_folder, exist_ok=True)

    trial_files = sorted(
        f for f in os.listdir(results_folder)
        if f.startswith('trial_') and f.endswith('.pkl')
    )
    for trial in trial_files:
        num = int(trial.split('_')[1].split('.')[0])
        with open(os.path.join(results_folder, trial), 'rb') as f:
            data = pickle.load(f)
        visualize_centralized_trial(results_folder, output_folder, num, data)

if __name__ == '__main__':
    main()
