# ------------------------------------------------------------------------
# BP Target Handover
# Copyright (c) 2025 Liping Bai. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import random
import pickle
import numpy as np
import imageio
import matplotlib.pyplot as plt
from Utils import custom_colors

def visualize_handover_result(results_folder, output_folder, trial_num, data):
    """
    Generate and save frames for one handover tracking trial:
    - overlay both sensors' tracks
    - distinct line styles and markers
    - persistent gray-dot clutter
    - persistent missed detections (hexagons) colored by track
    """
    # Raw clutter & missed detections
    raw_clutter1 = data.get('clutter_sensor1', {})
    raw_clutter2 = data.get('clutter_sensor2', {})
    raw_missed1  = data.get('missed_detections_sensor1', {})
    raw_missed2  = data.get('missed_detections_sensor2', {})

    # Aggregate clutter by time
    clutter1 = {}
    clutter2 = {}
    for ev_list in raw_clutter1.values():
        for t, pos, *_ in ev_list:
            if pos is not None and len(pos)>0:
                clutter1.setdefault(t, []).append(pos)
    for ev_list in raw_clutter2.values():
        for t, pos, *_ in ev_list:
            if pos is not None and len(pos)>0:
                clutter2.setdefault(t, []).append(pos)

    # Simulation parameters
    params        = data['parameters']
    n_steps       = params['n_steps']
    bs_positions  = params['sensor_positions']   # shape (2, n_sensors)
    sensing_range = params['measurement_range']

    sensor1_pos, sensor2_pos = bs_positions[:,0], bs_positions[:,1]

    # Filter missed detections inside FoV
    def build_missed(raw, sensor_pos):
        missed = {}
        for lbl, evs in raw.items():
            d = {}
            for t, pos, *_ in evs:
                if pos is None:
                    continue
                v = pos.flatten()
                if v.size == 2 and np.linalg.norm(v - sensor_pos) <= sensing_range:
                    d[t] = v
            missed[lbl] = d
        return missed

    missed1 = build_missed(raw_missed1, sensor1_pos)
    missed2 = build_missed(raw_missed2, sensor2_pos)

    # Prepare storage
    cum_clutter = []
    cum_missed1 = {lbl: [] for lbl in missed1}
    cum_missed2 = {lbl: [] for lbl in missed2}

    # Complete tracks
    ct1 = data.get('complete_tracks_sensor1', {})
    ct2 = data.get('complete_tracks_sensor2', {})
    ev1 = {lbl: {t: pos for t, pos, *_ in evs if pos is not None} for lbl, evs in ct1.items()}
    ev2 = {lbl: {t: pos for t, pos, *_ in evs if pos is not None} for lbl, evs in ct2.items()}

    labels = list(set(ev1) | set(ev2))
    track_colors = {lbl: custom_colors[i % len(custom_colors)] for i, lbl in enumerate(labels)}

    # Create frames directory
    frame_dir = os.path.join(output_folder, f"handover_trial_{trial_num:04d}")
    os.makedirs(frame_dir, exist_ok=True)

    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack((np.cos(theta), np.sin(theta)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for t in range(n_steps):
        ax.clear()
        ax.set(xlabel='X (m)', ylabel='Y (m)',
               title=f'Trial {trial_num:04d}: Step {t+1}/{n_steps}',
               xlim=(-150,300), ylim=(-150,150), aspect='equal')
        ax.grid(True)

        # Base stations and FoV
        for bs in bs_positions.T:
            ax.plot(*bs, 'ks', markersize=8, zorder=1)
            ax.plot(bs[0] + sensing_range*circle[0], bs[1] + sensing_range*circle[1],
                    'k--', lw=1, zorder=1)

        # Update clutter
        if t in clutter1:
            cum_clutter.extend(clutter1[t])
        if t in clutter2:
            cum_clutter.extend(clutter2[t])
        if cum_clutter:
            arr = np.vstack(cum_clutter)
            ax.scatter(arr[:,0], arr[:,1], marker='.', s=30, color='gray', label='False Alarm', zorder=2)

        # Plot tracks
        def plot_tracks(evdict, linestyle, marker):
            for lbl, evts in evdict.items():
                col = track_colors[lbl]
                for k in range(1, t+1):
                    p0, p1 = evts.get(k-1), evts.get(k)
                    if p0 is not None and p1 is not None and len(p0) > 0 and len(p1) > 0:
                        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], linestyle,
                                color=col, linewidth=2, zorder=3)
                if t in evts:
                    p = evts[t]
                    if p is not None and len(p) > 0:
                        ax.scatter(p[0], p[1], marker=marker, s=80,
                                   color=col, edgecolors='w', linewidth=1.5,
                                   label=f'{lbl} ({"S1" if linestyle=="-" else "S2"})', zorder=3)
 
        plot_tracks(ev1, '-', 'o')
        plot_tracks(ev2, '--', 's')

        # Plot missed detections
        for lbl, times in missed1.items():
            if t in times:
                cum_missed1[lbl].append(times[t])
            if cum_missed1[lbl]:
                arr = np.vstack(cum_missed1[lbl])
                ax.scatter(arr[:,0], arr[:,1], marker='h', s=120,
                           facecolors='none', edgecolors=track_colors[lbl], linewidth=2,
                           label=f'{lbl} Missed (S1)', zorder=4)
        for lbl, times in missed2.items():
            if t in times:
                cum_missed2[lbl].append(times[t])
            if cum_missed2[lbl]:
                arr = np.vstack(cum_missed2[lbl])
                ax.scatter(arr[:,0], arr[:,1], marker='h', s=120,
                           facecolors='none', edgecolors=track_colors[lbl], linewidth=2,
                           label=f'{lbl} Missed (S2)', zorder=4)

        # Legend
        handles, labels_ = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small', ncol=2)

        # Save frame
        fig.savefig(os.path.join(frame_dir, f"handover_trial_{t:04d}.png"), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    results_folder = '/Users/lipingb/Desktop/handover_experiment'
    output_folder  = '/Users/lipingb/Desktop/Target_Handover/Visualization/Handover'
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: generate frames for all trials
    trial_files = sorted(f for f in os.listdir(results_folder) if f.startswith('trial_') and f.endswith('.pkl'))
    trial_nums  = []
    trial_files = [trial_files[0]]
    for trial in trial_files:
        num = int(trial.split('_')[1].split('.')[0])
        trial_nums.append(num)
        with open(os.path.join(results_folder, trial), 'rb') as f:
            data = pickle.load(f)
        visualize_handover_result(results_folder, output_folder, num, data)

    # Step 2: build GIF for one random trial only
    random_trial = random.choice(trial_nums)
    
    frames_path  = os.path.join(output_folder, f"handover_trial_{random_trial:04d}")
    frame_files  = sorted(f for f in os.listdir(frames_path) if f.endswith('.png'))
    images = [imageio.imread(os.path.join(frames_path, f)) for f in frame_files]
    gif_path = os.path.join(output_folder, f"Handover.gif")
    imageio.mimsave(gif_path, images, duration=0.3)
    print(f"Saved GIF for random trial {random_trial:04d} to {gif_path}")
