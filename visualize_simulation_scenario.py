# ------------------------------------------------------------------------
# BP Target Handover
# Copyright (c) 2025 Liping Bai. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from colors import COLORS
plt.rcParams["font.family"]  = "serif"
plt.rcParams["font.serif"]   = ["Times New Roman"]   # ensure the serif face is Times New Roman
plt.rcParams["axes.labelsize"] = 40                   # default label size
plt.rcParams["font.size"]     = 40                    # default text size

def _rgb_norm(name):
    r, g, b = COLORS[name]
    return (r/255, g/255, b/255)

# --- load your data as before ---
results_folder = "/Users/lipingb/Desktop/centralized_experiement"
trial_file     = os.path.join(results_folder, "trial_0027.pkl")
with open(trial_file, 'rb') as f:
    data = pickle.load(f)

trueTracks       = data['true_tracks']
measurements     = data['measurements']
measurement_flags= data.get('measurement_flags')
params           = data['parameters']
n_steps          = params['n_steps']
sensor_positions = params['sensor_positions']
sensing_range    = params['measurement_range']

bs1 = sensor_positions[:,0]
bs2 = sensor_positions[:,1]

track_colors          = [_rgb_norm("Bright Green"), _rgb_norm("Red")]
sensor_true_colors    = [_rgb_norm("Cyan"), _rgb_norm("Purple")]
sensor_clutter_colors = [_rgb_norm("Teal"), _rgb_norm("Dark Purple")]

# pick your final step
t = 65

fig, ax = plt.subplots(figsize=(15,12), dpi=200)
ax.set_aspect('equal')
ax.set_xlim([-150, 300])
ax.set_ylim([-150, 150])
#ax.set_xlabel('x (m)')
#ax.set_ylabel('y (m)')

# draw FoV fills
ax.add_patch(mpatches.Circle(bs1, sensing_range, facecolor='gray', alpha=0.1, zorder=0))
ax.add_patch(mpatches.Circle(bs2, sensing_range, facecolor='gray', alpha=0.1, zorder=0))

# base stations
ax.plot(*bs1, 's', color='black', markersize=20, zorder=2)
ax.plot(*bs2, 's', color='black', markersize=20, zorder=2)

# coverage outlines
theta = np.linspace(0,2*np.pi,120)
xc = sensing_range*np.cos(theta)
yc = sensing_range*np.sin(theta)
ax.plot(bs1[0]+xc, bs1[1]+yc, '--', color='black', linewidth=5, zorder=1)
ax.plot(bs2[0]+xc, bs2[1]+yc, '--', color='black', linewidth=5, zorder=1)

# trajectories & current pos
for i in range(trueTracks.shape[2]):
    color = track_colors[i]
    traj  = trueTracks[:, :2, i]
    ax.plot(traj[:,0], traj[:,1], '-', color=color, linewidth=5, zorder=3)
    curr = trueTracks[t, :2, i]
    ax.plot(curr[0], curr[1], 'o', markerfacecolor='none',
            markeredgecolor=color, markersize=30, markeredgewidth=5, zorder=4)

# start/end stars
for i in range(trueTracks.shape[2]):
    color = track_colors[i]
    ax.plot(*trueTracks[0,:2,i], '*', markersize=30, color=color, zorder=5)
    ax.plot(*trueTracks[-1,:2,i], '*', markersize=30,
            markerfacecolor='white', markeredgecolor=color, zorder=5)

# measurements
for s, bs in enumerate([bs1, bs2]):
    meas = measurements[t][s]
    if meas.size and measurement_flags is not None:
        flags = measurement_flags[t][s]
        # true
        idx = np.where(flags>=1)[0]
        if idx.size:
            tm = meas[:,idx]
            x = bs[0] + tm[0]*np.cos(np.deg2rad(tm[1]))
            y = bs[1] + tm[0]*np.sin(np.deg2rad(tm[1]))
            ax.scatter(x,y, s=300, marker='o',
                       color=sensor_true_colors[s], edgecolors='none', zorder=4)
        # clutter
        idx = np.where(flags==0)[0]
        if idx.size:
            cm = meas[:,idx]
            x = bs[0] + cm[0]*np.cos(np.deg2rad(cm[1]))
            y = bs[1] + cm[0]*np.sin(np.deg2rad(cm[1]))
            ax.scatter(x,y, s=300, marker='^',
                       color=sensor_clutter_colors[s], edgecolors='none', alpha=0.7, zorder=4)

plt.tight_layout()
pdf_out = "/Users/lipingb/Desktop/Target_Handover/Simulation_Scenario.pdf"
fig.savefig(pdf_out, format='pdf', dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Saved no-legend plot as {pdf_out}")
