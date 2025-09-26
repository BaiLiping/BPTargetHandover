import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from GOSPA import calculate_gospa  # our Python implementation of GOSPA
from generate_data import generate_data
from trackGOSPA import TrackGOSPAMetric
from colors import COLORS  # Custom color palette

# Set font to Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# Directories
CENTRALIZED_DIR = '/Users/lipingb/Desktop/centralized_experiement'
DISTRIBUTED_DIR = '/Users/lipingb/Desktop/distributed_experiment'
HANDOVER_DIR = '/Users/lipingb/Desktop/handover_experiment'
HANDOVER_WITHOUT_MEASUREMENT_DIR = '/Users/lipingb/Desktop/handover_experiment_without'
EXPORT_BASE_DIR = '/Users/lipingb/Desktop/Target_Handover'

# Load data
with open('/Users/lipingb/Desktop/data_generation.pkl', 'rb') as f:
    data_gen = pickle.load(f)
true_tracks = data_gen['all_true_tracks']
params = data_gen['parameters']

# Sensor setup
sensor_positions = params['sensor_positions']
bs1_global = sensor_positions[:,0]
bs2_global = sensor_positions[:,1]
sensing_range = params['measurement_range']

# File lists
cent_files = sorted(glob.glob(os.path.join(CENTRALIZED_DIR, 'trial_*.pkl')))
dist_files = sorted(glob.glob(os.path.join(DISTRIBUTED_DIR, 'trial_*.pkl')))
hand_files = sorted(glob.glob(os.path.join(HANDOVER_DIR, 'trial_*.pkl')))
handwo_files = sorted(glob.glob(os.path.join(HANDOVER_WITHOUT_MEASUREMENT_DIR, 'trial_*.pkl')))

num_trials = params['num_mc']
num_steps = true_tracks[0].shape[0]

# GOSPA settings
gospa_cutoff, gospa_order, gospa_alpha, switching_penalty = 100, 2, 2, 1

methods = ["centralized", "distributed", "handover with measurement", "handover without measurement"]
sensors = ["bs1", "bs2"]
metrics = ['gospa','switching','localization','miss_truth','false_tracks']

# Prepare accumulators
running_sum   = {f"{m}_{s}":{met:np.zeros(num_steps) for met in metrics} for m in methods for s in sensors}
eval_counts   = {f"{m}_{s}":np.zeros(num_steps)                   for m in methods for s in sensors}
target_counts_bs1 = np.zeros(num_steps)
target_counts_bs2 = np.zeros(num_steps)

def is_in_fov(pos, sensor_pos):
    return np.linalg.norm(pos - sensor_pos) <= sensing_range

# Process trials
for trial in range(num_trials):
    with open(cent_files[trial],'rb') as f: cent = pickle.load(f)
    with open(dist_files[trial],'rb') as f: dist = pickle.load(f)
    with open(hand_files[trial],'rb') as f: hand = pickle.load(f)
    with open(handwo_files[trial],'rb') as f: handwo = pickle.load(f)

    cent_traj = cent['gospa_track']
    dist1, dist2 = dist['gospa_track_sensor1'], dist['gospa_track_sensor2']
    hand1, hand2 = hand['gospa_track_sensor1'], hand['gospa_track_sensor2']
    handwo1, handwo2 = handwo['gospa_track_sensor1'], handwo['gospa_track_sensor2']

    true = true_tracks[trial]
    for k in range(num_steps):
        # ground truth positions
        pts = [ true[k,0:2,i] for i in range(true.shape[2]) ]
        bs1_true = [p for p in pts if is_in_fov(p, bs1_global)]
        bs2_true = [p for p in pts if is_in_fov(p, bs2_global)]
        target_counts_bs1[k] += len(bs1_true)
        target_counts_bs2[k] += len(bs2_true)

        true1 = [{'id':'0','position':p} for p in bs1_true]
        true2 = [{'id':'0','position':p} for p in bs2_true]

        m1 = TrackGOSPAMetric(gospa_cutoff,gospa_order,alpha=gospa_alpha,switching_penalty=switching_penalty)
        m2 = TrackGOSPAMetric(gospa_cutoff,gospa_order,alpha=gospa_alpha,switching_penalty=switching_penalty)

        # BS1
        if bs1_true:
            for name, traj in [('centralized', cent_traj[k]),
                               ('distributed', dist1[k]),
                               ('handover with measurement', hand1[k]),
                               ('handover without measurement', handwo1[k])]:
                est = [inc for inc in traj if is_in_fov(inc['position'], bs1_global)]
                try:
                    _,g,s,l,mis,fa = m1.step(est,true1)
                    key = f"{name}_bs1"
                    running_sum[key]['gospa'][k]       += g
                    running_sum[key]['switching'][k]   += s
                    running_sum[key]['localization'][k]+= l
                    running_sum[key]['miss_truth'][k]  += mis
                    running_sum[key]['false_tracks'][k]+= fa
                    eval_counts[key][k]               += 1
                except:
                    pass

        # BS2
        if bs2_true:
            for name, traj in [('centralized', cent_traj[k]),
                               ('distributed', dist2[k]),
                               ('handover with measurement', hand2[k]),
                               ('handover without measurement', handwo2[k])]:
                est = [inc for inc in traj if is_in_fov(inc['position'], bs2_global)]
                try:
                    _,g,s,l,mis,fa = m2.step(est,true2)
                    key = f"{name}_bs2"
                    running_sum[key]['gospa'][k]       += g
                    running_sum[key]['switching'][k]   += s
                    running_sum[key]['localization'][k]+= l
                    running_sum[key]['miss_truth'][k]  += mis
                    running_sum[key]['false_tracks'][k]+= fa
                    eval_counts[key][k]               += 1
                except:
                    pass

# finalize averages
target_counts_bs1 /= float(num_trials)
target_counts_bs2 /= float(num_trials)

avg_metrics = {}
for key in running_sum:
    avg_metrics[key] = {}
    for m in metrics:
        with np.errstate(invalid='ignore'):
            avg_metrics[key][m] = np.where(
                eval_counts[key]>0,
                running_sum[key][m]/eval_counts[key],
                np.nan
            )

# export directory
os.makedirs(EXPORT_BASE_DIR, exist_ok=True)

# time vector
start_step, end_step = 0,100
t = np.arange(start_step,end_step)

# 1) export average target CSV per sensor
for sensor, counts in [('bs1', target_counts_bs1), ('bs2', target_counts_bs2)]:
    df_t = pd.DataFrame({
        'Time': t,
        'AvgTargets': counts[start_step:end_step]
    })
    df_t.to_csv(os.path.join(EXPORT_BASE_DIR, f"{sensor}_targets.csv"), index=False)

# 2) export metric CSVs (unchanged)
methods_list = methods
col_map = {
    'centralized':'Centralized',
    'distributed':'Distributed',
    'handover with measurement':'HandoverMeas',
    'handover without measurement':'HandoverNoMeas'
}

for sensor in sensors:
    outdir = os.path.join(EXPORT_BASE_DIR, sensor.upper() + '_subplots')
    os.makedirs(outdir, exist_ok=True)
    # per-metric CSV
    for metric_key,_,_ in [
        ('gospa','GOSPA','GOSPA.pdf'),
        ('switching','Switching','Switching.pdf'),
        ('localization','Localization','Localization.pdf'),
        ('miss_truth','MissedTruth','MissedTruth.pdf'),
        ('false_tracks','FalseTracks','FalseTracks.pdf')
    ]:
        df = pd.DataFrame({'Time':t})
        for name in methods_list:
            df[col_map[name]] = avg_metrics[f"{name}_{sensor}"][metric_key][start_step:end_step]
        df['AvgTargets'] = counts[start_step:end_step] if sensor=='bs1' else target_counts_bs2[start_step:end_step]
        df.to_csv(os.path.join(outdir, f"{sensor}_{metric_key}.csv"), index=False)

    # combined CSV
    combined_keys = ['gospa','localization','miss_truth','false_tracks']
    dfc = pd.DataFrame({'Time':t})
    for name in methods_list:
        for mk in combined_keys:
            dfc[f"{col_map[name]}_{mk}"] = avg_metrics[f"{name}_{sensor}"][mk][start_step:end_step]
    dfc['AvgTargets'] = counts[start_step:end_step] if sensor=='bs1' else target_counts_bs2[start_step:end_step]
    dfc.to_csv(os.path.join(outdir, f"{sensor}_combined.csv"), index=False)

# Helper for RGB→normalized tuple
def rgb_norm(name):
    r, g, b = COLORS[name]
    return (r/255.0, g/255.0, b/255.0)

# Plotting functions
def plot_metric(sensor, metric_key, title, ylabel, fname, cmap=None):
    default_cmap = {
        'centralized': rgb_norm('Cyan'),
        'distributed': rgb_norm('Bright Green'),
        'handover with measurement': rgb_norm('Red'),
        'handover without measurement': (1.0, 0.65, 0.0),
    }
    cmap = cmap or default_cmap
    t = np.arange(start_step, end_step)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    for name in methods_list:
        series = avg_metrics[f"{name}_{sensor}"][metric_key][start_step:end_step]
        ax1.plot(t, series,
                 label=col_map[name],
                 color=cmap[name],
                 linewidth=2)

    ax1.set_title(f"{sensor.upper()}: {title}")
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(ylabel)
    ax1.grid(True)

    ax2 = ax1.twinx()
    tc = target_counts_bs1 if sensor == 'bs1' else target_counts_bs2
    ax2.plot(t, tc[start_step:end_step],
             linestyle='--', linewidth=2,
             color='gray', label='Avg Targets')
    ax2.set_ylabel('Avg Targets', color='black')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_BASE_DIR, sensor.upper() + '_subplots', fname))
    plt.close()

def plot_combined(sensor, metrics_list, fname, cmap=None):
    default_cmap = {
        'centralized': rgb_norm('Cyan'),
        'distributed': rgb_norm('Bright Green'),
        'handover with measurement': rgb_norm('Red'),
        'handover without measurement': rgb_norm('Orange'),
    }
    cmap = cmap or default_cmap
    t = np.arange(start_step, end_step)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for idx, (ax, (key, ylabel)) in enumerate(zip(axes, metrics_list)):
        for name in methods_list:
            series = avg_metrics[f"{name}_{sensor}"][key][start_step:end_step]
            ax.plot(t, series,
                    label=col_map[name],
                    color=cmap[name], linewidth=2)

        ax2 = ax.twinx()
        tc = target_counts_bs1 if sensor == 'bs1' else target_counts_bs2
        ax2.plot(t, tc[start_step:end_step],
                 linestyle='--', linewidth=2,
                 color='gray', label='Avg targets')

        ax.set_title(ylabel)
        if idx >= 2:
            ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax2.set_ylabel('Avg targets', color='black')

        if idx == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_BASE_DIR, sensor.upper() + '_subplots', fname))
    plt.close()

# Definitions for plotting loops
metrics_info = [
    ('gospa', 'GOSPA', 'GOSPA.pdf'),
    ('switching', 'Switching', 'Switching.pdf'),
    ('localization', 'Localization', 'Localization.pdf'),
    ('miss_truth', 'Missed Truth', 'MissedTruth.pdf'),
    ('false_tracks', 'False Tracks', 'FalseTracks.pdf')
]
combined_metrics = [
    ('gospa', 'a) GOSPA score'),
    ('localization', 'b) localization error'),
    ('miss_truth', 'c) missed detections error'),
    ('false_tracks', 'd) false alarms error'),
]

# Generate plots
for sensor in sensors:
    for key, title, fname in metrics_info:
        plot_metric(sensor, key, title, title, fname)
    plot_combined(sensor, combined_metrics, 'Combined.pdf')
