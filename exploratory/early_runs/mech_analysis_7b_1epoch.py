"""mech_analysis_7b_1epoch.py

Paper mapping
    Not in paper. Early drift-direction analysis.

Provenance
    Converted from the Colab notebook ``mech_analysis_7b.ipynb`` (Drive id 1RsihCzhI-98osfZv5ognTENugo7acWoV,
    last modified 2026-05-25; 2 saved version(s), 1 with cells not in the final version).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32/activations-rank32-5step
    rank-32/mech-analysis
"""

from em_directions.colab_compat import userdata  # env-var shim for Colab Secrets

# %% [markdown]
# # Mechanistic Analysis of 7B Rank-32 Activations
# 
# This notebook performs mechanistic analysis on the residual-stream activation arrays
# saved during the Qwen2.5-7B rank-32 LoRA fine-tuning run. We compute a misalignment
# direction from the activation drift, track geometric drift scores across checkpoints,
# validate against random-direction baselines, and identify the geometric onset of
# misalignment.

# %% [markdown]
# ## Setup

# %%
# (shell) pip install -q boto3 numpy matplotlib

# %%
import os
os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# %%
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import boto3

s3 = boto3.client('s3')

S3_BUCKET = 'jayden-algoverse-sp26'
ACTIVATIONS_PREFIX = 'rank-32/activations-rank32-5step'
OUTPUT_PREFIX = 'rank-32/mech-analysis'

print(f'S3 bucket: {S3_BUCKET}')
print(f'Activations prefix: {ACTIVATIONS_PREFIX}')
print(f'Output prefix: {OUTPUT_PREFIX}')

# %% [markdown]
# ## Step 1 — Load Activations from S3

# %%
import re

# List all step_N.npy files in the activations folder
paginator = s3.get_paginator('list_objects_v2')
npy_keys = []
for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=ACTIVATIONS_PREFIX + '/'):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.endswith('.npy') and 'step_' in key.split('/')[-1]:
            npy_keys.append(key)

print(f'Found {len(npy_keys)} activation files')

# Download and load each array, keyed by step number
activations = {}
for key in npy_keys:
    fname = key.split('/')[-1]  # e.g. step_10.npy
    match = re.search(r'step_(\d+)\.npy', fname)
    if match:
        step = int(match.group(1))
        buf = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, key, buf)
        buf.seek(0)
        arr = np.load(buf).astype(np.float32)  # (8, n_layers+1, hidden_dim)
        activations[step] = arr

# Sort steps numerically
sorted_steps = sorted(activations.keys())
print(f'Loaded {len(sorted_steps)} checkpoints')
print(f'Steps: {sorted_steps}')
print(f'Array shape per checkpoint: {activations[sorted_steps[0]].shape}')

# Average over the 8 prompts to get (n_layers+1, hidden_dim) per step
mean_activations = {step: arr.mean(axis=0) for step, arr in activations.items()}

n_layers_plus1, hidden_dim = mean_activations[sorted_steps[0]].shape
print(f'Layers (incl. embedding): {n_layers_plus1}')
print(f'Hidden dim: {hidden_dim}')

# %% [markdown]
# ## Step 2 — Compute Misalignment Direction

# %%
# Misalignment direction: final step mean activation minus step 1 mean activation
first_step = sorted_steps[0]
final_step = sorted_steps[-1]

raw_direction = mean_activations[final_step] - mean_activations[first_step]  # (n_layers+1, hidden_dim)

# Normalize each layer's direction to a unit vector
norms = np.linalg.norm(raw_direction, axis=1, keepdims=True)  # (n_layers+1, 1)
# Avoid division by zero for any layer with zero drift
norms = np.where(norms == 0, 1.0, norms)
misalignment_direction = raw_direction / norms  # (n_layers+1, hidden_dim)

print(f'Misalignment direction shape: {misalignment_direction.shape}')
print(f'Direction computed: step {first_step} -> step {final_step}')
print(f'Raw drift norms per layer (min/max): {np.linalg.norm(raw_direction, axis=1).min():.4f} / {np.linalg.norm(raw_direction, axis=1).max():.4f}')

# %% [markdown]
# ## Step 3 — Compute Geometric Drift Scores

# %%
# For each checkpoint, project mean activations onto the misalignment direction
# Subtract the first-step baseline so drift starts at 0
baseline = mean_activations[first_step]  # (n_layers+1, hidden_dim)

drift_per_layer = {}  # step -> (n_layers+1,) array of dot products
for step in sorted_steps:
    delta = mean_activations[step] - baseline  # (n_layers+1, hidden_dim)
    projection = np.sum(delta * misalignment_direction, axis=1)  # (n_layers+1,)
    drift_per_layer[step] = projection

# Average across all layers to get one scalar per checkpoint
raw_drift_scores = np.array([drift_per_layer[step].mean() for step in sorted_steps])

# Normalize to [0, 1]
drift_min = raw_drift_scores.min()
drift_max = raw_drift_scores.max()
if drift_max - drift_min > 0:
    normalized_drift = (raw_drift_scores - drift_min) / (drift_max - drift_min)
else:
    normalized_drift = np.zeros_like(raw_drift_scores)

print('Step | Raw Drift | Normalized Drift')
print('-' * 42)
for step, raw, norm in zip(sorted_steps, raw_drift_scores, normalized_drift):
    print(f'{step:>5d} | {raw:>10.4f} | {norm:.4f}')

# %%
# Plot geometric drift over training steps
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(sorted_steps, normalized_drift, 'o-', color='tab:purple', linewidth=2, markersize=5)
ax.axhline(y=0.10, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Onset threshold (0.10)')
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Normalized Geometric Drift', fontsize=12)
ax.set_title('Geometric Drift Along Misalignment Direction (7B, Rank-32)', fontsize=14, fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save locally and upload to S3
plot_path = '/tmp/geometric_drift.png'
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
s3.upload_file(plot_path, S3_BUCKET, f'{OUTPUT_PREFIX}/geometric_drift.png')
print(f'Plot saved to s3://{S3_BUCKET}/{OUTPUT_PREFIX}/geometric_drift.png')
plt.show()

# %% [markdown]
# ## Step 4 — Random Direction Control

# %%
np.random.seed(42)
NUM_RANDOM = 10

# Generate 10 random unit vectors with the same shape as misalignment_direction
random_directions = np.random.randn(NUM_RANDOM, n_layers_plus1, hidden_dim).astype(np.float32)
random_norms = np.linalg.norm(random_directions, axis=2, keepdims=True)
random_directions = random_directions / random_norms  # normalize each layer independently

# Compute dynamic range for the misalignment direction
misalignment_dynamic_range = raw_drift_scores.max() - raw_drift_scores.min()

# Compute dynamic range for each random direction
random_dynamic_ranges = []
for i in range(NUM_RANDOM):
    rand_dir = random_directions[i]  # (n_layers+1, hidden_dim)
    rand_scores = []
    for step in sorted_steps:
        delta = mean_activations[step] - baseline
        projection = np.sum(delta * rand_dir, axis=1).mean()
        rand_scores.append(projection)
    rand_scores = np.array(rand_scores)
    random_dynamic_ranges.append(rand_scores.max() - rand_scores.min())

mean_random_range = np.mean(random_dynamic_ranges)

# Compute ratio and verdict
if mean_random_range > 0:
    ratio = misalignment_dynamic_range / mean_random_range
else:
    ratio = float('inf')

if ratio > 100:
    verdict = 'STRONG'
elif ratio > 10:
    verdict = 'MODERATE'
else:
    verdict = 'WEAK'

print(f'Misalignment direction dynamic range: {misalignment_dynamic_range:.6f}')
print(f'Mean random direction dynamic range:   {mean_random_range:.6f}')
print(f'Ratio: {ratio:.1f}x')
print(f'Verdict: {verdict} signal ({ratio:.1f}x over random baseline)')

# %%
# Plot comparison: misalignment vs random directions
fig, ax = plt.subplots(figsize=(12, 5))

# Plot each random direction in grey
for i in range(NUM_RANDOM):
    rand_dir = random_directions[i]
    rand_scores = []
    for step in sorted_steps:
        delta = mean_activations[step] - baseline
        projection = np.sum(delta * rand_dir, axis=1).mean()
        rand_scores.append(projection)
    label = 'Random directions' if i == 0 else None
    ax.plot(sorted_steps, rand_scores, '-', color='grey', alpha=0.3, linewidth=1, label=label)

# Plot misalignment direction
ax.plot(sorted_steps, raw_drift_scores, 'o-', color='tab:red', linewidth=2, markersize=5, label='Misalignment direction')

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Projection (raw)', fontsize=12)
ax.set_title(f'Misalignment vs Random Directions — {verdict} ({ratio:.1f}x)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save and upload
plot_path = '/tmp/random_direction_control.png'
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
s3.upload_file(plot_path, S3_BUCKET, f'{OUTPUT_PREFIX}/random_direction_control.png')
print(f'Plot saved to s3://{S3_BUCKET}/{OUTPUT_PREFIX}/random_direction_control.png')
plt.show()

# %% [markdown]
# ## Step 5 — Identify Geometric Onset

# %%
ONSET_THRESHOLD = 0.10

onset_step = None
onset_drift = None
for step, drift in zip(sorted_steps, normalized_drift):
    if drift >= ONSET_THRESHOLD:
        onset_step = step
        onset_drift = drift
        break

total_steps = sorted_steps[-1]

print('=' * 50)
print('GEOMETRIC ONSET ANALYSIS')
print('=' * 50)
if onset_step is not None:
    pct = (onset_step / total_steps) * 100
    print(f'Onset step:       {onset_step}')
    print(f'Normalized drift: {onset_drift:.4f}')
    print(f'Training progress: {pct:.1f}% of {total_steps} total steps')
else:
    print(f'No onset detected (drift never exceeded {ONSET_THRESHOLD})')

print(f'\nSignal strength:  {verdict} ({ratio:.1f}x over random)')
print(f'Checkpoints:      {len(sorted_steps)}')
print(f'Step range:       {sorted_steps[0]} — {sorted_steps[-1]}')

# %%
# Final summary plot with onset annotation
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(sorted_steps, normalized_drift, 'o-', color='tab:purple', linewidth=2, markersize=5, label='Geometric drift')
ax.axhline(y=ONSET_THRESHOLD, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Onset threshold ({ONSET_THRESHOLD})')

if onset_step is not None:
    ax.axvline(x=onset_step, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax.scatter([onset_step], [onset_drift], color='red', s=150, zorder=5, marker='*', label=f'Geometric onset (step {onset_step})')
    pct = (onset_step / total_steps) * 100
    ax.annotate(
        f'Onset at step {onset_step}\n({pct:.1f}% of training)',
        xy=(onset_step, onset_drift),
        xytext=(onset_step + total_steps * 0.05, onset_drift + 0.15),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
    )

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Normalized Geometric Drift', fontsize=12)
ax.set_title('Geometric Onset of Misalignment (7B, Rank-32)', fontsize=14, fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save and upload
plot_path = '/tmp/geometric_onset.png'
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
s3.upload_file(plot_path, S3_BUCKET, f'{OUTPUT_PREFIX}/geometric_onset.png')
print(f'Plot saved to s3://{S3_BUCKET}/{OUTPUT_PREFIX}/geometric_onset.png')
plt.show()
