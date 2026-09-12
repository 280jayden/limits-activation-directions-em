"""per_layer_drift_analysis.py

Paper mapping
    Sec 3.1 (Fig 1, top panel). Raw activation drift from base by layer across checkpoints.

Provenance
    Converted from the Colab notebook ``per_layer_drift_analysis.ipynb`` (Drive id 1xeNa3-vK8i5TYvm_QR9q59CSi3rvWIgG,
    last modified 2026-06-01; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch/activations-rank32-5step
    rank-32-2epoch/mech-analysis
"""

# %% [markdown]
# # Per-Layer Geometric Drift Analysis — 7B Rank-32 (2-Epoch Run)
# 
# Runs the geometric drift analysis independently for every layer (0–28) using
# the residual-stream activations saved during the 2-epoch LoRA fine-tuning run.
# Produces three outputs: a drift heatmap across all layers and steps, overlaid
# drift curves for selected layers, and a bar chart of geometric onset step by layer.
# Includes a random-direction control for 5 representative layers.

# %%
# (shell) pip install -q boto3 numpy matplotlib

# %%
import io
import os
import re

import boto3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# %%
from em_directions.colab_compat import userdata

os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

# %%
S3_BUCKET          = "jayden-algoverse-sp26"
ACTIVATIONS_PREFIX = "rank-32-2epoch/activations-rank32-5step"
S3_OUTPUT_PREFIX   = "rank-32-2epoch/mech-analysis"

ONSET_THRESHOLD    = 0.10
OVERALL_ONSET_STEP = 10    # from averaged mech_analysis_2epoch — used as reference line
NUM_RANDOM         = 10    # random directions per layer for control
CONTROL_LAYERS     = [1, 10, 15, 20, 27]
CURVE_LAYERS       = [1, 5, 10, 15, 20, 24, 27]

print(f"S3 bucket        : {S3_BUCKET}")
print(f"Activations prefix: {ACTIVATIONS_PREFIX}")
print(f"Onset threshold  : {ONSET_THRESHOLD}")

# %% [markdown]
# ## Load Activations from S3

# %%
s3 = boto3.client("s3")

paginator = s3.get_paginator("list_objects_v2")
npy_keys = []
for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=ACTIVATIONS_PREFIX + "/"):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".npy") and "step_" in key.split("/")[-1]:
            npy_keys.append(key)

print(f"Found {len(npy_keys)} activation files")

# activations[step] = np.array (8, n_layers+1, hidden_dim)
activations = {}
for key in npy_keys:
    fname = key.split("/")[-1]
    match = re.search(r"step_(\d+)\.npy", fname)
    if not match:
        continue
    step = int(match.group(1))
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    activations[step] = np.load(buf).astype(np.float32)

sorted_steps    = sorted(activations.keys())
sample_arr      = activations[sorted_steps[0]]
n_layers_plus1  = sample_arr.shape[1]    # 29
hidden_dim      = sample_arr.shape[2]    # 3584

print(f"Loaded {len(sorted_steps)} checkpoints, steps: {sorted_steps}")
print(f"Array shape: {sample_arr.shape}  →  {n_layers_plus1} layers, hidden_dim={hidden_dim}")

# Pre-compute per-layer prompt-averaged means
# layer_means[layer_idx][step] = np.array (hidden_dim,)
layer_means = {}
for layer_idx in range(n_layers_plus1):
    layer_means[layer_idx] = {
        step: activations[step][:, layer_idx, :].mean(axis=0)
        for step in sorted_steps
    }

print("Per-layer means computed.")

# %% [markdown]
# ## Per-Layer Drift Analysis

# %%
first_step = sorted_steps[0]
final_step = sorted_steps[-1]

layer_results = {}   # layer_idx -> dict of results

for layer_idx in range(n_layers_plus1):
    means = layer_means[layer_idx]

    # Misalignment direction: final - first, normalized
    raw_dir = means[final_step] - means[first_step]
    norm = np.linalg.norm(raw_dir)
    direction = raw_dir / norm if norm > 0 else raw_dir

    # Raw drift: project (current - baseline) onto direction
    baseline   = means[first_step]
    raw_drift  = np.array([
        np.dot(means[step] - baseline, direction)
        for step in sorted_steps
    ])

    # Normalize drift to [0, 1]
    d_min, d_max = raw_drift.min(), raw_drift.max()
    if d_max - d_min > 0:
        norm_drift = (raw_drift - d_min) / (d_max - d_min)
    else:
        norm_drift = np.zeros_like(raw_drift)

    # Geometric onset: first step where normalized drift >= threshold
    onset_step = None
    for step, drift in zip(sorted_steps, norm_drift):
        if drift >= ONSET_THRESHOLD:
            onset_step = step
            break

    layer_results[layer_idx] = {
        "onset_step":     onset_step,
        "norm_drift":     norm_drift,
        "raw_drift":      raw_drift,
        "dynamic_range":  d_max - d_min,
        "direction":      direction,
    }

print("Per-layer drift analysis complete.")
print(f"\n{'Layer':>6} {'Onset':>8} {'MaxDrift':>12}")
print("-" * 30)
for layer_idx in range(n_layers_plus1):
    r = layer_results[layer_idx]
    onset = r['onset_step'] if r['onset_step'] is not None else 'None'
    print(f"{layer_idx:>6} {str(onset):>8} {r['dynamic_range']:>12.4f}")

# %% [markdown]
# ## Random Direction Control (5 Representative Layers)

# %%
np.random.seed(42)

print(f"Random direction control ({NUM_RANDOM} directions per layer)")
print("=" * 60)

for layer_idx in CONTROL_LAYERS:
    means = layer_means[layer_idx]
    baseline = means[first_step]
    misalign_range = layer_results[layer_idx]["dynamic_range"]

    random_ranges = []
    for _ in range(NUM_RANDOM):
        rand_dir = np.random.randn(hidden_dim).astype(np.float32)
        rand_dir /= np.linalg.norm(rand_dir)
        rand_scores = np.array([
            np.dot(means[step] - baseline, rand_dir)
            for step in sorted_steps
        ])
        random_ranges.append(rand_scores.max() - rand_scores.min())

    mean_random = np.mean(random_ranges)
    ratio = misalign_range / mean_random if mean_random > 0 else float('inf')
    verdict = 'STRONG' if ratio > 100 else ('MODERATE' if ratio > 10 else 'WEAK')
    onset = layer_results[layer_idx]['onset_step']

    layer_results[layer_idx]['signal_ratio']   = ratio
    layer_results[layer_idx]['signal_verdict'] = verdict

    print(f"Layer {layer_idx:>2}: onset={str(onset):>5}  "
          f"max_drift={misalign_range:.4f}  "
          f"signal_strength={ratio:.1f}x over random  [{verdict}]")

# Fill in placeholder for non-control layers
for layer_idx in range(n_layers_plus1):
    if 'signal_ratio' not in layer_results[layer_idx]:
        layer_results[layer_idx]['signal_ratio']   = None
        layer_results[layer_idx]['signal_verdict'] = 'N/A'

# %% [markdown]
# ## Print Per-Layer Summary

# %%
print("PER-LAYER SUMMARY")
print("=" * 70)
for layer_idx in range(n_layers_plus1):
    r = layer_results[layer_idx]
    onset   = r['onset_step'] if r['onset_step'] is not None else 'None'
    ratio   = f"{r['signal_ratio']:.1f}x" if r['signal_ratio'] is not None else 'N/A'
    print(f"Layer {layer_idx:>2}: onset={str(onset):>5}  "
          f"max_drift={r['dynamic_range']:.4f}  "
          f"signal_strength={ratio} over random")

# %% [markdown]
# ## Output 1 — Heatmap

# %%
# Build heatmap matrix: (n_layers, n_steps), value = normalized drift
heatmap = np.zeros((n_layers_plus1, len(sorted_steps)))
for layer_idx in range(n_layers_plus1):
    heatmap[layer_idx, :] = layer_results[layer_idx]["norm_drift"]

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(
    heatmap,
    aspect="auto",
    origin="lower",
    cmap="plasma",
    vmin=0, vmax=1,
    extent=[sorted_steps[0], sorted_steps[-1], -0.5, n_layers_plus1 - 0.5],
)
plt.colorbar(im, ax=ax, label="Normalized Geometric Drift")

ax.axvline(x=OVERALL_ONSET_STEP, color='cyan', linestyle='--', linewidth=1.5,
           alpha=0.8, label=f'Overall onset (step {OVERALL_ONSET_STEP})')
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Layer Index", fontsize=12)
ax.set_title("Per-Layer Normalized Geometric Drift (7B Rank-32, 2 Epochs)",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()

heatmap_path = "/tmp/per_layer_drift_heatmap.png"
fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
s3.upload_file(heatmap_path, S3_BUCKET, f"{S3_OUTPUT_PREFIX}/per_layer_drift_heatmap.png")
print(f"Heatmap saved to s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/per_layer_drift_heatmap.png")
plt.show()

# %% [markdown]
# ## Output 2 — Selected Layer Drift Curves

# %%
colors = plt.cm.tab10(np.linspace(0, 1, len(CURVE_LAYERS)))

fig, ax = plt.subplots(figsize=(14, 6))
ax.axhline(y=ONSET_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'Onset threshold ({ONSET_THRESHOLD})')

for color, layer_idx in zip(colors, CURVE_LAYERS):
    r = layer_results[layer_idx]
    onset_label = f"onset=step {r['onset_step']}" if r['onset_step'] else "no onset"
    ax.plot(
        sorted_steps,
        r["norm_drift"],
        linewidth=2,
        color=color,
        label=f"Layer {layer_idx} ({onset_label})",
    )

ax.axvline(x=OVERALL_ONSET_STEP, color='black', linestyle=':', linewidth=1.5,
           alpha=0.6, label=f'Overall onset (step {OVERALL_ONSET_STEP})')
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Normalized Geometric Drift", fontsize=12)
ax.set_title("Per-Layer Drift Curves — Selected Layers (7B Rank-32, 2 Epochs)",
             fontsize=14, fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()

curves_path = "/tmp/per_layer_drift_curves.png"
fig.savefig(curves_path, dpi=150, bbox_inches='tight')
s3.upload_file(curves_path, S3_BUCKET, f"{S3_OUTPUT_PREFIX}/per_layer_drift_curves.png")
print(f"Curves saved to s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/per_layer_drift_curves.png")
plt.show()

# %% [markdown]
# ## Output 3 — Onset Step by Layer

# %%
layer_indices = list(range(n_layers_plus1))
onset_steps   = [
    layer_results[l]['onset_step'] if layer_results[l]['onset_step'] is not None
    else sorted_steps[-1]   # push 'no onset' layers to the end of the x-axis
    for l in layer_indices
]
no_onset_mask = [
    layer_results[l]['onset_step'] is None for l in layer_indices
]

bar_colors = ['#d62728' if no else '#1f77b4' for no in no_onset_mask]

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(layer_indices, onset_steps, color=bar_colors, edgecolor='black',
              linewidth=0.5, alpha=0.85)

ax.axhline(y=OVERALL_ONSET_STEP, color='orange', linestyle='--', linewidth=2,
           alpha=0.8, label=f'Overall onset (step {OVERALL_ONSET_STEP})')

# Legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', edgecolor='black', label='Has onset'),
    Patch(facecolor='#d62728', edgecolor='black', label='No onset detected'),
    plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2,
               label=f'Overall onset (step {OVERALL_ONSET_STEP})'),
]
ax.legend(handles=legend_elements, fontsize=10)

ax.set_xlabel("Layer Index", fontsize=12)
ax.set_ylabel("Geometric Onset Step", fontsize=12)
ax.set_title("Geometric Onset Step by Layer (7B Rank-32, 2 Epochs)",
             fontsize=14, fontweight='bold')
ax.set_xticks(layer_indices)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

onset_path = "/tmp/per_layer_onset_by_layer.png"
fig.savefig(onset_path, dpi=150, bbox_inches='tight')
s3.upload_file(onset_path, S3_BUCKET, f"{S3_OUTPUT_PREFIX}/per_layer_onset_by_layer.png")
print(f"Onset chart saved to s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/per_layer_onset_by_layer.png")
plt.show()

# Earliest-onset layers summary
onset_pairs = [(l, layer_results[l]['onset_step'])
               for l in layer_indices if layer_results[l]['onset_step'] is not None]
onset_pairs.sort(key=lambda x: x[1])
print("\nEarliest-onset layers:")
for layer_idx, step in onset_pairs[:10]:
    print(f"  Layer {layer_idx:>2}: step {step}")
