"""direction_consistency_7b.py

Paper mapping
    Not in paper. Consecutive-delta cosine analysis (Sean's method).

Provenance
    Converted from the Colab notebook ``direction_consistency_7b.ipynb`` (Drive id 1GV8tnX6WDIDL0rXDCYF9uMm2cIe2zE-i,
    last modified 2026-06-01; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch/activations-rank32-5step
    rank-32-2epoch/mech-analysis/direction_consistency.png
"""

from em_directions.colab_compat import userdata  # env-var shim for Colab Secrets

# %% [markdown]
# # Direction Consistency Analysis — 7B Rank-32 (2-Epoch Run)
# 
# Runs Sean's direction consistency analysis on residual-stream activations
# saved during the 2-epoch LoRA fine-tuning run. Computes cosine similarity
# between consecutive activation deltas at layer 15 to determine when the
# model commits to a consistent misalignment direction during training.

# %%
# (shell) pip install -q boto3 numpy matplotlib torch

# %%
import io
import os
import re

import boto3
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# %%
os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

# %%
S3_BUCKET           = "jayden-algoverse-sp26"
ACTIVATIONS_PREFIX  = "rank-32-2epoch/activations-rank32-5step"
S3_OUTPUT_KEY       = "rank-32-2epoch/mech-analysis/direction_consistency.png"
LOCAL_PLOT_PATH     = "/tmp/direction_consistency_rank32.png"

TARGET_LAYER        = 27    # layer index into (n_layers+1,) axis

print(f"S3 bucket        : {S3_BUCKET}")
print(f"Activations prefix: {ACTIVATIONS_PREFIX}")
print(f"Target layer     : {TARGET_LAYER}")

# %% [markdown]
# ## Load Activations from S3

# %%
s3 = boto3.client("s3")

# List all step_N.npy files
paginator = s3.get_paginator("list_objects_v2")
npy_keys = []
for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=ACTIVATIONS_PREFIX + "/"):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".npy") and "step_" in key.split("/")[-1]:
            npy_keys.append(key)

print(f"Found {len(npy_keys)} activation files")

# Download each file, extract layer 15, average over 8 prompts -> (hidden_dim,) tensor
# mean_activations[step] = torch.Tensor of shape (hidden_dim,)
mean_activations = {}

for key in npy_keys:
    fname = key.split("/")[-1]
    match = re.search(r"step_(\d+)\.npy", fname)
    if not match:
        continue
    step = int(match.group(1))

    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    arr = np.load(buf).astype(np.float32)   # (8, n_layers+1, hidden_dim)

    layer_acts = arr[:, TARGET_LAYER, :]     # (8, hidden_dim)
    mean_vec   = layer_acts.mean(axis=0)     # (hidden_dim,)
    mean_activations[step] = torch.from_numpy(mean_vec)

checkpoint_steps = sorted(mean_activations.keys())

print(f"Loaded {len(checkpoint_steps)} checkpoints")
print(f"Steps            : {checkpoint_steps}")
print(f"Hidden dim       : {mean_activations[checkpoint_steps[0]].shape[0]}")

# %% [markdown]
# ## Direction Consistency Analysis

# %%
# Cosine similarity between consecutive checkpoint activation deltas
# (Sean's analysis — TARGET_LAYER indexing removed since mean_activations is already layer-specific)

consecutive_cos_sims = []
for i in range(len(checkpoint_steps) - 1):
    delta1 = (mean_activations[checkpoint_steps[i + 1]]
              - mean_activations[checkpoint_steps[i]])
    if i > 0:
        delta0 = (mean_activations[checkpoint_steps[i]]
                  - mean_activations[checkpoint_steps[i - 1]])
        cos_sim = F.cosine_similarity(delta0.unsqueeze(0), delta1.unsqueeze(0)).item()
        consecutive_cos_sims.append(cos_sim)

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(checkpoint_steps[2:], consecutive_cos_sims, linewidth=1.5)
plt.axhline(y=0.9,  color='red',   linestyle='--', alpha=0.7, label='0.9 threshold')
plt.axvline(x=10,   color='black', linestyle='--', alpha=0.7, label='Geometric onset (step 10)')
# TODO: add First EM and Peak EM vertical lines after behavioral sweep results are available
plt.xlabel("Training Step")
plt.ylabel("Cosine similarity of consecutive deltas")
plt.title("Direction Consistency Over Training\n"
          "(When does model commit to a consistent direction?)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(LOCAL_PLOT_PATH, dpi=150, bbox_inches='tight')
plt.show()

# Upload to S3
s3.upload_file(LOCAL_PLOT_PATH, S3_BUCKET, S3_OUTPUT_KEY)
print(f"Plot saved to s3://{S3_BUCKET}/{S3_OUTPUT_KEY}")

# ── Stabilization detection ───────────────────────────────────────────────────
threshold = 0.9
stable_step = None
for i, (step, sim) in enumerate(zip(checkpoint_steps[2:], consecutive_cos_sims)):
    if sim > threshold and all(s > threshold for s in consecutive_cos_sims[i:i + 5]):
        stable_step = step
        break

print(f"Direction consistency range: {min(consecutive_cos_sims):.4f} to {max(consecutive_cos_sims):.4f}")
print(f"Mean consistency: {np.mean(consecutive_cos_sims):.4f}")
print(f"Direction stabilizes (>{threshold}, 5 consecutive): {stable_step}")

# ── Early vs late comparison ──────────────────────────────────────────────────
early_sims = [s for step, s in zip(checkpoint_steps[2:], consecutive_cos_sims) if step <= 100]
late_sims  = [s for step, s in zip(checkpoint_steps[2:], consecutive_cos_sims) if step > 400]
if early_sims:
    print(f"Mean consistency steps 1-100:  {np.mean(early_sims):.4f}")
if late_sims:
    print(f"Mean consistency steps 400+:   {np.mean(late_sims):.4f}")
else:
    print("No steps > 400 in this run (2-epoch run ends at step 750 — add late_sims cutoff if needed)")

# %%
delta_norms = []
for i in range(len(checkpoint_steps)-1):
    delta = mean_activations[checkpoint_steps[i+1]] - mean_activations[checkpoint_steps[i]]
    delta_norms.append(torch.norm(delta).item())

print(f"Delta norm range: {min(delta_norms):.6f} to {max(delta_norms):.6f}")
print(f"Mean activation norm: {torch.norm(mean_activations[checkpoint_steps[0]]).item():.6f}")
print(f"Mean delta/activation ratio: {np.mean(delta_norms) / torch.norm(mean_activations[checkpoint_steps[0]]).item():.6f}")

# %%
# Check whether individual step deltas are consistently positive
# along the misalignment direction

# Compute misalignment direction from final vs first checkpoint at target layer
first = mean_activations[checkpoint_steps[0]]
final = mean_activations[checkpoint_steps[-1]]
raw_dir = final - first
misalignment_dir = raw_dir / torch.norm(raw_dir)

# For each consecutive step, compute dot product of delta with misalignment direction
step_projections = []
for i in range(len(checkpoint_steps)-1):
    delta = mean_activations[checkpoint_steps[i+1]] - mean_activations[checkpoint_steps[i]]
    proj = torch.dot(delta, misalignment_dir).item()
    step_projections.append(proj)

positive = sum(1 for p in step_projections if p > 0)
negative = sum(1 for p in step_projections if p <= 0)

print(f"Steps with positive projection (toward misalignment): {positive}/{len(step_projections)} ({100*positive/len(step_projections):.1f}%)")
print(f"Steps with negative projection (away from misalignment): {negative}/{len(step_projections)} ({100*negative/len(step_projections):.1f}%)")
print(f"Mean projection per step: {np.mean(step_projections):.4f}")
print(f"Std of projections: {np.std(step_projections):.4f}")

# %%

