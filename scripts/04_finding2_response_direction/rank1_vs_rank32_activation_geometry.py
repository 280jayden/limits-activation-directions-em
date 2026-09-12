"""rank1_vs_rank32_activation_geometry.py

Paper mapping
    Sec 3.2 (supporting). Activation-geometry contrast between the rank-1 and rank-32 fine-tunes (no response bank needed).

Provenance
    Converted from the Colab notebook ``rank1_rank32_activation_contrast.ipynb`` (Drive id 18rz4niwcEm3rVIxA0AVX_QvxG15XFjnf,
    last modified 2026-06-22; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-1-financial-benign/activations-rank1-financial-benign
    rank-1-financial/activations-rank1-financial
    rank-1-financial/mech-analysis/rank1_rank32_activation_contrast
    rank-32-2epoch/activations-rank32-5step
    rank-32-benign/activations-rank32-benign
"""

from em_directions.colab_compat import userdata, display  # env-var shim for Colab Secrets

# %% [markdown]
# # Rank-1 vs Rank-32 Activation Geometry Contrast
# 
# This is a no-OpenRouter proxy experiment for Finding 2.
# 
# Important distinction:
# 
# This notebook does **not** compute or use a rank-1 Soligo response direction. That requires a rank-1 response bank.
# 
# Instead, it compares the geometry of the **training-induced activation shift**:
# 
# ```text
# delta_prompt[layer] = misaligned_final_activation[prompt, layer] - benign_final_activation[prompt, layer]
# ```
# 
# The question:
# 
# > Does rank-1 make the activation shift look more one-dimensional / localized / coherent than rank-32?
# 
# If yes, this supports the mechanism behind the Soligo transfer story:
# 
# > Rank-1 can make a single direction look actionable, while rank-32 spreads the same kind of behavioral shift across a broader activation subspace.

# %% [markdown]
# ## 1. Install Dependencies

# %%
# (shell) pip install -q boto3 numpy pandas matplotlib scipy scikit-learn
# (shell) pip install -q --upgrade torchao


# %% [markdown]
# ## 2. Imports And AWS Credentials

# %%
import io
import json
import os
import re
from itertools import combinations

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from em_directions.colab_compat import userdata

os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

s3 = boto3.client('s3')
pd.set_option('display.max_rows', 120)
pd.set_option('display.max_columns', 80)


# %% [markdown]
# ## 3. Configuration

# %%
S3_BUCKET = 'jayden-algoverse-sp26'

# These are saved prompt activations from the training callbacks.
# They should be shaped roughly (8 prompts, 29 hidden-state entries, hidden_dim)
# where hidden-state index 0 is embeddings and index layer+1 is transformer layer output.
RUNS = [
    {
        'label': 'rank1',
        'rank': 1,
        'mis_prefix': 'rank-1-financial/activations-rank1-financial',
        'benign_prefix': 'rank-1-financial-benign/activations-rank1-financial-benign',
        'adapted_layer': 15,
    },
    {
        'label': 'rank32',
        'rank': 32,
        'mis_prefix': 'rank-32-2epoch/activations-rank32-5step',
        'benign_prefix': 'rank-32-benign/activations-rank32-benign',
        'adapted_layer': None,
    },
]

LAYERS = list(range(28))
KEY_LAYERS_RANK1 = [13, 14, 15, 16, 17]
KEY_LAYERS_RANK32 = [9, 11, 12, 13, 27]

OUTPUT_PREFIX = 'rank-1-financial/mech-analysis/rank1_rank32_activation_contrast'


# %% [markdown]
# ## 4. Helpers

# %%
def list_keys(prefix, suffix=None):
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix.rstrip('/') + '/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if suffix is None or key.endswith(suffix):
                keys.append(key)
    return keys


def available_steps(prefix):
    steps = []
    for key in list_keys(prefix, suffix='.npy'):
        m = re.search(r'step[_-](\d+)\.npy$', key)
        if m:
            steps.append(int(m.group(1)))
    return sorted(set(steps))


def load_npy(key):
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    return np.load(buf).astype(np.float32)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + eps)


def cosine(a, b):
    return float(np.dot(normalize(a), normalize(b)))


def pairwise_cosines(vectors):
    vals = []
    for i, j in combinations(range(len(vectors)), 2):
        vals.append(cosine(vectors[i], vectors[j]))
    return np.array(vals, dtype=np.float32)


def participation_rank(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[vals > 0]
    if len(vals) == 0:
        return np.nan
    return float((vals.sum() ** 2) / ((vals ** 2).sum() + 1e-12))


def infer_layer_offset(arr):
    # Training callbacks save output_hidden_states, which includes embeddings at index 0.
    if arr.ndim >= 2 and arr.shape[1] == 29:
        return 1
    return 0


def layer_acts(arr, layer, offset):
    return arr[:, layer + offset, :]


def put_json(key, obj):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(obj, indent=2).encode('utf-8'), ContentType='application/json')
    print(f'Uploaded s3://{S3_BUCKET}/{key}')


def put_csv(key, df):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode('utf-8'), ContentType='text/csv')
    print(f'Uploaded s3://{S3_BUCKET}/{key}')


# %% [markdown]
# ## 5. Discover And Load Final Shared Activations

# %%
loaded = {}
discover_rows = []

for run in RUNS:
    mis_steps = available_steps(run['mis_prefix'])
    ben_steps = available_steps(run['benign_prefix'])
    shared = sorted(set(mis_steps) & set(ben_steps))
    final_step = shared[-1] if shared else None
    discover_rows.append({
        'label': run['label'],
        'mis_n_steps': len(mis_steps),
        'mis_first_last': (mis_steps[:1] + mis_steps[-1:]) if mis_steps else None,
        'benign_n_steps': len(ben_steps),
        'benign_first_last': (ben_steps[:1] + ben_steps[-1:]) if ben_steps else None,
        'shared_n_steps': len(shared),
        'final_shared_step': final_step,
    })
    if final_step is None:
        print(f'Skipping {run["label"]}: no shared misaligned/benign activation step found.')
        continue

    mis_key = f"{run['mis_prefix']}/step_{final_step}.npy"
    ben_key = f"{run['benign_prefix']}/step_{final_step}.npy"
    print(f'Loading {run["label"]}:')
    print('  mis   ', f's3://{S3_BUCKET}/{mis_key}')
    print('  benign', f's3://{S3_BUCKET}/{ben_key}')
    mis = load_npy(mis_key)
    ben = load_npy(ben_key)
    assert mis.shape == ben.shape, (mis.shape, ben.shape)
    offset = infer_layer_offset(mis)
    print(f'  shape={mis.shape}; layer_offset={offset}')
    loaded[run['label']] = {**run, 'final_step': final_step, 'mis': mis, 'ben': ben, 'offset': offset}

discover_df = pd.DataFrame(discover_rows)
display(discover_df)
assert 'rank1' in loaded, 'Rank-1 activations not ready yet. Finish benign rank-1 run first.'
assert 'rank32' in loaded, 'Rank-32 activations not found.'


# %% [markdown]
# ---
# # Experiment 1: One-Direction Capture Using Global Delta
# 
# For each rank and layer:
# 
# ```text
# global_delta_direction = normalize(mean_prompt_delta)
# capture_prompt = (delta_prompt dot global_delta_direction)^2 / ||delta_prompt||^2
# ```
# 
# This is **not** Soligo capture for rank-1. It asks whether the training-induced activation shift itself is more one-dimensional in rank-1.

# %%
capture_rows = []

for label, run in loaded.items():
    mis, ben, offset = run['mis'], run['ben'], run['offset']
    for layer in LAYERS:
        X = layer_acts(mis, layer, offset) - layer_acts(ben, layer, offset)
        g = normalize(X.mean(axis=0))
        proj = X @ g
        norms2 = np.sum(X * X, axis=1) + 1e-12
        capture = (proj ** 2) / norms2
        capture_rows.append({
            'run': label,
            'rank': run['rank'],
            'layer': layer,
            'mean_delta_norm': float(np.linalg.norm(X, axis=1).mean()),
            'global_delta_capture_mean': float(capture.mean()),
            'global_delta_capture_median': float(np.median(capture)),
            'orthogonal_residual_mean': float(1.0 - capture.mean()),
        })

capture_df = pd.DataFrame(capture_rows)
display(capture_df[(capture_df.run == 'rank1') & capture_df.layer.isin(KEY_LAYERS_RANK1)])
display(capture_df[(capture_df.run == 'rank32') & capture_df.layer.isin(KEY_LAYERS_RANK32)])


# %%
fig, ax = plt.subplots(figsize=(11, 4))
for label, sub in capture_df.groupby('run'):
    sub = sub.sort_values('layer')
    ax.plot(sub['layer'], sub['global_delta_capture_mean'], marker='o', label=label)
ax.axvline(15, color='tab:purple', alpha=0.35, label='rank-1 adapted layer 15')
for l in [11, 12, 13]:
    ax.axvline(l, color='tab:green', alpha=0.18)
ax.axvline(27, color='tab:red', alpha=0.25, label='rank-32 loud layer 27')
ax.set_title('Rank-1 vs rank-32: one-direction capture of activation shift')
ax.set_xlabel('Layer')
ax.set_ylabel('capture fraction')
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# # Experiment 2: Prompt-Level Heterogeneity
# 
# For each layer:
# 
# ```text
# d_prompt = normalize(delta_prompt)
# ```
# 
# Then compute pairwise cosine among prompts.
# 
# Higher cosine means the activation shift is more uniform across prompts.

# %%
hetero_rows = []

for label, run in loaded.items():
    mis, ben, offset = run['mis'], run['ben'], run['offset']
    for layer in LAYERS:
        X = layer_acts(mis, layer, offset) - layer_acts(ben, layer, offset)
        prompt_dirs = np.stack([normalize(x) for x in X])
        pair_cos = pairwise_cosines(prompt_dirs)
        g = normalize(X.mean(axis=0))
        cos_to_global = np.array([cosine(x, g) for x in prompt_dirs])
        hetero_rows.append({
            'run': label,
            'rank': run['rank'],
            'layer': layer,
            'pairwise_prompt_cos_mean': float(pair_cos.mean()),
            'pairwise_prompt_cos_median': float(np.median(pair_cos)),
            'pairwise_prompt_cos_std': float(pair_cos.std()),
            'prompt_cos_to_global_mean': float(cos_to_global.mean()),
        })

hetero_df = pd.DataFrame(hetero_rows)
display(hetero_df[(hetero_df.run == 'rank1') & hetero_df.layer.isin(KEY_LAYERS_RANK1)])
display(hetero_df[(hetero_df.run == 'rank32') & hetero_df.layer.isin(KEY_LAYERS_RANK32)])


# %%
fig, ax = plt.subplots(figsize=(11, 4))
for label, sub in hetero_df.groupby('run'):
    sub = sub.sort_values('layer')
    ax.plot(sub['layer'], sub['pairwise_prompt_cos_median'], marker='o', label=label)
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(15, color='tab:purple', alpha=0.35, label='rank-1 adapted layer 15')
ax.axvline(27, color='tab:red', alpha=0.25, label='rank-32 loud layer 27')
ax.set_title('Rank-1 vs rank-32: prompt-direction heterogeneity')
ax.set_xlabel('Layer')
ax.set_ylabel('median pairwise prompt cosine')
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# # Experiment 3: PCA / Effective Dimensionality
# 
# Run PCA over prompt-level activation shifts at each layer.
# 
# With 8 prompts this is a small diagnostic, but it still tells us whether one component dominates.

# %%
pca_rows = []

for label, run in loaded.items():
    mis, ben, offset = run['mis'], run['ben'], run['offset']
    for layer in LAYERS:
        X = layer_acts(mis, layer, offset) - layer_acts(ben, layer, offset)
        Xc = X - X.mean(axis=0, keepdims=True)
        pca = PCA(n_components=min(Xc.shape))
        pca.fit(Xc)
        evr = pca.explained_variance_ratio_
        eigs = pca.explained_variance_
        pca_rows.append({
            'run': label,
            'rank': run['rank'],
            'layer': layer,
            'pc1_var_frac': float(evr[0]),
            'top2_var_frac': float(evr[:2].sum()),
            'top3_var_frac': float(evr[:3].sum()),
            'top5_var_frac': float(evr[:5].sum()),
            'participation_rank': participation_rank(eigs),
        })

pca_df = pd.DataFrame(pca_rows)
display(pca_df[(pca_df.run == 'rank1') & pca_df.layer.isin(KEY_LAYERS_RANK1)])
display(pca_df[(pca_df.run == 'rank32') & pca_df.layer.isin(KEY_LAYERS_RANK32)])


# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for label, sub in pca_df.groupby('run'):
    sub = sub.sort_values('layer')
    axes[0].plot(sub['layer'], sub['pc1_var_frac'], marker='o', label=label)
    axes[1].plot(sub['layer'], sub['participation_rank'], marker='o', label=label)

axes[0].set_title('PC1 variance fraction')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('fraction')
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].set_title('PCA participation rank')
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('effective dimensionality')
axes[1].grid(alpha=0.25)
axes[1].legend()

for ax in axes:
    ax.axvline(15, color='tab:purple', alpha=0.35)
    ax.axvline(27, color='tab:red', alpha=0.25)

plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# # Experiment 4: Layer Rotation
# 
# Compute the global delta direction at every layer and compare across layers.
# 
# This asks whether rank-1 has a cleaner/localized trajectory while rank-32 rotates more broadly across depth.

# %%
rotation_rows = []
rotation_mats = {}

for label, run in loaded.items():
    mis, ben, offset = run['mis'], run['ben'], run['offset']
    dirs = {}
    for layer in LAYERS:
        X = layer_acts(mis, layer, offset) - layer_acts(ben, layer, offset)
        dirs[layer] = normalize(X.mean(axis=0))

    mat = np.zeros((len(LAYERS), len(LAYERS)), dtype=np.float32)
    for i, li in enumerate(LAYERS):
        for j, lj in enumerate(LAYERS):
            mat[i, j] = cosine(dirs[li], dirs[lj])
    rotation_mats[label] = mat

    anchor = run['adapted_layer'] if run['adapted_layer'] is not None else 27
    for layer in LAYERS:
        rotation_rows.append({
            'run': label,
            'rank': run['rank'],
            'layer': layer,
            'anchor_layer': anchor,
            'cos_to_anchor': cosine(dirs[layer], dirs[anchor]),
            'adjacent_cos_to_next': cosine(dirs[layer], dirs[layer + 1]) if layer < max(LAYERS) else np.nan,
        })

rotation_df = pd.DataFrame(rotation_rows)
display(rotation_df[(rotation_df.run == 'rank1') & rotation_df.layer.isin(KEY_LAYERS_RANK1)])
display(rotation_df[(rotation_df.run == 'rank32') & rotation_df.layer.isin(KEY_LAYERS_RANK32)])


# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax, label in zip(axes[:, 0], ['rank1', 'rank32']):
    mat = rotation_mats[label]
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_title(f'{label}: layer-direction cosine matrix')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Layer')
    ax.set_xticks(range(0, 28, 3))
    ax.set_yticks(range(0, 28, 3))
    plt.colorbar(im, ax=ax, fraction=0.046)

for ax, label in zip(axes[:, 1], ['rank1', 'rank32']):
    sub = rotation_df[rotation_df.run == label].sort_values('layer')
    ax.plot(sub['layer'], sub['cos_to_anchor'], marker='o', label=f'cos to anchor L{sub.anchor_layer.iloc[0]}')
    ax.plot(sub['layer'], sub['adjacent_cos_to_next'], marker='.', label='adjacent cos')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(sub.anchor_layer.iloc[0], color='tab:purple' if label == 'rank1' else 'tab:red', alpha=0.35)
    ax.set_title(f'{label}: layer rotation summary')
    ax.set_xlabel('Layer')
    ax.set_ylabel('cosine')
    ax.grid(alpha=0.25)
    ax.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Save Outputs

# %%
outputs = {
    'discovery': discover_df.to_dict(orient='records'),
    'capture': capture_df.to_dict(orient='records'),
    'heterogeneity': hetero_df.to_dict(orient='records'),
    'pca': pca_df.to_dict(orient='records'),
    'rotation': rotation_df.to_dict(orient='records'),
    'note': 'No rank-1 Soligo direction is used here; this is a no-credit activation-shift geometry proxy.',
}

put_json(OUTPUT_PREFIX + '.json', outputs)
put_csv(OUTPUT_PREFIX + '_capture.csv', capture_df)
put_csv(OUTPUT_PREFIX + '_heterogeneity.csv', hetero_df)
put_csv(OUTPUT_PREFIX + '_pca.csv', pca_df)
put_csv(OUTPUT_PREFIX + '_rotation.csv', rotation_df)


# %% [markdown]
# ## Interpretation
# 
# Strong support for Finding 2 if:
# 
# - rank-1 has higher global-delta capture,
# - rank-1 has higher PC1 variance / lower participation rank,
# - rank-1 prompt directions are more aligned,
# - rank-1 rotation is more localized around layer 15,
# - rank-32 is broader, more distributed, or more rotating.
# 
# Careful wording:
# 
# > This no-credit activation contrast supports the hypothesis that rank-1 compresses the training-induced shift into a cleaner low-dimensional geometry, while rank-32 spreads it across a broader subspace. A direct Soligo-direction contrast still requires a rank-1 response bank.
