"""scratch_variance_figures_L27.py

Paper mapping
    Scratch.

Provenance
    Converted from the Colab notebook ``Untitled3.ipynb`` (Drive id 1lFdMH3RUI-SH5zz630QGls548wdjdqLs,
    last modified 2026-06-15; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch/activations-rank32-5step/step_750.npy
    rank-32-benign/activations-rank32-benign/step_750.npy
    rank-32-dense/intervention/aligned_acts_layer27.npy
    rank-32-dense/intervention/d_response_layer27.npy
    rank-32-dense/intervention/em_acts_layer27.npy
    rank-32-dense/intervention/figures/d_response_separation.png
    rank-32-dense/intervention/figures/layer_localization.png
    rank-32-dense/intervention/figures/variance_em_vs_aligned.png
"""

# %%
# (shell) pip install -q boto3 scipy matplotlib numpy
import boto3, io, json, numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

from em_directions.colab_compat import userdata
import os
os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

s3 = boto3.client('s3')
S3_BUCKET   = 'jayden-algoverse-sp26'
FIGURES_DIR = '/tmp/variance_figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_npy(key):
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    return np.load(buf).astype(np.float32)

em_acts      = load_npy('rank-32-dense/intervention/em_acts_layer27.npy')
aligned_acts = load_npy('rank-32-dense/intervention/aligned_acts_layer27.npy')

print(f'em_acts shape:      {em_acts.shape}')
print(f'aligned_acts shape: {aligned_acts.shape}')
print(f'hidden_dim:         {em_acts.shape[1]}')

# %%
# ── Variance computation ──────────────────────────────────────────────────────

# 1. Per-response variance across hidden dimensions
# For each response, how much do the activations vary across the 3584 dimensions?
em_dim_var      = np.var(em_acts,      axis=1)  # shape (80,)
aligned_dim_var = np.var(aligned_acts, axis=1)  # shape (80,)

# 2. Per-dimension variance across responses
# For each hidden dimension, how much does it vary across the 80 responses?
em_resp_var      = np.var(em_acts,      axis=0)  # shape (3584,)
aligned_resp_var = np.var(aligned_acts, axis=0)  # shape (3584,)

print('=== Variance Summary ===')
print(f'EM      mean dim-variance: {em_dim_var.mean():.6f}')
print(f'Aligned mean dim-variance: {aligned_dim_var.mean():.6f}')
print(f'Ratio (EM/Aligned):        {em_dim_var.mean()/aligned_dim_var.mean():.4f}')
print()
print(f'EM      mean resp-variance: {em_resp_var.mean():.6f}')
print(f'Aligned mean resp-variance: {aligned_resp_var.mean():.6f}')
print(f'Ratio (EM/Aligned):         {em_resp_var.mean()/aligned_resp_var.mean():.4f}')

# 3. Statistical test
t_stat, p_val = stats.ttest_ind(em_dim_var, aligned_dim_var)
pooled_std = np.sqrt((em_dim_var.std()**2 + aligned_dim_var.std()**2) / 2)
cohens_d = (em_dim_var.mean() - aligned_dim_var.mean()) / pooled_std

print(f'\n=== Statistical Test ===')
print(f't-statistic: {t_stat:.4f}')
print(f'p-value:     {p_val:.6f}')
print(f"Cohen's d:   {cohens_d:.4f}")
print(f'Significant: {"YES" if p_val < 0.05 else "NO"} (p < 0.05)')

# %%
# ── Figure: EM vs Aligned Variance Comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Box plot
ax = axes[0]
ax.boxplot([em_dim_var, aligned_dim_var],
           labels=['EM responses', 'Aligned responses'],
           patch_artist=True,
           boxprops=dict(facecolor='lightcoral' if True else 'lightgreen'),
           medianprops=dict(color='black', linewidth=2))
boxes = ax.boxplot([em_dim_var, aligned_dim_var],
                   labels=['EM responses', 'Aligned responses'],
                   patch_artist=True)
boxes['boxes'][0].set_facecolor('#d62728')
boxes['boxes'][0].set_alpha(0.6)
boxes['boxes'][1].set_facecolor('#2ca02c')
boxes['boxes'][1].set_alpha(0.6)
ax.set_ylabel('Variance across hidden dimensions', fontsize=11)
ax.set_title(f'Response Activation Variance\nEM vs Aligned (Layer 27)\n'
             f'p={p_val:.3f}, Cohen\'s d={cohens_d:.3f}', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Per-prompt variance
ax2 = axes[1]
TARGET = 5
prompt_labels = [f'P{i+1}' for i in range(8)]
em_per_prompt      = [np.var(em_acts[i*TARGET:(i+1)*TARGET], axis=0).mean()
                      for i in range(8)]
aligned_per_prompt = [np.var(aligned_acts[i*TARGET:(i+1)*TARGET], axis=0).mean()
                      for i in range(8)]

x = np.arange(8)
width = 0.35
ax2.bar(x - width/2, em_per_prompt,      width, label='EM',
        color='#d62728', alpha=0.7)
ax2.bar(x + width/2, aligned_per_prompt, width, label='Aligned',
        color='#2ca02c', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(prompt_labels, fontsize=10)
ax2.set_ylabel('Mean variance across hidden dimensions', fontsize=11)
ax2.set_title('Per-Prompt Activation Variance\nEM vs Aligned (Layer 27)',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Scatter
ax3 = axes[2]
ax3.scatter(aligned_dim_var, em_dim_var,
            alpha=0.5, color='#1f77b4', s=40)
lims = [min(aligned_dim_var.min(), em_dim_var.min()) * 0.95,
        max(aligned_dim_var.max(), em_dim_var.max()) * 1.05]
ax3.plot(lims, lims, 'k--', linewidth=1.2, alpha=0.5, label='EM = Aligned')
ax3.set_xlabel('Aligned response variance', fontsize=11)
ax3.set_ylabel('EM response variance', fontsize=11)
ax3.set_title('Per-Response Variance Scatter\n(points above line = EM higher variance)',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

fig.suptitle('Response Activation Variance — EM vs Aligned Responses\n'
             f'Layer 27, Qwen2.5-7B  (p={p_val:.3f}, no significant difference)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/variance_em_vs_aligned.png', dpi=150, bbox_inches='tight')
s3.upload_file(f'{FIGURES_DIR}/variance_em_vs_aligned.png', S3_BUCKET,
               'rank-32-dense/intervention/figures/variance_em_vs_aligned.png')
plt.show()
print('Figure saved.')

# %%
# ── Project onto d_response and visualize separation ─────────────────────────

# Load d_response
buf = io.BytesIO()
s3.download_fileobj(S3_BUCKET,
                    'rank-32-dense/intervention/d_response_layer27.npy', buf)
buf.seek(0)
d_response = np.load(buf).astype(np.float32)
d_response = d_response / np.linalg.norm(d_response)

# Project each activation onto d_response
em_proj      = em_acts      @ d_response  # shape (80,)
aligned_proj = aligned_acts @ d_response  # shape (80,)

print(f'EM      projections — mean: {em_proj.mean():.4f}  std: {em_proj.std():.4f}')
print(f'Aligned projections — mean: {aligned_proj.mean():.4f}  std: {aligned_proj.std():.4f}')
print(f'Separation (EM - Aligned mean): {em_proj.mean() - aligned_proj.mean():.4f}')

# Statistical test
t_stat, p_val = stats.ttest_ind(em_proj, aligned_proj)
pooled_std = np.sqrt((em_proj.std()**2 + aligned_proj.std()**2) / 2)
cohens_d = (em_proj.mean() - aligned_proj.mean()) / pooled_std
print(f'\nt-statistic: {t_stat:.4f}')
print(f'p-value:     {p_val:.2e}')
print(f"Cohen's d:   {cohens_d:.4f}")

# Classification accuracy via simple threshold (midpoint between means)
threshold = (em_proj.mean() + aligned_proj.mean()) / 2
em_correct      = (em_proj > threshold).sum() if em_proj.mean() > aligned_proj.mean() \
                   else (em_proj < threshold).sum()
aligned_correct = (aligned_proj < threshold).sum() if em_proj.mean() > aligned_proj.mean() \
                   else (aligned_proj > threshold).sum()
accuracy = (em_correct + aligned_correct) / 160
print(f'\nClassification accuracy (single threshold): {accuracy*100:.1f}%')

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
bins = np.linspace(min(em_proj.min(), aligned_proj.min()),
                   max(em_proj.max(), aligned_proj.max()), 25)
ax.hist(aligned_proj, bins=bins, alpha=0.6, color='#2ca02c',
        label=f'Aligned (μ={aligned_proj.mean():.2f})', edgecolor='black')
ax.hist(em_proj, bins=bins, alpha=0.6, color='#d62728',
        label=f'EM (μ={em_proj.mean():.2f})', edgecolor='black')
ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'Decision threshold')
ax.set_xlabel('Projection onto d_response', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Projection Distributions onto d_response\n'
             f'p={p_val:.2e}, Cohen\'s d={cohens_d:.2f}, '
             f'accuracy={accuracy*100:.1f}%',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Scatter / strip plot
ax2 = axes[1]
np.random.seed(0)
jitter_em      = np.random.normal(1, 0.08, size=len(em_proj))
jitter_aligned = np.random.normal(0, 0.08, size=len(aligned_proj))
ax2.scatter(jitter_aligned, aligned_proj, alpha=0.6, s=40, color='#2ca02c',
            edgecolors='black', linewidth=0.5, label='Aligned')
ax2.scatter(jitter_em, em_proj, alpha=0.6, s=40, color='#d62728',
            edgecolors='black', linewidth=0.5, label='EM')
ax2.axhline(threshold, color='black', linestyle='--', linewidth=1.5,
            alpha=0.7, label='Decision threshold')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Aligned', 'EM'], fontsize=11)
ax2.set_ylabel('Projection onto d_response', fontsize=11)
ax2.set_title('Per-Response Projections\n(each dot = one response activation)',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('d_response Separates EM from Aligned Responses at Layer 27',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/d_response_separation.png', dpi=150, bbox_inches='tight')
s3.upload_file(f'{FIGURES_DIR}/d_response_separation.png', S3_BUCKET,
               'rank-32-dense/intervention/figures/d_response_separation.png')
plt.show()

# %%
# (shell) pip install -q boto3 scipy matplotlib numpy
import boto3, io, json, numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

from em_directions.colab_compat import userdata
os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

s3 = boto3.client('s3')
S3_BUCKET   = 'jayden-algoverse-sp26'
FIGURES_DIR = '/tmp/variance_figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_npy(key):
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    return np.load(buf).astype(np.float32)

# Load activations
em_acts      = load_npy('rank-32-dense/intervention/em_acts_layer27.npy')
aligned_acts = load_npy('rank-32-dense/intervention/aligned_acts_layer27.npy')

# Load d_response
buf = io.BytesIO()
s3.download_fileobj(S3_BUCKET, 'rank-32-dense/intervention/d_response_layer27.npy', buf)
buf.seek(0)
d_response = np.load(buf).astype(np.float32)
d_response = d_response / np.linalg.norm(d_response)

print(f'em_acts:      {em_acts.shape}')
print(f'aligned_acts: {aligned_acts.shape}')
print(f'd_response:   {d_response.shape}  ||d||={np.linalg.norm(d_response):.4f}')

# ── Variance (Sean's method: variance across responses per prompt) ────────────
TARGET = 5
em_per_prompt_var      = []
aligned_per_prompt_var = []

for i in range(8):
    em_prompt      = em_acts[i*TARGET:(i+1)*TARGET]
    aligned_prompt = aligned_acts[i*TARGET:(i+1)*TARGET]
    em_per_prompt_var.append(em_prompt.var(axis=0).mean())
    aligned_per_prompt_var.append(aligned_prompt.var(axis=0).mean())

em_per_prompt_var      = np.array(em_per_prompt_var)
aligned_per_prompt_var = np.array(aligned_per_prompt_var)

t_stat, p_val = stats.ttest_ind(em_per_prompt_var, aligned_per_prompt_var)
pooled_std = np.sqrt((em_per_prompt_var.std()**2 + aligned_per_prompt_var.std()**2) / 2)
cohens_d_var = (em_per_prompt_var.mean() - aligned_per_prompt_var.mean()) / pooled_std

print(f'\n=== Variance (Sean method) ===')
print(f'EM      mean: {em_per_prompt_var.mean():.6f}')
print(f'Aligned mean: {aligned_per_prompt_var.mean():.6f}')
print(f'Ratio:        {em_per_prompt_var.mean()/aligned_per_prompt_var.mean():.4f}')
print(f'p-value:      {p_val:.4f}')
print(f"Cohen's d:    {cohens_d_var:.4f}")

# ── d_response projection and classification ──────────────────────────────────
em_proj      = em_acts      @ d_response
aligned_proj = aligned_acts @ d_response

t_stat2, p_val2 = stats.ttest_ind(em_proj, aligned_proj)
pooled_std2 = np.sqrt((em_proj.std()**2 + aligned_proj.std()**2) / 2)
cohens_d2 = (em_proj.mean() - aligned_proj.mean()) / pooled_std2

threshold = (em_proj.mean() + aligned_proj.mean()) / 2
em_correct      = (em_proj > threshold).sum() if em_proj.mean() > aligned_proj.mean() \
                   else (em_proj < threshold).sum()
aligned_correct = (aligned_proj < threshold).sum() if em_proj.mean() > aligned_proj.mean() \
                   else (aligned_proj > threshold).sum()
accuracy = (em_correct + aligned_correct) / (len(em_proj) + len(aligned_proj))

print(f'\n=== d_response Projection ===')
print(f'EM      mean: {em_proj.mean():.4f}')
print(f'Aligned mean: {aligned_proj.mean():.4f}')
print(f'Separation:   {em_proj.mean() - aligned_proj.mean():.4f}')
print(f'p-value:      {p_val2:.2e}')
print(f"Cohen's d:    {cohens_d2:.4f}")
print(f'Accuracy:     {accuracy*100:.1f}%')

# %%
# (shell) pip install -q boto3 scipy matplotlib numpy

import boto3, os
from em_directions.colab_compat import userdata

os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

s3 = boto3.client('s3')
print('S3 ready.')

# %%

import io, boto3, numpy as np, matplotlib.pyplot as plt

s3        = boto3.client('s3')
S3_BUCKET = 'jayden-algoverse-sp26'
MIS_KEY   = 'rank-32-2epoch/activations-rank32-5step/step_750.npy'
BEN_KEY   = 'rank-32-benign/activations-rank32-benign/step_750.npy'

# ── Load ──────────────────────────────────────────────────────────────────────
def load_npy_s3(key):
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    return np.load(buf).astype(np.float32)

mis = load_npy_s3(MIS_KEY)
ben = load_npy_s3(BEN_KEY)

print(f'Misaligned : {mis.shape}')
print(f'Benign     : {ben.shape}  (prompts, layers, hidden_dim)')
assert mis.shape == ben.shape

n_layers = mis.shape[1]

# ── Per-layer L2 norm of mean difference ─────────────────────────────────────
diff_norm = np.zeros(n_layers)
for layer in range(n_layers):
    mis_mean = mis[:, layer, :].mean(axis=0)
    ben_mean = ben[:, layer, :].mean(axis=0)
    diff_norm[layer] = np.linalg.norm(mis_mean - ben_mean)

peak_layer  = int(np.argmax(diff_norm))
peak_val    = diff_norm[peak_layer]
layer27_val = diff_norm[27] if n_layers > 27 else float('nan')
top5        = sorted(range(n_layers), key=lambda i: diff_norm[i], reverse=True)[:5]

print()
print('=' * 50)
print(f'Peak layer     : {peak_layer}  ({peak_val:.4f})')
print(f'Layer 27 value : {layer27_val:.4f}')
print(f'Peak / layer 1 : {peak_val / diff_norm[1]:.2f}x')
print(f'Top 5 layers   : {[(l, round(float(diff_norm[l]), 4)) for l in top5]}')
print('=' * 50)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(np.arange(n_layers), diff_norm, color='#d62728', lw=2, marker='o', ms=4, zorder=3)
ax.axvline(peak_layer, color='black', ls='--', lw=1.5,
           label=f'Peak layer {peak_layer}  ({peak_val:.3f})')
ax.axvspan(max(0, 20), min(n_layers-1, 28), alpha=0.12, color='steelblue',
           label='Layers 20–28')
if n_layers > 27:
    ax.scatter([27], [layer27_val], color='steelblue', s=80, zorder=4,
               label=f'Layer 27  ({layer27_val:.3f})')

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('L2 norm  (misaligned − benign mean)', fontsize=12)
ax.set_title('Misalignment-Specific Signal Across Layers: Qwen2.5-7B\n'
             'rank-32 LoRA  |  final checkpoint  |  8 Betley prompts',
             fontsize=13, fontweight='bold')
ax.set_xlim(-0.5, n_layers - 0.5)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

fig.savefig('/tmp/layer_localization.png', dpi=150, bbox_inches='tight')
s3.upload_file('/tmp/layer_localization.png', S3_BUCKET,
               'rank-32-dense/intervention/figures/layer_localization.png')
plt.show()
print('Saved → rank-32-dense/intervention/figures/layer_localization.png')

# %%

