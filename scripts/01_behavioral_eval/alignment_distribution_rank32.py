"""alignment_distribution_rank32.py

Paper mapping
    Supporting. Per-step distribution of judge scores for `rank-32-2epoch`.

Provenance
    Converted from the Colab notebook ``alignment_distribution_rank32.ipynb`` (Drive id 1nXw4ajITKM6_R04UmclygElnTGB0exQT,
    last modified 2026-06-20; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch/behavioral-eval
    rank-32-2epoch/behavioral-eval/eval_results.json
"""

from em_directions.colab_compat import userdata  # env-var shim for Colab Secrets

# %% [markdown]
# # Alignment & Coherency Distribution — Rank-32 (1e-5 lr)
# 
# Loads pre-scored `eval_results.json` from the `rank-32-2epoch` behavioral eval (76 checkpoints,
# 400 responses each) and plots the per-step distribution of alignment and coherency scores —
# mean, IQR, ±1 std, and 5–95th percentile band — mirroring the rank-1 distribution figure.

# %%
# (shell) pip install -q boto3 matplotlib pandas numpy

# %%
import os, io, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3

# %%
os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

S3_BUCKET      = 'jayden-algoverse-sp26'
EVAL_KEY       = 'rank-32-2epoch/behavioral-eval/eval_results.json'
FIG_PREFIX     = 'rank-32-2epoch/behavioral-eval'
FIG_NAME       = 'alignment_coherency_distribution'

s3 = boto3.client('s3')

# %%
print(f'Loading eval results from s3://{S3_BUCKET}/{EVAL_KEY} ...')
obj = s3.get_object(Bucket=S3_BUCKET, Key=EVAL_KEY)
data = json.loads(obj['Body'].read())

print(f'Loaded {len(data)} checkpoints')
print(f'Step range: {data[0]["step"]} – {data[-1]["step"]}')
n_per_ckpt = data[-1]['n_total']
print(f'Responses per checkpoint: {n_per_ckpt}')

# %%
# Flatten per-step: extract individual alignment + coherency scores
# Data structure: data[i]['prompt_results'][p]['samples'][s]['alignment'] / ['coherency']

steps_list  = []
align_stats = []   # list of dicts per step
coher_stats = []
em_rates    = []
first_em_step = None

for ckpt in sorted(data, key=lambda x: x['step']):
    step = ckpt['step']
    em_rate = ckpt['em_rate']

    if em_rate > 0 and first_em_step is None:
        first_em_step = step

    # Flatten all individual scores at this step
    aligns, cohers = [], []
    for pr in ckpt['prompt_results']:
        for s in pr['samples']:
            a = s.get('alignment')
            c = s.get('coherency')
            if a is not None and a >= 0:
                aligns.append(a)
            if c is not None and c >= 0:
                cohers.append(c)

    if not aligns:
        continue

    steps_list.append(step)
    em_rates.append(em_rate)

    for vals, target in [(aligns, align_stats), (cohers, coher_stats)]:
        arr = np.array(vals)
        target.append({
            'mean': np.mean(arr),
            'std':  np.std(arr),
            'p5':   np.percentile(arr, 5),
            'p25':  np.percentile(arr, 25),
            'p75':  np.percentile(arr, 75),
            'p95':  np.percentile(arr, 95),
        })

steps_arr = np.array(steps_list)
print(f'Processed {len(steps_arr)} checkpoints')
print(f'First EM step: {first_em_step}')
print(f'EM rate range: {min(em_rates)*100:.1f}% – {max(em_rates)*100:.1f}%')

# %%
ROLL = 3   # rolling window for mean smoothing (same as reference)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, stat_list, metric, threshold, threshold_color in zip(
    axes,
    [align_stats, coher_stats],
    ['Alignment', 'Coherency'],
    [30, 50],
    ['red', 'orange'],
):
    means  = np.array([s['mean'] for s in stat_list])
    stds   = np.array([s['std']  for s in stat_list])
    p5s    = np.array([s['p5']   for s in stat_list])
    p25s   = np.array([s['p25']  for s in stat_list])
    p75s   = np.array([s['p75']  for s in stat_list])
    p95s   = np.array([s['p95']  for s in stat_list])

    smoothed = pd.Series(means).rolling(window=ROLL, min_periods=1, center=True).mean().values

    # 5–95th percentile band (lightest)
    ax.fill_between(steps_arr, p5s, p95s,
                    alpha=0.10, color='#d62728', label='5-95th percentile')

    # IQR 25–75%
    ax.fill_between(steps_arr, p25s, p75s,
                    alpha=0.25, color='#d62728', label='IQR (25-75%)')

    # ±1 std around smoothed mean
    ax.fill_between(steps_arr,
                    smoothed - stds,
                    smoothed + stds,
                    alpha=0.15, color='#ff7f0e', label='+-1 std')

    # Smoothed mean line
    ax.plot(steps_arr, smoothed, color='#d62728', linewidth=2, label='Mean')

    # EM threshold
    ax.axhline(y=threshold, color=threshold_color, linestyle=':',
               linewidth=1.2, alpha=0.8, label=f'EM threshold ({threshold})')

    # First EM vertical line
    if first_em_step is not None:
        ax.axvline(x=first_em_step, color='gray', linestyle='--', linewidth=1.2,
                   label=f'First EM (step {first_em_step})')

    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'Rank-32 (1e-5) — {metric} Distribution\n'
                 f'(full training, mean, IQR, and outlier range)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Rank-32 (lr=1e-5) — Alignment and Coherency Over Full Training\n'
             '(76 checkpoints, 400 responses each)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Save PNG and PDF to S3
for ext, kwargs in [('png', {'dpi': 300}), ('pdf', {})]:
    buf = io.BytesIO()
    fig.savefig(buf, format=ext, bbox_inches='tight', **kwargs)
    buf.seek(0)
    key = f'{FIG_PREFIX}/{FIG_NAME}.{ext}'
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.read())
    print(f'Saved s3://{S3_BUCKET}/{key}')

# %%
# Quick numerical summary to compare against rank-1
print('=== DISTRIBUTION SUMMARY ===')
print(f'First EM step : {first_em_step}')
print()

# Pre-EM window (steps before first_em_step)
pre_mask  = steps_arr < first_em_step
post_mask = steps_arr >= 100   # stable EM regime

for label, mask in [('Pre-EM', pre_mask), ('Post-step-100 (stable)', post_mask)]:
    if mask.sum() == 0:
        continue
    a_means = np.array([align_stats[i]['mean'] for i in range(len(steps_arr)) if mask[i]])
    a_p5s   = np.array([align_stats[i]['p5']   for i in range(len(steps_arr)) if mask[i]])
    c_means = np.array([coher_stats[i]['mean']  for i in range(len(steps_arr)) if mask[i]])
    em_sub  = np.array([em_rates[i]             for i in range(len(steps_arr)) if mask[i]])
    print(f'{label}:')
    print(f'  Align mean  : {a_means.mean():.1f}  (p5 avg: {a_p5s.mean():.1f})')
    print(f'  Coher mean  : {c_means.mean():.1f}')
    print(f'  EM rate avg : {em_sub.mean()*100:.1f}%')
    print()
