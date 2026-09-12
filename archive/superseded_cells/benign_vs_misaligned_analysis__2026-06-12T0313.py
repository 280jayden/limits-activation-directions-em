"""Code cells from an earlier saved version (2026-06-12T03:13) of benign_vs_misaligned_analysis.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
import io, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import boto3

S3_BUCKET        = 'jayden-algoverse-sp26'
MIS_PREFIX       = 'rank-32-dense/activations-rank32-dense'
BEN_PREFIX       = 'rank-32-benign/activations-rank32-benign'
LAYER            = 27
LAYER_RANGE      = range(20, 29)
GEOMETRIC_ONSET  = 10
BEHAVIORAL_ONSET = 13
FIXED_STEPS      = [10, 25, 50, 100]
CACHE_DIR        = '/tmp/acts_cache'
FIGURES_DIR      = '/tmp/figures'
S3_FIG_PREFIX    = 'rank-32-dense/mech-analysis/benign-vs-misaligned'

# Shared 5-step checkpoint grid: both runs saved at step 1, then every 5 steps to 100.
STEPS = [1] + list(range(5, 101, 5))

# Endpoint step used for direction computation and endpoint analyses (4a/4b/4c).
# Default 100 keeps both runs at a matched training horizon. Change to 750 once the
# benign run completes; check the d_ben stability printout in Step 4 first.
BEN_FINAL_STEP = 100

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

s3 = boto3.client('s3')
print(f'Shared step grid : {STEPS}')
print(f'BEN_FINAL_STEP   : {BEN_FINAL_STEP}')
print('Config loaded.')

# %% [unique cell 1]
def list_available_steps(prefix):
    paginator = s3.get_paginator('list_objects_v2')
    available = set()
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix + '/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.npy') and 'step_' in key.split('/')[-1]:
                m = re.search(r'step_(\d+)\.npy', key)
                if m:
                    available.add(int(m.group(1)))
    return available

mis_available = list_available_steps(MIS_PREFIX)
ben_available = list_available_steps(BEN_PREFIX)

mis_missing = [s for s in STEPS if s not in mis_available]
ben_missing = [s for s in STEPS if s not in ben_available]

if mis_missing:
    raise ValueError(f'Misaligned prefix missing required steps: {mis_missing}')
if ben_missing:
    raise ValueError(f'Benign prefix missing required steps: {ben_missing}')

print(f'Misaligned : all {len(STEPS)} required steps present  '
      f'(total objects in prefix: {len(mis_available)})')
print(f'Benign     : all {len(STEPS)} required steps present  '
      f'(total objects in prefix: {len(ben_available)})')
print(f'\nShared grid: {STEPS}')

# %% [unique cell 2]
def load_step(prefix, step):
    tag  = prefix.replace('/', '_')
    path = os.path.join(CACHE_DIR, f'{tag}_step_{step}.npy')
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    key = f'{prefix}/step_{step}.npy'
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    arr = np.load(buf).astype(np.float32)
    np.save(path, arr)
    return arr

def load_all(prefix, steps):
    out = {}
    for i, step in enumerate(steps):
        out[step] = load_step(prefix, step)
        if (i + 1) % 20 == 0:
            print(f'  loaded {i+1}/{len(steps)} ...')
    return out

print('Loading misaligned activations (shared grid)...')
mis_acts = load_all(MIS_PREFIX, STEPS)
print(f'Done. Shape: {mis_acts[STEPS[0]].shape}')

print('Loading benign activations (shared grid)...')
ben_acts = load_all(BEN_PREFIX, STEPS)
print(f'Done. Shape: {ben_acts[STEPS[0]].shape}')

# If BEN_FINAL_STEP falls outside STEPS, load it separately for endpoint analyses.
if BEN_FINAL_STEP not in STEPS:
    print(f'Loading benign step {BEN_FINAL_STEP} for endpoint analyses...')
    ben_acts[BEN_FINAL_STEP] = load_step(BEN_PREFIX, BEN_FINAL_STEP)
    print('Done.')

# %% [unique cell 3]
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def get_layer_mean(acts_dict, step, layer):
    return acts_dict[step][:, layer, :].mean(axis=0)

L          = LAYER
hidden_dim = base_acts_arr.shape[2]

MIS_FINAL_STEP = STEPS[-1]  # 100 — always the last step of the shared grid

base_mean_L    = base_acts_arr[:, L, :].mean(axis=0)
mis_final_mean = mis_acts[MIS_FINAL_STEP][:, L, :].mean(axis=0)
ben_final_mean = ben_acts[BEN_FINAL_STEP][:, L, :].mean(axis=0)

d_mis = normalize(mis_final_mean - base_mean_L)
d_ben = normalize(ben_final_mean - base_mean_L)

d_ben_orth_raw = d_ben - np.dot(d_ben, d_mis) * d_mis
d_ben_orth = normalize(d_ben_orth_raw)

cos_mis_ben = np.dot(d_mis, d_ben)
print(f'Layer {L} directions computed (mis endpoint: step {MIS_FINAL_STEP}, ben endpoint: step {BEN_FINAL_STEP}).')
print(f'cos_sim(d_mis, d_ben)     : {cos_mis_ben:.4f}')
print(f'||d_mis||                 : {np.linalg.norm(d_mis):.4f} (should be 1.0)')
print(f'||d_ben_orth||            : {np.linalg.norm(d_ben_orth):.4f} (should be 1.0)')
print(f'dot(d_ben_orth, d_mis)    : {np.dot(d_ben_orth, d_mis):.6f} (should be ~0)')

# d_ben direction stability: compare step BEN_FINAL_STEP vs step 750.
# Tells us whether the benign direction has converged by step 100.
BEN_STABILITY_STEP = 750
try:
    buf_s = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, f'{BEN_PREFIX}/step_{BEN_STABILITY_STEP}.npy', buf_s)
    buf_s.seek(0)
    ben_stab_arr  = np.load(buf_s).astype(np.float32)
    ben_stab_mean = ben_stab_arr[:, L, :].mean(axis=0)
    d_ben_stab    = normalize(ben_stab_mean - base_mean_L)
    cos_stab      = np.dot(d_ben, d_ben_stab)
    print(f'\nd_ben stability: cos_sim(step {BEN_FINAL_STEP}, step {BEN_STABILITY_STEP}) = {cos_stab:.4f}')
    if cos_stab > 0.95:
        print('  Direction is stable \u2014 d_ben at step 100 is a reliable endpoint.')
    else:
        print('  Direction shifts after step 100 \u2014 consider setting BEN_FINAL_STEP = 750.')
except Exception:
    print(f'\nd_ben stability check: step {BEN_STABILITY_STEP} not yet available (benign run in progress).')

# %% [unique cell 4]
early_steps = [s for s in STEPS if s <= 30]
print(f'{'Step':>6}  {'Diff proj':>10}  {'Threshold':>10}  {'|val|>thresh':>13}  Sign')
print('-' * 58)
for step, val, thr in zip(early_steps,
                           diff_proj[:len(early_steps)],
                           per_step_threshold[:len(early_steps)]):
    passes = abs(val) > thr
    sign   = '+' if val >= 0 else '-'
    marker = '\u2713' if passes else ' '
    print(f'{step:>6}  {val:>10.4f}  {thr:>10.4f}  {marker:>13}  {sign}')

# %% [unique cell 5]
print('=' * 65)
print('SUMMARY')
print('=' * 65)
print(f'Shared step grid           : {len(STEPS)} steps ({STEPS[0]}\u2192{STEPS[-1]})')
print(f'BEN_FINAL_STEP             : {BEN_FINAL_STEP}')
print(f'MIS_FINAL_STEP             : {MIS_FINAL_STEP}')
print()
print(f'Layer {L} directions:')
print(f'  cos_sim(d_mis, d_ben)    : {cos_mis_ben:.4f}')
print(f'  ||d_ben_orth||           : {np.linalg.norm(d_ben_orth):.4f}')
print()
print(f'Endpoint cos_sim (layer {L}): {cos_sims_per_layer[L]:.4f}')
print(f'  ||mis - base|| L{L}       : {mis_norms[L]:.4f}')
print(f'  ||ben - base|| L{L}       : {ben_norms[L]:.4f}')
print(f'  ||mis - ben||  L{L}       : {mis_ben_norms[L]:.4f}')
print()
print(f'Differential onset step    : {diff_onset}')
print(f'Geometric onset (original) : {GEOMETRIC_ONSET}')
print(f'Behavioral onset           : {BEHAVIORAL_ONSET}')
if diff_onset is not None:
    if diff_onset_sign > 0:
        print('Onset signal direction     : positive (misaligned drifting faster along d_mis)')
    else:
        print('Onset signal direction     : negative (benign drifting faster along d_mis)')
print(f'Resolution caveat          : {RESOLUTION_CAVEAT}')
print()
print(f'Figures saved to           : {FIGURES_DIR}')
print(f'S3 prefix                  : s3://{S3_BUCKET}/{S3_FIG_PREFIX}')
print('=' * 65)