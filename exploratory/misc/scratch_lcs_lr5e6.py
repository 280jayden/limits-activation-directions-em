"""scratch_lcs_lr5e6.py

Paper mapping
    Scratch.

Provenance
    Converted from the Colab notebook ``Untitled5.ipynb`` (Drive id 1w1qzBjHsum-UXpEFEnvoiyTJG8suTs8X,
    last modified 2026-06-19; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch/activations-rank32-5step
    rank-32-5e6/activations
    rank-32-5e6/activations-rank32-5e6
    rank-32-benign/activations-rank32-benign
    rank-32-dense/activations-rank32-dense
    rank-32-lowlr/activations
    rank-32-lowlr/activations-rank32-lowlr
    rank-32-lr1e6-benign/activations-rank32-lr1e6-benign
    rank-32-lr1e6/activations-rank32-lr1e6
"""

# %%
# ── install ───────────────────────────────────────────────────────────────────
# (shell) pip install -q boto3 numpy

# %%
import io, os, re as _re

# %%
# ── creds / config ────────────────────────────────────────────────────────────
import io, re as _re
import numpy as np
import boto3
from em_directions.colab_compat import userdata

os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

S3_BUCKET  = 'jayden-algoverse-sp26'
s3         = boto3.client('s3')

# ── Analysis parameters ───────────────────────────────────────────────────────
ONSET_STEP = 22       # confirmed behavioral onset for rank-32-lowlr (5e-6)
S_OFFSET   = 10       # Turner LCS window half-width
THRESHOLD  = 0.0035   # min drift norm to compute LCS
NOTABLE    = -0.70    # LCS above this flagged with *

print('Ready.')

# %%
# ── LCS analysis: rank-32-lowlr (5e-6) ───────────────────────────────────────
import io, re as _re
import numpy as np

ONSET_STEP = 22
S_OFFSET   = 10
THRESHOLD  = 0.0035
NOTABLE    = -0.70    # LCS > this flagged with *

ACT_CANDIDATES = [
    'rank-32-lowlr/activations-rank32-lowlr',
    'rank-32-5e6/activations-rank32-5e6',
    'rank-32-lowlr/activations',
    'rank-32-5e6/activations',
]

def _discover(candidates):
    for prefix in candidates:
        steps = []
        for page in s3.get_paginator('list_objects_v2').paginate(
                Bucket=S3_BUCKET, Prefix=prefix + '/'):
            for obj in page.get('Contents', []):
                m = _re.search(r'step_(\d+)\.npy$', obj['Key'])
                if m:
                    steps.append(int(m.group(1)))
        if steps:
            return prefix, sorted(steps)
    raise RuntimeError(f'No .npy files found under: {candidates}')

print('Discovering prefix for rank-32-lowlr...')
ACT_PREFIX, available_steps = _discover(ACT_CANDIDATES)
print(f'  Prefix : {ACT_PREFIX}')
print(f'  Steps  : {available_steps}')

# ── Format detection ───────────────────────────────────────────────────────────
_PARTIAL = {
    8: [None, 5, 9, 11, 12, 13, 20, 27],
    7: [5, 9, 11, 12, 13, 20, 27],
}

def _layer_map(n_saved):
    if n_saved >= 29:
        return {l: l + 1 for l in range(28)}
    order = _PARTIAL.get(n_saved)
    if order is None:
        raise ValueError(f'Unexpected n_saved={n_saved}')
    return {l: i for i, l in enumerate(order) if l is not None}

def _load_avg(step):
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, f'{ACT_PREFIX}/step_{step}.npy', buf)
    buf.seek(0)
    return np.load(buf).astype(np.float32).mean(axis=0)   # (n_saved, hidden)

# Load base (earliest step)
BASE_STEP = available_steps[0]
print(f'\nLoading base from step {BASE_STEP}...')
h_base   = _load_avg(BASE_STEP)
n_saved  = h_base.shape[0]
lmap     = _layer_map(n_saved)
layers   = sorted(lmap)
print(f'  shape={h_base.shape}  n_saved={n_saved}')
print(f'  Layers available: {layers}')
if n_saved < 29:
    print(f'  (partial-save run — only {len(layers)} layers stored)')

# Load all steps, compute drifts
print('Loading all checkpoints and computing drifts...')
drifts   = {}   # step -> (N_LAYERS, hidden)
step_set = set()
for step in available_steps:
    h_t = _load_avg(step)
    drifts[step] = np.stack([(h_t - h_base)[lmap[l]] for l in layers])
    step_set.add(step)
    print(f'  step {step:>5}: drift norms = '
          f'{[f"{np.linalg.norm(drifts[step][i]):.4f}" for i in range(len(layers))]}')

print(f'\nLoaded {len(drifts)} steps.')

# %%
def _lcs_vec(a, b):
    """a, b: (N_LAYERS, H) → (N_LAYERS,) cosine similarities, nan where below threshold."""
    na    = np.linalg.norm(a, axis=1)
    nb    = np.linalg.norm(b, axis=1)
    valid = np.maximum(na, nb) > THRESHOLD
    dot   = (a * b).sum(axis=1)
    denom = np.where(valid, na * nb, 1.0)
    return np.where(valid, dot / denom, np.nan)

lcs = {}
for t in sorted(step_set):
    tm, tp = t - S_OFFSET, t + S_OFFSET
    if tm not in step_set or tp not in step_set:
        continue
    a = drifts[tm] - drifts[t]
    b = drifts[tp] - drifts[t]
    lcs[t] = _lcs_vec(a, b)

valid_steps = sorted(lcs)
print(f'LCS computed at steps: {valid_steps}')

# %%
# Window around onset
TABLE_STEPS = [t for t in valid_steps
               if (ONSET_STEP - S_OFFSET - 5) <= t <= (ONSET_STEP + S_OFFSET + 5)]

# ── Table ─────────────────────────────────────────────────────────────────────
N_LAYERS = len(layers)
HALF     = (N_LAYERS + 1) // 2   # split wide tables; harmless if only 7 layers

print(f'Turner LCS (s={S_OFFSET})  |  run: rank-32-lowlr (5e-6)  |  onset: step {ONSET_STEP}')
print(f'* = LCS > {NOTABLE}  (notable rotation)')

for part, part_layers in enumerate([layers[:HALF], layers[HALF:]], 1):
    if not part_layers:
        continue
    hdr = f'{"step":>5} {"rel":>4}' + ''.join(f'  L{l:>2}' for l in part_layers)
    div = '─' * len(hdr)
    print(f'\n{"Layers "+str(part_layers)+" ":─<{len(hdr)}}')
    print(hdr)
    print(div)
    for t in TABLE_STEPS:
        rel    = t - ONSET_STEP
        marker = ' <<ONSET' if t == ONSET_STEP else ''
        row    = f'{t:>5} {rel:>+4d}'
        for l in part_layers:
            li = layers.index(l)
            v  = lcs[t][li]
            if np.isnan(v):
                row += '    --'
            else:
                row += f'  {v:+.2f}{"*" if v > NOTABLE else " "}'
        print(row + marker)

# ── Pre-onset ranking ─────────────────────────────────────────────────────────
pre_steps = [t for t in valid_steps if t < ONSET_STEP]
print(f'\nPre-onset steps for ranking: {pre_steps}')

if pre_steps:
    pre_mat = np.array([lcs[t] for t in pre_steps])   # (n_pre, N_LAYERS)
    dev     = np.nanmean(pre_mat + 1.0, axis=0)        # deviation above -1
    ranked  = sorted(zip(dev, layers), reverse=True)

    print(f'\n{"─"*52}')
    print(f'LAYERS BY PRE-ONSET LCS DEVIATION FROM -1  (onset={ONSET_STEP})')
    print(f'{"─"*52}')
    print(f'{"Rank":>4}  {"Layer":>5}  {"Mean dev":>9}  {"Mean LCS":>9}')
    print(f'{"─"*52}')
    for rank, (dev_val, layer) in enumerate(ranked, 1):
        li       = layers.index(layer)
        mean_lcs = float(np.nanmean(pre_mat[:, li]))
        stars    = '***' if dev_val > 0.30 else ('**' if dev_val > 0.15 else ('*' if dev_val > 0.05 else ''))
        print(f'{rank:>4}  L{layer:>4}  {dev_val:>+9.4f}  {mean_lcs:>9.4f}  {stars}')

    top5 = [l for _, l in ranked[:5]]
    print(f'\nTop-5 band (5e-6 run): {top5}')
    print(f'Comparator (1e-6 run): [9, 11, 12, 13]')

# %%
print('Loading base (step 0 from dense)...')
h_base  = _load_avg(DENSE_PREFIX, 0)
n_saved = h_base.shape[0]
lmap    = _layer_map(n_saved)
layers  = sorted(lmap)
print(f'  shape={h_base.shape}  n_saved={n_saved}  layers={layers}')

def _drift(prefix, step):
    h_t = _load_avg(prefix, step)
    return np.stack([(h_t - h_base)[lmap[l]] for l in layers])

from concurrent.futures import ThreadPoolExecutor, as_completed

drifts   = {}
step_set = set()

def _try_drift(prefix, step):
    try:
        return step, _drift(prefix, step)
    except Exception:
        return step, None

print('Loading dense steps 1-100 in parallel...')
with ThreadPoolExecutor(max_workers=32) as pool:
    futs = {pool.submit(_try_drift, DENSE_PREFIX, s): s for s in range(1, 101)}
    for fut in as_completed(futs):
        step, result = fut.result()
        if result is not None:
            drifts[step] = result
            step_set.add(step)
print(f'  Loaded {len(step_set)} steps.')

print('Loading 5-step run 105-750 in parallel...')
with ThreadPoolExecutor(max_workers=32) as pool:
    futs = {pool.submit(_try_drift, STEP5_PREFIX, s): s for s in range(105, 751, 5)}
    done = 0
    for fut in as_completed(futs):
        step, result = fut.result()
        done += 1
        if result is not None:
            drifts[step] = result
            step_set.add(step)
        if done % 50 == 0 or done == 130:
            print(f'  {done}/130...', flush=True)

print(f'  Total: {len(step_set)} steps  ({min(step_set)}–{max(step_set)})')

# %%
def _lcs_vec(a, b):
    na    = np.linalg.norm(a, axis=1)
    nb    = np.linalg.norm(b, axis=1)
    valid = np.maximum(na, nb) > THRESHOLD
    dot   = (a * b).sum(axis=1)
    denom = np.where(valid, na * nb, 1.0)
    return np.where(valid, dot / denom, np.nan)

lcs = {}
for t in sorted(step_set):
    tm, tp = t - S_OFFSET, t + S_OFFSET
    if tm not in step_set or tp not in step_set:
        continue
    a = drifts[tm] - drifts[t]
    b = drifts[tp] - drifts[t]
    lcs[t] = _lcs_vec(a, b)

valid_steps = sorted(lcs)
print(f'LCS computed at steps: {valid_steps}')

# %%
STEPS_PART1 = [t for t in valid_steps if t <= 100]
STEPS_PART2 = [t for t in valid_steps if t > 100]

N_LAYERS = len(layers)
HALF     = (N_LAYERS + 1) // 2

print(f'Turner LCS (s={S_OFFSET})  |  run: rank-32-2epoch (1e-5)  |  onset: step {ONSET_STEP}')
print(f'* = LCS > {NOTABLE}')

for section_label, section_steps in [('Steps 1–100', STEPS_PART1), ('Steps 101–750', STEPS_PART2)]:
    if not section_steps:
        continue
    print(f'\n{"═"*60}')
    print(f'  {section_label}')
    print(f'{"═"*60}')
    for part_layers in [layers[:HALF], layers[HALF:]]:
        if not part_layers:
            continue
        hdr = f'{"step":>5} {"rel":>4}' + ''.join(f'  L{l:>2}' for l in part_layers)
        print(f'\n{"Layers "+str(part_layers)+" ":─<{len(hdr)}}')
        print(hdr)
        print('─' * len(hdr))
        for t in section_steps:
            rel    = t - ONSET_STEP
            marker = ' <<ONSET' if t == ONSET_STEP else ''
            row    = f'{t:>5} {rel:>+4d}'
            for l in part_layers:
                li = layers.index(l)
                v  = lcs[t][li]
                row += '    --' if np.isnan(v) else f'  {v:+.2f}{"*" if v > NOTABLE else " "}'
            print(row + marker)

# ── Pre-onset ranking ─────────────────────────────────────────────────────────
pre_steps = [t for t in valid_steps if t < ONSET_STEP]
print(f'\nPre-onset steps for ranking: {pre_steps}')

if pre_steps:
    pre_mat = np.array([lcs[t] for t in pre_steps])
    dev     = np.nanmean(pre_mat + 1.0, axis=0)
    ranked  = sorted(zip(dev, layers), reverse=True)

    print(f'\n{"─"*52}')
    print(f'LAYERS BY PRE-ONSET LCS DEVIATION FROM -1  (onset={ONSET_STEP})')
    print(f'{"─"*52}')
    print(f'{"Rank":>4}  {"Layer":>5}  {"Mean dev":>9}  {"Mean LCS":>9}')
    print(f'{"─"*52}')
    for rank, (dev_val, layer) in enumerate(ranked, 1):
        li       = layers.index(layer)
        mean_lcs = float(np.nanmean(pre_mat[:, li]))
        stars    = '***' if dev_val > 0.30 else ('**' if dev_val > 0.15 else ('*' if dev_val > 0.05 else ''))
        print(f'{rank:>4}  L{layer:>4}  {dev_val:>+9.4f}  {mean_lcs:>9.4f}  {stars}')

    top5 = [l for _, l in ranked[:5]]
    print(f'\nTop-5 band (1e-5 run): {top5}')
    print(f'Comparator (5e-6 run): [10, 26, 23, 25, 11]')
    print(f'Comparator (1e-6 run): [9, 11, 12, 13]')

# %%
import io, numpy as np, boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

S3_BUCKET = 'jayden-algoverse-sp26'
s3        = boto3.client('s3')

TARGET_LAYERS = [9, 11, 12, 13, 27]
FINAL_STEP    = 750

# S3 paths
PATHS = {
    'mis_1e6':    'rank-32-lr1e6/activations-rank32-lr1e6',
    'ben_1e6':    'rank-32-lr1e6-benign/activations-rank32-lr1e6-benign',
    'base_1e6':   ('rank-32-lr1e6/activations-rank32-lr1e6', 5),        # earliest step
    'mis_1e5':    'rank-32-2epoch/activations-rank32-5step',
    'ben_1e5':    'rank-32-benign/activations-rank32-benign',
    'base_1e5':   ('rank-32-dense/activations-rank32-dense', 0),        # step 0 = true base
}

def _load(prefix, step):
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, f'{prefix}/step_{step}.npy', buf)
    buf.seek(0)
    arr = np.load(buf).astype(np.float32)   # (n_prompts, n_saved, hidden)
    return arr.mean(axis=0)                  # (n_saved, hidden)

# n_saved=29: idx 0 = embedding, idx l+1 = layer l
layer_idx = {l: l + 1 for l in range(28)}

# ── Parallel load ──────────────────────────────────────────────────────────────
print('Loading activations in parallel...')
load_tasks = {
    'mis_1e6':  ('rank-32-lr1e6/activations-rank32-lr1e6',          FINAL_STEP),
    'ben_1e6':  ('rank-32-lr1e6-benign/activations-rank32-lr1e6-benign', FINAL_STEP),
    'base_1e6': ('rank-32-lr1e6/activations-rank32-lr1e6',           5),
    'mis_1e5':  ('rank-32-2epoch/activations-rank32-5step',          FINAL_STEP),
    'ben_1e5':  ('rank-32-benign/activations-rank32-benign',         FINAL_STEP),
    'base_1e5': ('rank-32-dense/activations-rank32-dense',           0),
}

acts = {}
with ThreadPoolExecutor(max_workers=6) as pool:
    futs = {pool.submit(_load, prefix, step): name
            for name, (prefix, step) in load_tasks.items()}
    for fut in as_completed(futs):
        name = futs[fut]
        acts[name] = fut.result()
        print(f'  loaded {name}  shape={acts[name].shape}')

# ── Compute directions per layer ───────────────────────────────────────────────
def _dir(a, b, layer):
    return a[layer_idx[layer]] - b[layer_idx[layer]]   # (hidden,)

def _cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

directions = {}   # (run, method, layer) -> vector

for run, mis_key, ben_key, base_key in [
    ('1e-6', 'mis_1e6', 'ben_1e6', 'base_1e6'),
    ('1e-5', 'mis_1e5', 'ben_1e5', 'base_1e5'),
]:
    for layer in TARGET_LAYERS:
        directions[(run, 'benign_sub',     layer)] = _dir(acts[mis_key],  acts[ben_key],  layer)
        directions[(run, 'drift_to_final', layer)] = _dir(acts[mis_key],  acts[base_key], layer)

# ── Within-run pairwise similarity ────────────────────────────────────────────
SEP = '─' * 72
for run in ['1e-6', '1e-5']:
    print(f'\n{SEP}')
    print(f'  RUN {run}  —  pairwise cosine similarity between directions')
    print(SEP)
    print(f'{"Layer":>6}  {"benign_sub vs drift_to_final":>28}  '
          f'{"||benign_sub||":>14}  {"||drift_to_final||":>18}')
    print(SEP)
    for layer in TARGET_LAYERS:
        bs  = directions[(run, 'benign_sub',     layer)]
        dtf = directions[(run, 'drift_to_final', layer)]
        cos = _cos(bs, dtf)
        print(f'  L{layer:<4}  {cos:>+28.4f}  {np.linalg.norm(bs):>14.3f}  {np.linalg.norm(dtf):>18.3f}')

# ── Cross-run similarity: does each direction generalise? ─────────────────────
print(f'\n{SEP}')
print(f'  CROSS-RUN: cos_sim between 1e-5 and 1e-6 directions at same layer')
print(SEP)
print(f'{"Layer":>6}  {"benign_sub(1e5) vs benign_sub(1e6)":>36}  '
      f'{"drift(1e5) vs drift(1e6)":>26}')
print(SEP)
for layer in TARGET_LAYERS:
    bs_cross  = _cos(directions[('1e-5', 'benign_sub',     layer)],
                     directions[('1e-6', 'benign_sub',     layer)])
    dtf_cross = _cos(directions[('1e-5', 'drift_to_final', layer)],
                     directions[('1e-6', 'drift_to_final', layer)])
    print(f'  L{layer:<4}  {bs_cross:>+36.4f}  {dtf_cross:>+26.4f}')

print(f'\n{SEP}')
print('Key: high within-run cos_sim → methods agree on the direction.')
print('     high cross-run cos_sim  → direction generalises across LR runs.')

# %%
# ── Projection analysis ────────────────────────────────────────────────────────
# Requires: acts, directions, layer_idx, TARGET_LAYERS from previous cell

CONTAMINATION_THRESH = 0.40   # flag benign alignment above this

# Compute drifts relative to base (mis_final - base, ben_final - base)
drifts_mis, drifts_ben = {}, {}
for run, mis_key, ben_key, base_key in [
    ('1e-6', 'mis_1e6', 'ben_1e6', 'base_1e6'),
    ('1e-5', 'mis_1e5', 'ben_1e5', 'base_1e5'),
]:
    for layer in TARGET_LAYERS:
        idx = layer_idx[layer]
        drifts_mis[(run, layer)] = acts[mis_key][idx]  - acts[base_key][idx]
        drifts_ben[(run, layer)] = acts[ben_key][idx]  - acts[base_key][idx]

SEP = '─' * 82
print(SEP)
print('  PROJECTION ANALYSIS  —  cos_sim(model_drift, extracted_direction)')
print(f'  * = benign projection > {CONTAMINATION_THRESH} (direction may carry generic fine-tuning signal)')
print(f'  Note: proj_mis for drift_to_final = 1.000 by construction (direction IS the mis drift)')
print(SEP)
print(f'{"Layer":>6}  {"Method":>16}  {"Run":>5}  {"proj_mis":>10}  {"proj_ben":>10}'
      f'  {"delta(mis-ben)":>14}')
print(SEP)

for method in ['benign_sub', 'drift_to_final']:
    print(f'\n  [{method}]')
    for run in ['1e-6', '1e-5']:
        print(f'  run={run}')
        for layer in TARGET_LAYERS:
            d        = directions[(run, method, layer)]
            d_unit   = d / (np.linalg.norm(d) + 1e-12)
            proj_mis = float(np.dot(drifts_mis[(run, layer)], d_unit) /
                             (np.linalg.norm(drifts_mis[(run, layer)]) + 1e-12))
            proj_ben = float(np.dot(drifts_ben[(run, layer)], d_unit) /
                             (np.linalg.norm(drifts_ben[(run, layer)]) + 1e-12))
            delta    = proj_mis - proj_ben
            flag     = '  * CONTAMINATED' if proj_ben > CONTAMINATION_THRESH else ''
            print(f'    L{layer:<4}  {proj_mis:>+10.4f}  {proj_ben:>+10.4f}  {delta:>+14.4f}{flag}')

print(f'\n{SEP}')
print('Ideal result: proj_mis ≈ +1.0, proj_ben ≈ 0 or negative.')
print('If delta is large (>> 0.5): direction is misalignment-specific.')
print('If delta ≈ 0: direction captures generic fine-tuning drift shared by both models.')

# %%

