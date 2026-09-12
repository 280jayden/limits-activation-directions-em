"""local_cosine_similarity_rotation.py

Paper mapping
    Supporting. Turner et al. App F local-cosine-similarity rotation detector across all layers and three learning rates.

Provenance
    Converted from the Colab notebook ``local_cosine_sim_rotation.ipynb`` (Drive id 1tLGP2KZ4hE7V-ZG_Dqsj7Bgoi0T_34xt,
    last modified 2026-06-18; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch/activations-rank32-2epoch
    rank-32-2epoch/mech-analysis
    rank-32-5e6/activations-rank32-5e6
    rank-32-5e6/mech-analysis
    rank-32-dense/activations-rank32-dense
    rank-32-dense/mech-analysis
    rank-32-lowlr/activations-rank32-lowlr
    rank-32-lowlr/mech-analysis
    rank-32-lr1e6/activations
    rank-32-lr1e6/activations-rank32-lr1e6
    rank-32-lr1e6/mech-analysis
    rank-32/activations-rank32
    rank-32/mech-analysis
"""

# %% [markdown]
# # Turner et al. Local Cosine Similarity Rotation Detection — All Three LR Runs
# 
# Implements Turner et al. Appendix F across **all layers** in the saved activation files.
# 
# **Turner's formula:** for each step $t$ and offset $s \in \{5, 10, 15\}$:
# $$a = \text{drift}_{t-s} - \text{drift}_t, \quad b = \text{drift}_{t+s} - \text{drift}_t$$
# $$\text{LCS}(t, s) = \cos(a, b) \quad \text{when} \quad \max(\|a\|_2, \|b\|_2) > 0.0035$$
# 
# Interpretation: $-1$ = straight path, $0$ = orthogonal rotation, $+1$ = reversal.
# A peak rising toward 0 or above indicates a rotation event.
# 
# | Run | LR | Confirmed onset |
# |-----|----|-----------------|
# | rank-32-2epoch | 1e-5 | step 12 |
# | rank-32-lowlr  | 5e-6 | step ~22 |
# | rank-32-lr1e6  | 1e-6 | step 70 |

# %%
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# %%
# (shell) pip install -q boto3 numpy matplotlib

# %%
import io, json, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import boto3
from em_directions.colab_compat import userdata

os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

S3_BUCKET = 'jayden-algoverse-sp26'
s3        = boto3.client('s3')

# ── Turner constants ──────────────────────────────────────────────────────────
OFFSETS   = [5, 10, 15]    # step offsets s
THRESHOLD = 0.0035         # min max(||a||, ||b||) to compute LCS

# ── Run configs ───────────────────────────────────────────────────────────────
RUN_CONFIGS = [
    {
        'label':           'rank-32-2epoch (1e-5)',
        'short':           '1e5',
        'onset':           12,
        'color':           'steelblue',
        'act_prefixes':    [
            'rank-32-2epoch/activations-rank32-2epoch',
            'rank-32-dense/activations-rank32-dense',
            'rank-32/activations-rank32',
        ],
        'mech_prefix':     'rank-32-2epoch/mech-analysis',
    },
    {
        'label':           'rank-32-lowlr (5e-6)',
        'short':           '5e6',
        'onset':           22,
        'color':           'darkorange',
        'act_prefixes':    [
            'rank-32-lowlr/activations-rank32-lowlr',
            'rank-32-5e6/activations-rank32-5e6',
        ],
        'mech_prefix':     'rank-32-lowlr/mech-analysis',
    },
    {
        'label':           'rank-32-lr1e6 (1e-6)',
        'short':           '1e6',
        'onset':           70,
        'color':           'firebrick',
        'act_prefixes':    [
            'rank-32-lr1e6/activations-rank32-lr1e6',
        ],
        'mech_prefix':     'rank-32-lr1e6/mech-analysis',
    },
]

# Base model prefix candidates (step_0.npy)
BASE_PREFIXES = [
    'rank-32-2epoch/activations-rank32-2epoch',
    'rank-32-dense/activations-rank32-dense',
    'rank-32/activations-rank32',
    'rank-32-lr1e6/activations-rank32-lr1e6',
    'rank-32-lowlr/activations-rank32-lowlr',
]

print(f'Config ready. OFFSETS={OFFSETS}, THRESHOLD={THRESHOLD}')

# %%
# ── Discover activation prefixes and list available step_N.npy files ──────────

def list_step_npys(prefix):
    """Return sorted list of integer step numbers found as step_N.npy under prefix."""
    steps = []
    pager = s3.get_paginator('list_objects_v2')
    for page in pager.paginate(Bucket=S3_BUCKET, Prefix=prefix + '/'):
        for obj in page.get('Contents', []):
            m = re.search(r'step_(\d+)\.npy$', obj['Key'])
            if m:
                steps.append(int(m.group(1)))
    return sorted(steps)

SEP = '=' * 65
print(SEP)
print('S3 DISCOVERY')
print(SEP)

# Find base model step_0
base_prefix_found = None
for bp in BASE_PREFIXES:
    key = f'{bp}/step_0.npy'
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        base_prefix_found = bp
        print(f'Base model (step_0): s3://{S3_BUCKET}/{key}')
        break
    except Exception:
        pass
if base_prefix_found is None:
    print('WARNING: step_0.npy not found in any candidate prefix.')
    print('Will use earliest available step as pseudo-base (less accurate).')

print()
for cfg in RUN_CONFIGS:
    print(f"{cfg['label']}")
    found = False
    for prefix in cfg['act_prefixes']:
        steps = list_step_npys(prefix)
        if steps:
            non_zero = [s for s in steps if s > 0]
            print(f'  Prefix : s3://{S3_BUCKET}/{prefix}/')
            print(f'  Steps  : {steps[0]}–{steps[-1]}  ({len(steps)} total, {len(non_zero)} non-zero)')
            cfg['resolved_prefix'] = prefix
            cfg['all_steps']       = steps
            found = True
            break
    if not found:
        tried = ', '.join(cfg['act_prefixes'])
        print(f'  *** NOT FOUND in: {tried}')
        cfg['resolved_prefix'] = None
        cfg['all_steps']       = []
    print()

print(SEP)
for cfg in RUN_CONFIGS:
    status = f"{len(cfg.get('all_steps', []))} steps" if cfg.get('resolved_prefix') else 'MISSING'
    print(f"  {cfg['label']:35s}  ->  {status}")
print(SEP)

# %%
# ── Load all activations and compute prompt-averaged drift vectors ─────────────
# Each .npy: shape (n_prompts, n_layers+1, hidden_dim)
# Index 0 = embedding; index k+1 = output of transformer layer k.
# Average across prompts → (n_layers+1, hidden_dim).
# Drift[step][layer] = avg_acts[step][layer+1] - h_base[layer+1]
# (layer+1 to skip the embedding index)

def load_npy(prefix, step):
    key = f'{prefix}/step_{step}.npy'
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    return np.load(buf).astype(np.float32)   # (n_prompts, n_layers+1, hidden_dim)

# Load base model — shared across all runs
if base_prefix_found:
    raw_base = load_npy(base_prefix_found, 0)
    h_base   = raw_base.mean(axis=0)         # (n_layers+1, hidden_dim)
    print(f'Base model loaded from step_0: shape {h_base.shape}')
    N_LAYERS = h_base.shape[0] - 1           # exclude embedding → 28 for Qwen2.5-7B
    LAYER_INDICES = list(range(N_LAYERS))     # 0..27
    print(f'N_LAYERS = {N_LAYERS}  (indices 0–{N_LAYERS-1})')
else:
    h_base   = None
    N_LAYERS = None

# Load per-run activations and build drift[step] = (N_LAYERS, hidden_dim)
for cfg in RUN_CONFIGS:
    if not cfg.get('resolved_prefix'):
        cfg['drifts'] = {}
        continue

    prefix   = cfg['resolved_prefix']
    steps    = [s for s in cfg['all_steps'] if s > 0]
    drifts   = {}

    # If no shared base, use earliest step in this run
    run_base = h_base
    if run_base is None:
        raw_first = load_npy(prefix, steps[0])
        run_base  = raw_first.mean(axis=0)
        steps     = steps[1:]
        print(f"{cfg['label']}: no step_0, using step {cfg['all_steps'][0]} as pseudo-base")

    if N_LAYERS is None:
        N_LAYERS     = run_base.shape[0] - 1
        LAYER_INDICES = list(range(N_LAYERS))

    print(f"Loading {cfg['label']} ({len(steps)} checkpoints)...")
    for step in steps:
        raw   = load_npy(prefix, step)
        avg   = raw.mean(axis=0)               # (n_layers+1, hidden_dim)
        drifts[step] = avg[1:] - run_base[1:]  # (N_LAYERS, hidden_dim) — skip embedding

    cfg['drifts']       = drifts
    cfg['sorted_steps'] = sorted(drifts.keys())
    print(f'  {len(drifts)} drift vectors, steps {cfg["sorted_steps"][0]}–{cfg["sorted_steps"][-1]}')

print(f'\nAll runs loaded. N_LAYERS={N_LAYERS}')

# %%
# ── Vectorized LCS computation (Turner Appendix F) ────────────────────────────
# For each run × offset × step: lcs[layer] = cos(a, b) where
#   a = drift[t-s] - drift[t]   (shape: N_LAYERS, hidden_dim)
#   b = drift[t+s] - drift[t]
# Valid only when max(||a||, ||b||) > THRESHOLD (element-wise per layer).

def compute_lcs_for_run(drifts, sorted_steps):
    """Returns lcs_data[offset][step] = np.array(N_LAYERS,) of LCS values (NaN where invalid)."""
    step_set  = set(sorted_steps)
    lcs_data  = {s: {} for s in OFFSETS}

    for offset in OFFSETS:
        for t in sorted_steps:
            tm = t - offset
            tp = t + offset
            if tm not in step_set or tp not in step_set:
                continue

            d_t  = drifts[t]   # (N_LAYERS, hidden_dim)
            d_tm = drifts[tm]
            d_tp = drifts[tp]

            a = d_tm - d_t   # (N_LAYERS, hidden_dim)
            b = d_tp - d_t

            norm_a = np.linalg.norm(a, axis=1)   # (N_LAYERS,)
            norm_b = np.linalg.norm(b, axis=1)
            valid  = np.maximum(norm_a, norm_b) > THRESHOLD

            # Cosine similarity vectorized over layers
            dot = (a * b).sum(axis=1)            # (N_LAYERS,)
            denom = norm_a * norm_b
            # Avoid div-by-zero for invalid entries
            safe_denom = np.where(valid, denom, 1.0)
            cos = np.where(valid, dot / safe_denom, np.nan)
            lcs_data[offset][t] = cos

    return lcs_data


for cfg in RUN_CONFIGS:
    if not cfg.get('drifts'):
        cfg['lcs_data'] = {}
        continue
    print(f"Computing LCS for {cfg['label']}...")
    cfg['lcs_data'] = compute_lcs_for_run(cfg['drifts'], cfg['sorted_steps'])
    for offset in OFFSETS:
        n_valid = len(cfg['lcs_data'][offset])
        print(f'  offset={offset}: {n_valid} valid steps')

print('LCS computation complete.')

# %%
# ── Heatmap: x=step, y=layer, color=LCS at s=10 ──────────────────────────────
# Color: -1 = straight (dark blue), 0 = rotation (white), +1 = reversal (dark red)

HEATMAP_OFFSET = 10
CMAP           = plt.cm.RdBu_r   # -1=blue, 0=white, +1=red

for cfg in RUN_CONFIGS:
    lcs_data = cfg.get('lcs_data', {})
    if not lcs_data or HEATMAP_OFFSET not in lcs_data or not lcs_data[HEATMAP_OFFSET]:
        print(f"Skipping heatmap for {cfg['label']} (no LCS data at s={HEATMAP_OFFSET})")
        continue

    step_map = lcs_data[HEATMAP_OFFSET]
    valid_steps = sorted(step_map.keys())
    onset       = cfg['onset']

    # Build 2D array: shape (N_LAYERS, len(valid_steps))
    mat = np.full((N_LAYERS, len(valid_steps)), np.nan)
    for j, step in enumerate(valid_steps):
        mat[:, j] = step_map[step]

    fig, ax = plt.subplots(figsize=(16, 8))

    # pcolormesh handles non-uniform x-axis correctly
    # Build x edges for non-uniform steps
    xs = np.array(valid_steps, dtype=float)
    # Cell edges: midpoints between steps, plus outer edges
    gaps  = np.diff(xs) / 2
    x_edges = np.concatenate([[xs[0] - gaps[0]], xs[:-1] + gaps, [xs[-1] + gaps[-1]]])
    y_edges = np.arange(N_LAYERS + 1) - 0.5

    # Mask NaN for display
    masked = np.ma.masked_invalid(mat)
    pcm = ax.pcolormesh(x_edges, y_edges, masked, cmap=CMAP,
                        vmin=-1, vmax=1, shading='flat')
    plt.colorbar(pcm, ax=ax, label='Local Cosine Similarity (s=10)', fraction=0.02)

    ax.axvline(x=onset, color='lime', linestyle='--', linewidth=2,
               label=f'Behavioral onset (step {onset})')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title(
        f'Turner LCS Rotation Heatmap — {cfg["label"]}\n'
        f'Blue=-1 (straight path), White=0 (rotation), Red=+1 (reversal)  |  s={HEATMAP_OFFSET}',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='upper left')
    ax.set_yticks(range(0, N_LAYERS, 4))
    ax.set_yticklabels([f'L{i}' for i in range(0, N_LAYERS, 4)])
    plt.tight_layout()

    local_png = f'/tmp/lcs_heatmap_{cfg["short"]}.png'
    fig.savefig(local_png, dpi=150, bbox_inches='tight')
    cfg['heatmap_local'] = local_png
    plt.show()
    print(f"Heatmap saved locally: {local_png}")

# %%
# ── Per-layer line plots: top 5 layers by peak deviation from -1 ──────────────
# Deviation from straight = LCS - (-1) = LCS + 1   (higher = more rotation)
# Rank all layers by max(LCS + 1) across all valid steps and offsets.

TOP_N = 5
OFFSET_STYLES = {5: ':', 10: '-', 15: '--'}
OFFSET_COLORS = {5: 'steelblue', 10: 'firebrick', 15: 'darkorange'}

for cfg in RUN_CONFIGS:
    lcs_data = cfg.get('lcs_data', {})
    if not lcs_data:
        continue
    onset = cfg['onset']

    # Compute per-layer peak deviation from -1 (using all offsets and all steps)
    layer_peak = np.full(N_LAYERS, -np.inf)
    for offset in OFFSETS:
        for step, lcs_vec in lcs_data[offset].items():
            # lcs_vec: (N_LAYERS,) array, may contain NaN
            dev = lcs_vec + 1.0   # deviation from -1
            layer_peak = np.fmax(layer_peak, np.nan_to_num(dev, nan=-np.inf))

    top5_layers = list(np.argsort(layer_peak)[-TOP_N:][::-1])
    print(f"{cfg['label']} — top {TOP_N} layers by peak deviation from -1:")
    for rank, li in enumerate(top5_layers):
        print(f'  #{rank+1}: Layer {li:2d}  peak_dev={layer_peak[li]:.4f}  (LCS_peak={layer_peak[li]-1:.4f})')

    cfg['top5_layers'] = top5_layers
    cfg['layer_peak']  = layer_peak

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(TOP_N, 1, figsize=(14, 3 * TOP_N), sharex=True)
    if TOP_N == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, top5_layers):
        for offset in OFFSETS:
            step_map = lcs_data[offset]
            plot_steps = sorted(step_map.keys())
            plot_vals  = [step_map[t][layer_idx] for t in plot_steps]
            ax.plot(plot_steps, plot_vals,
                    linestyle=OFFSET_STYLES[offset],
                    color=OFFSET_COLORS[offset],
                    linewidth=1.8,
                    label=f's={offset}')

        ax.axhline(y=-1,  color='gray',  linestyle='-',  linewidth=0.7, alpha=0.5)
        ax.axhline(y=0,   color='gray',  linestyle='--', linewidth=0.7, alpha=0.5, label='LCS=0 (orthogonal)')
        ax.axvline(x=onset, color='red', linestyle='--', linewidth=1.5,
                   label=f'Onset step {onset}')
        ax.axvspan(plot_steps[0] if plot_steps else 0, onset,
                   alpha=0.07, color='red')
        ax.set_ylim(-1.15, 0.3)
        ax.set_ylabel('LCS', fontsize=9)
        ax.set_title(
            f'Layer {layer_idx}  (peak dev from -1: {layer_peak[layer_idx]:.4f})',
            fontsize=10
        )
        if ax is axes[0]:
            ax.legend(fontsize=8, loc='upper right', ncol=5)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Training Step', fontsize=11)
    fig.suptitle(
        f'Turner LCS — {cfg["label"]} — Top {TOP_N} Layers by Peak Deviation from -1',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    local_png = f'/tmp/lcs_lineplot_{cfg["short"]}.png'
    fig.savefig(local_png, dpi=150, bbox_inches='tight')
    cfg['lineplot_local'] = local_png
    plt.show()
    print(f"Line plots saved locally: {local_png}")
    print()

# %%
# ── Key question: any layer shows LCS peak meaningfully above -1 pre-onset? ───
NOTABLE_THRESHOLD   = -0.7   # LCS > -0.7 = deviation > 0.3 from straight path
STRONG_THRESHOLD    = -0.5   # LCS > -0.5 = deviation > 0.5

SEP = '=' * 75
print(SEP)
print('KEY QUESTION: ANY rotation event (LCS > -0.7) BEFORE behavioral onset?')
print(SEP)

all_pre_onset_peaks = {}   # for JSON export

for cfg in RUN_CONFIGS:
    lcs_data = cfg.get('lcs_data', {})
    onset    = cfg['onset']
    label    = cfg['label']
    if not lcs_data:
        print(f'\n{label}: NO DATA')
        continue

    print(f'\n{label}  (onset=step {onset})')

    # Collect all pre-onset (layer, step, offset, lcs) tuples above NOTABLE_THRESHOLD
    peaks = []
    for offset in OFFSETS:
        for step, lcs_vec in lcs_data[offset].items():
            if step >= onset:
                continue
            for layer_idx in range(N_LAYERS):
                v = lcs_vec[layer_idx]
                if np.isnan(v):
                    continue
                if v > NOTABLE_THRESHOLD:
                    peaks.append((v, layer_idx, step, offset))

    peaks.sort(reverse=True)

    if not peaks:
        print(f'  => NO rotation events above {NOTABLE_THRESHOLD} before onset.')
        print(f'     This is the "genuinely no rotation events" result.')
        # Report the absolute best pre-onset value across all layers
        best_pre = -2.0
        best_info = None
        for offset in OFFSETS:
            for step, lcs_vec in lcs_data[offset].items():
                if step >= onset:
                    continue
                valid = lcs_vec[~np.isnan(lcs_vec)]
                if len(valid) > 0 and valid.max() > best_pre:
                    best_pre  = valid.max()
                    best_lyr  = int(np.nanargmax(lcs_vec))
                    best_info = (best_lyr, step, offset)
        if best_info:
            print(f'     Best pre-onset LCS anywhere: {best_pre:.4f}  '
                  f'(layer {best_info[0]}, step {best_info[1]}, s={best_info[2]})')
    else:
        print(f'  Found {len(peaks)} pre-onset events above {NOTABLE_THRESHOLD}:')
        print(f'  {"LCS":>8}  {"Layer":>6}  {"Step":>6}  {"s":>4}  {"Strong?"}')
        print('  ' + '-' * 40)
        for v, li, st, off in peaks[:20]:   # show top 20
            strong = '*** STRONG ***' if v > STRONG_THRESHOLD else ''
            print(f'  {v:>8.4f}  {li:>6}  {st:>6}  {off:>4}  {strong}')
        if len(peaks) > 20:
            print(f'  ... and {len(peaks)-20} more')

    all_pre_onset_peaks[label] = [
        {'lcs': round(v, 6), 'layer': li, 'step': st, 'offset': off}
        for v, li, st, off in peaks
    ]

print()
print(SEP)
print('GLOBAL SUMMARY (all runs, any step, peak LCS per layer):')
print(SEP)
for cfg in RUN_CONFIGS:
    if 'layer_peak' not in cfg:
        continue
    top3 = np.argsort(cfg['layer_peak'])[-3:][::-1]
    print(f"{cfg['label']}:")
    for li in top3:
        print(f'  Layer {li:2d}  peak LCS = {cfg["layer_peak"][li]-1:.4f}  (dev from -1)')
print(SEP)

# %%
# ── Upload figures and save JSON to each run's mech-analysis S3 prefix ────────

for cfg in RUN_CONFIGS:
    if not cfg.get('lcs_data'):
        continue

    mech  = cfg['mech_prefix']
    label = cfg['label']

    # Upload heatmap
    if cfg.get('heatmap_local'):
        key = f'{mech}/local_cosine_sim_heatmap_multilayer.png'
        s3.upload_file(cfg['heatmap_local'], S3_BUCKET, key)
        print(f'{label}: heatmap -> s3://{S3_BUCKET}/{key}')

    # Upload line plot
    if cfg.get('lineplot_local'):
        key = f'{mech}/local_cosine_sim_lineplot_multilayer.png'
        s3.upload_file(cfg['lineplot_local'], S3_BUCKET, key)
        print(f'{label}: lineplot -> s3://{S3_BUCKET}/{key}')

    # Build and upload JSON
    lcs_data = cfg['lcs_data']

    # Compact serialization: lcs_json[offset][step] = list of N_LAYERS floats (None=NaN)
    lcs_json = {}
    for offset in OFFSETS:
        lcs_json[str(offset)] = {
            str(step): [
                round(float(v), 5) if not np.isnan(v) else None
                for v in vec
            ]
            for step, vec in sorted(lcs_data[offset].items())
        }

    # Pre-onset peaks summary
    pre_peaks = all_pre_onset_peaks.get(label, [])

    # Layer peak deviation table
    layer_peak_table = {}
    if 'layer_peak' in cfg:
        layer_peak_table = {str(i): round(float(cfg['layer_peak'][i] - 1.0), 6)
                            for i in range(N_LAYERS)}

    out = {
        'run':                  label,
        'onset':                cfg['onset'],
        'n_layers':             N_LAYERS,
        'offsets':              OFFSETS,
        'threshold':            THRESHOLD,
        'top5_layers':          cfg.get('top5_layers', []),
        'layer_peak_dev_from_neg1': layer_peak_table,
        'pre_onset_notable_peaks':  pre_peaks[:50],  # top 50
        'lcs':                  lcs_json,
    }

    json_key = f'{mech}/local_cosine_sim_rotation_multilayer.json'
    s3.put_object(
        Bucket=S3_BUCKET, Key=json_key,
        Body=json.dumps(out, indent=2).encode()
    )
    print(f'{label}: JSON  -> s3://{S3_BUCKET}/{json_key}')
    print()

# %%
# ── Discover and load drift-direction + LCS JSONs for the overlay ─────────────

# Candidate mech-analysis prefixes where drift_direction JSON may live
DRIFT_DIR_CANDIDATES = {
    'rank-32-2epoch (1e-5)': [
        'rank-32-2epoch/mech-analysis',
        'rank-32/mech-analysis',
        'rank-32-dense/mech-analysis',
    ],
    'rank-32-lowlr (5e-6)': [
        'rank-32-lowlr/mech-analysis',
        'rank-32-5e6/mech-analysis',
    ],
    'rank-32-lr1e6 (1e-6)': [
        'rank-32-lr1e6/mech-analysis',
    ],
}

LCS_OVERLAY_OFFSET = 10    # s=10
LAYER_27           = 27
LAYERS_1113        = [11, 12, 13]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_s3_json(key):
    try:
        buf = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, key, buf)
        buf.seek(0)
        return json.load(buf)
    except Exception:
        return None

def _discover_drift_json(mech_prefixes):
    """
    Find a JSON with structure {results: {layer_str: {step_str: {cos_sim: float}}}}
    in any of the given prefixes. Prioritises files starting with 'drift_direction'.
    Returns (s3_key, data) or (None, None).
    """
    for prefix in mech_prefixes:
        priority, rest = [], []
        pager = s3.get_paginator('list_objects_v2')
        for page in pager.paginate(Bucket=S3_BUCKET, Prefix=prefix + '/'):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if not key.endswith('.json'):
                    continue
                fname = key.split('/')[-1]
                (priority if fname.startswith('drift_direction') else rest).append(key)
        for key in priority + rest:
            data = _load_s3_json(key)
            if not data or 'results' not in data:
                continue
            layers = list(data['results'].keys())
            if not layers:
                continue
            first_step = next(iter(data['results'][layers[0]].values()), {})
            if 'cos_sim' not in first_step:
                continue
            print(f'  drift_direction JSON: {key.split("/")[-1]}'
                  f'  (layers {sorted(int(l) for l in layers)})')
            return key, data
    return None, None

def _get_cosim(drift_data, layer_idx):
    """Returns {step_int: cos_sim_float} for one layer. Empty dict if absent."""
    if drift_data is None:
        return {}
    return {
        int(s): v['cos_sim']
        for s, v in drift_data.get('results', {}).get(str(layer_idx), {}).items()
    }

def _get_lcs(lcs_json, layer_idx, offset=LCS_OVERLAY_OFFSET):
    """Returns {step_int: lcs_float} for one layer. Empty dict if absent."""
    if lcs_json is None:
        return {}
    result = {}
    for step_str, arr in lcs_json.get('lcs', {}).get(str(offset), {}).items():
        if arr is not None and layer_idx < len(arr) and arr[layer_idx] is not None:
            result[int(step_str)] = arr[layer_idx]
    return result

def _avg_series(dicts):
    """
    Average a list of {step: val} dicts.
    Returns (avg, lo, hi) as three dicts with the same step keys.
    Only includes steps where at least one dict has a value.
    """
    all_steps = sorted(set().union(*[set(d.keys()) for d in dicts if d]))
    avg, lo, hi = {}, {}, {}
    for step in all_steps:
        vals = [d[step] for d in dicts if step in d]
        if vals:
            avg[step] = float(np.mean(vals))
            lo[step]  = float(np.min(vals))
            hi[step]  = float(np.max(vals))
    return avg, lo, hi

def _xy(d):
    """Sort {step: val} → (x_array, y_array)."""
    if not d:
        return np.array([]), np.array([])
    xs, ys = zip(*sorted(d.items()))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)

def _val_near(xs, ys, target):
    """Return ys value at the step closest to target. None if empty."""
    if not len(xs):
        return None
    return float(ys[int(np.argmin(np.abs(xs - target)))])

# ── Load per-run ──────────────────────────────────────────────────────────────
print('Loading overlay data')
print('=' * 65)
for cfg in RUN_CONFIGS:
    label = cfg['label']
    print(f'\n{label}')
    candidates = DRIFT_DIR_CANDIDATES.get(label, [cfg['mech_prefix']])

    _, dd = _discover_drift_json(candidates)
    cfg['dd'] = dd
    if dd is None:
        print('  *** drift_direction JSON not found')

    lcs_key = f'{cfg["mech_prefix"]}/local_cosine_sim_rotation_multilayer.json'
    ld = _load_s3_json(lcs_key)
    cfg['ld'] = ld
    if ld is None:
        print(f'  *** LCS JSON not found at {lcs_key}')
    else:
        n = len(ld.get('lcs', {}).get(str(LCS_OVERLAY_OFFSET), {}))
        print(f'  LCS JSON: {n} steps at s={LCS_OVERLAY_OFFSET}')

print('\n' + '=' * 65)
print('Ready.')

# %%
# ── Dual-axis overlay: cos_sim_to_final (left) + LCS s=10 (right) ─────────────

from matplotlib.lines import Line2D

COSIM_COL  = 'royalblue'
LCS_COL    = 'crimson'
REF_COL    = '#bbbbbb'

def _draw_panel(ax, dd, ld, layer_or_layers, onset, title=''):
    """
    Renders one panel. left axis = cos_sim_to_final; right axis = LCS.
    layer_or_layers: int or list of ints. Lists → avg ± range band.
    """
    layers = [layer_or_layers] if isinstance(layer_or_layers, int) else list(layer_or_layers)
    multi  = len(layers) > 1

    cs_per_layer  = [_get_cosim(dd, l) for l in layers]
    lcs_per_layer = [_get_lcs(ld, l)   for l in layers]

    if multi:
        cs_avg, cs_lo, cs_hi   = _avg_series([s for s in cs_per_layer  if s])
        lc_avg, lc_lo, lc_hi   = _avg_series([s for s in lcs_per_layer if s])
    else:
        cs_avg = cs_per_layer[0];  cs_lo = cs_avg;  cs_hi = cs_avg
        lc_avg = lcs_per_layer[0]; lc_lo = lc_avg;  lc_hi = lc_avg

    xs_cs, ys_cs = _xy(cs_avg)
    xs_lc, ys_lc = _xy(lc_avg)
    _,  ys_cs_lo = _xy(cs_lo)
    _,  ys_cs_hi = _xy(cs_hi)
    _,  ys_lc_lo = _xy(lc_lo)
    _,  ys_lc_hi = _xy(lc_hi)

    ax2 = ax.twinx()

    # ── cos_sim ───────────────────────────────────────────────────────────────
    if len(xs_cs):
        ax.plot(xs_cs, ys_cs, color=COSIM_COL, lw=2.2, label='cos_sim to final')
        if multi:
            ax.fill_between(xs_cs, ys_cs_lo, ys_cs_hi, alpha=0.12, color=COSIM_COL)

    # ── LCS ───────────────────────────────────────────────────────────────────
    if len(xs_lc):
        ax2.plot(xs_lc, ys_lc, color=LCS_COL, lw=2.2, ls='--',
                 label=f'LCS (s={LCS_OVERLAY_OFFSET})')
        if multi:
            ax2.fill_between(xs_lc, ys_lc_lo, ys_lc_hi, alpha=0.12, color=LCS_COL)

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axhline( 0.9,  color=REF_COL, ls=':', lw=1.1, zorder=0)
    ax2.axhline( 0.0, color=REF_COL, ls=':', lw=1.1, zorder=0)
    ax2.axhline(-1.0, color=REF_COL, ls='-', lw=0.5, alpha=0.45, zorder=0)

    # ref labels
    ax.text(0.01,  0.90 + 0.02, 'cos=0.9', transform=ax.get_yaxis_transform(),
            fontsize=6.5, color=REF_COL, va='bottom')
    ax2.text(1.01,  0.00 + 0.02, 'LCS=0',  transform=ax2.get_yaxis_transform(),
             fontsize=6.5, color=REF_COL, va='bottom', ha='left')

    # ── Onset ─────────────────────────────────────────────────────────────────
    ax.axvline(onset, color='black', ls='--', lw=2.0, zorder=4)
    ax.axvspan(0, onset, alpha=0.04, color='salmon', zorder=0)

    # Annotate cos_sim value at onset
    v = _val_near(xs_cs, ys_cs, onset)
    if v is not None:
        ax.text(onset, v + 0.055, f'  {v:.2f}',
                fontsize=8, color=COSIM_COL, fontweight='bold', va='bottom')

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax.set_ylim(-0.05, 1.08)
    ax2.set_ylim(-1.15, 0.45)
    ax.set_ylabel('cos_sim to final', color=COSIM_COL, fontsize=8.5)
    ax2.set_ylabel('LCS', color=LCS_COL, fontsize=8.5)
    ax.tick_params(axis='y', colors=COSIM_COL, labelsize=8)
    ax2.tick_params(axis='y', colors=LCS_COL, labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(True, alpha=0.2, zorder=0)

    # ── Combined legend ───────────────────────────────────────────────────────
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    onset_h = Line2D([0], [0], color='black', ls='--', lw=2.0)
    ax.legend(h1 + h2 + [onset_h],
              l1 + l2 + [f'Onset step {onset}'],
              fontsize=7.5, loc='upper left', framealpha=0.85)

    if title:
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)
    return ax2

# ── 3 × 2 figure ──────────────────────────────────────────────────────────────
n_runs = len(RUN_CONFIGS)
fig, axes = plt.subplots(2, n_runs, figsize=(6 * n_runs, 10))
if n_runs == 1:
    axes = axes.reshape(2, 1)

for col, cfg in enumerate(RUN_CONFIGS):
    dd, ld  = cfg.get('dd'), cfg.get('ld')
    onset   = cfg['onset']
    label   = cfg['label']

    # Row 0 — Layer 27
    _draw_panel(axes[0][col], dd, ld, LAYER_27, onset,
                title=f'{label}\nLayer {LAYER_27}')
    axes[0][col].set_xlabel('Training step', fontsize=8.5)

    # Row 1 — Layers 11–13 averaged
    _draw_panel(axes[1][col], dd, ld, LAYERS_1113, onset,
                title=f'Layers {LAYERS_1113[0]}–{LAYERS_1113[-1]} (avg ± range)')
    axes[1][col].set_xlabel('Training step', fontsize=8.5)

fig.suptitle(
    'Direction Stabilization vs. Active Rotation\n'
    f'Blue solid: cos_sim to final (left)  ·  '
    f'Red dashed: LCS s={LCS_OVERLAY_OFFSET} (right)  ·  '
    'Black dashed: behavioral onset  ·  '
    'Shaded: pre-onset window',
    fontsize=10.5, fontweight='bold', y=1.02
)
plt.tight_layout()

OVERLAY_LOCAL = '/tmp/cosim_vs_lcs_overlay.png'
fig.savefig(OVERLAY_LOCAL, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {OVERLAY_LOCAL}')

# %%
# ── Upload overlay figure + per-run summary JSON to S3 ───────────────────────

for cfg in RUN_CONFIGS:
    mech  = cfg['mech_prefix']
    label = cfg['label']
    onset = cfg['onset']
    dd, ld = cfg.get('dd'), cfg.get('ld')

    # Same combined figure goes to every run's prefix
    png_key = f'{mech}/cosim_vs_lcs_overlay.png'
    s3.upload_file(OVERLAY_LOCAL, S3_BUCKET, png_key)
    print(f'{label}: PNG  -> s3://{S3_BUCKET}/{png_key}')

    # ── Per-run summary JSON ──────────────────────────────────────────────────
    def _series_json(xy_dict):
        xs, ys = _xy(xy_dict)
        return {str(int(x)): round(float(y), 5) for x, y in zip(xs, ys)}

    # Layer 27
    cs27 = _get_cosim(dd, LAYER_27);  xs_c27, ys_c27 = _xy(cs27)
    lc27 = _get_lcs(ld, LAYER_27);   xs_l27, ys_l27 = _xy(lc27)

    # Layers 11-13 average
    cs_avg_1113, _, _ = _avg_series([_get_cosim(dd, l) for l in LAYERS_1113
                                     if _get_cosim(dd, l)])
    lc_avg_1113, _, _ = _avg_series([_get_lcs(ld, l)   for l in LAYERS_1113
                                     if _get_lcs(ld, l)])
    xs_ca, ys_ca = _xy(cs_avg_1113)
    xs_la, ys_la = _xy(lc_avg_1113)

    summary = {
        'run':    label,
        'onset':  onset,
        'layer_27': {
            'cosim_at_onset': _val_near(xs_c27, ys_c27, onset),
            'lcs_at_onset':   _val_near(xs_l27, ys_l27, onset),
            'cosim_series':   _series_json(cs27),
            'lcs_series':     _series_json(lc27),
        },
        'layers_11_13_avg': {
            'cosim_at_onset': _val_near(xs_ca, ys_ca, onset),
            'lcs_at_onset':   _val_near(xs_la, ys_la, onset),
            'cosim_series':   _series_json(cs_avg_1113),
            'lcs_series':     _series_json(lc_avg_1113),
        },
    }

    json_key = f'{mech}/cosim_vs_lcs_overlay.json'
    s3.put_object(Bucket=S3_BUCKET, Key=json_key,
                  Body=json.dumps(summary, indent=2).encode())
    print(f'{label}: JSON -> s3://{S3_BUCKET}/{json_key}')
    print()

print('All overlay outputs uploaded.')

# %%
# ── DIAGNOSTIC + FIXED OVERLAY ────────────────────────────────────────────────
import matplotlib.gridspec as gridspec

# ── Step 1: print raw LCS values to confirm data is loaded ───────────────────
print('=' * 62)
print(f'DIAGNOSTIC: LCS for layer {LAYER_27}, s={LCS_OVERLAY_OFFSET}')
print('=' * 62)
for cfg in RUN_CONFIGS:
    ld = cfg.get('ld')
    print(f"\n{cfg['label']}")
    if ld is None:
        print('  *** ld is None — JSON not loaded (check path / run LCS notebook first)')
        continue
    offset_data = ld.get('lcs', {}).get(str(LCS_OVERLAY_OFFSET), {})
    if not offset_data:
        print(f'  *** lcs["{LCS_OVERLAY_OFFSET}"] is empty')
        continue
    steps = sorted(int(k) for k in offset_data.keys())
    print(f'  {len(steps)} steps  {steps[:3]} … {steps[-3:]}')
    shown = 0
    for step in steps:
        arr = offset_data.get(str(step))
        if arr and len(arr) > LAYER_27 and arr[LAYER_27] is not None:
            print(f'    step {step:4d}  lcs={arr[LAYER_27]:.5f}')
            shown += 1
        if shown >= 5:
            break
    if shown == 0:
        print(f'  *** every layer-{LAYER_27} value at s={LCS_OVERLAY_OFFSET} is None')
print('\n' + '=' * 62)

# ── Step 2: fixed figure — cos_sim and LCS as stacked rows, shared x-axis ────
# (avoids all twinx scaling failures; both rows share the training-step axis)

COSIM_COL = 'royalblue'
LCS_COL   = 'crimson'
REF_COL   = '#cccccc'

def _cosim_row(ax, dd, layer_or_layers, onset, title=''):
    layers = [layer_or_layers] if isinstance(layer_or_layers, int) else list(layer_or_layers)
    cs_all = [_get_cosim(dd, l) for l in layers]
    valid  = [s for s in cs_all if s]

    if len(layers) > 1 and valid:
        avg, lo, hi = _avg_series(valid)
        xs, ys       = _xy(avg)
        _, ys_lo     = _xy(lo)
        _, ys_hi     = _xy(hi)
    elif valid:
        xs, ys = _xy(valid[0]); ys_lo = ys; ys_hi = ys
    else:
        xs, ys = np.array([]), np.array([])

    if len(xs):
        ax.plot(xs, ys, color=COSIM_COL, lw=2.2, label='cos_sim to final')
        if len(layers) > 1:
            ax.fill_between(xs, ys_lo, ys_hi, alpha=0.13, color=COSIM_COL)

    ax.axhline(0.9, color=REF_COL, ls=':', lw=1.2, label='cos=0.9')
    ax.axvline(onset, color='black', ls='--', lw=2.0, label=f'onset={onset}')
    ax.axvspan(0, onset, alpha=0.04, color='salmon')

    v = _val_near(xs, ys, onset)
    if v is not None:
        ax.text(onset, v + 0.06, f' {v:.2f}',
                fontsize=8.5, color=COSIM_COL, fontweight='bold', va='bottom')

    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel('cos_sim\nto final', color=COSIM_COL, fontsize=8.5)
    ax.tick_params(axis='y', colors=COSIM_COL, labelsize=8)
    ax.tick_params(axis='x', labelbottom=False)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)


def _lcs_row(ax, ld, layer_or_layers, onset, add_xlabel=False):
    layers = [layer_or_layers] if isinstance(layer_or_layers, int) else list(layer_or_layers)
    lc_all = [_get_lcs(ld, l) for l in layers]
    valid  = [s for s in lc_all if s]

    if len(layers) > 1 and valid:
        avg, lo, hi = _avg_series(valid)
        xs, ys       = _xy(avg)
        _, ys_lo     = _xy(lo)
        _, ys_hi     = _xy(hi)
    elif valid:
        xs, ys = _xy(valid[0]); ys_lo = ys; ys_hi = ys
    else:
        xs, ys = np.array([]), np.array([])

    if len(xs):
        ax.plot(xs, ys, color=LCS_COL, lw=2.2,
                label=f'LCS s={LCS_OVERLAY_OFFSET}')
        if len(layers) > 1:
            ax.fill_between(xs, ys_lo, ys_hi, alpha=0.13, color=LCS_COL)
        print(f'    plotted {len(xs)} pts  range [{ys.min():.3f}, {ys.max():.3f}]')
    else:
        print(f'    *** no LCS points for layers={layers}')

    ax.axhline( 0.0, color=REF_COL, ls=':', lw=1.2, label='LCS=0')
    ax.axhline(-1.0, color=REF_COL, ls='-', lw=0.5, alpha=0.5, label='LCS=−1 (straight)')
    ax.axvline(onset, color='black', ls='--', lw=2.0)
    ax.axvspan(0, onset, alpha=0.04, color='salmon')

    ax.set_ylim(-1.12, 0.45)   # explicit — never auto-scale LCS
    ax.set_ylabel(f'LCS\ns={LCS_OVERLAY_OFFSET}', color=LCS_COL, fontsize=8.5)
    ax.tick_params(axis='y', colors=LCS_COL, labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    if add_xlabel:
        ax.set_xlabel('Training step', fontsize=8.5)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)


# Layout: 4 rows × n_runs cols
# rows 0,1 = layer 27 (cosim / lcs); rows 2,3 = layers 11-13 (cosim / lcs)
n_runs = len(RUN_CONFIGS)
fig = plt.figure(figsize=(6 * n_runs, 14))
gs  = gridspec.GridSpec(4, n_runs,
                        height_ratios=[2.8, 1.2, 2.8, 1.2],
                        hspace=0.06, wspace=0.40)

print('\nPlotting:')
for col, cfg in enumerate(RUN_CONFIGS):
    dd, ld  = cfg.get('dd'), cfg.get('ld')
    onset   = cfg['onset']
    label   = cfg['label']
    print(f'\n  {label}')

    ax_cs27 = fig.add_subplot(gs[0, col])
    ax_lc27 = fig.add_subplot(gs[1, col], sharex=ax_cs27)
    _cosim_row(ax_cs27, dd, LAYER_27, onset,
               title=f'{label}\nLayer {LAYER_27}')
    print(f'  Layer {LAYER_27} LCS:')
    _lcs_row(ax_lc27, ld, LAYER_27, onset)

    ax_cs13 = fig.add_subplot(gs[2, col])
    ax_lc13 = fig.add_subplot(gs[3, col], sharex=ax_cs13)
    _cosim_row(ax_cs13, dd, LAYERS_1113, onset,
               title=f'Layers {LAYERS_1113[0]}–{LAYERS_1113[-1]} (avg ± range)')
    print(f'  Layers {LAYERS_1113} LCS:')
    _lcs_row(ax_lc13, ld, LAYERS_1113, onset, add_xlabel=True)

fig.suptitle(
    'Direction Stabilization (blue, top) vs. Active Rotation (red, bottom)\n'
    f'cos_sim to final direction  ·  LCS s={LCS_OVERLAY_OFFSET}  ·  '
    'dashed = behavioral onset  ·  shaded = pre-onset window',
    fontsize=10.5, fontweight='bold', y=1.01
)

OVERLAY_LOCAL = '/tmp/cosim_vs_lcs_overlay.png'
fig.savefig(OVERLAY_LOCAL, dpi=150, bbox_inches='tight')
plt.show()
print(f'\nSaved: {OVERLAY_LOCAL}')
print('Re-run the save-overlay cell to push to S3.')

# %%
# ── DIAGNOSE AND FIX LCS OVERLAY ─────────────────────────────────────────────
import matplotlib.gridspec as gridspec

SEP = '=' * 65
KNOWN_SELECTED_LAYERS = [5, 9, 11, 12, 13, 20, 27]  # from drift_direction notebooks

# ── Part 1: print exact structure of each LCS JSON ───────────────────────────
print(SEP)
print('DIAGNOSTIC: LCS JSON structure per run')
print(SEP)

for cfg in RUN_CONFIGS:
    label = cfg['label']
    ld    = cfg.get('ld') or _load_s3_json(
                f'{cfg["mech_prefix"]}/local_cosine_sim_rotation_multilayer.json')
    cfg['ld'] = ld
    print(f'\n{label}')

    if ld is None:
        print('  *** JSON not found')
        cfg['layer_to_idx'] = {}
        continue

    # Top-level structure
    print(f'  top-level keys : {list(ld.keys())}')
    for k in ['n_layers', 'offsets', 'threshold', 'top5_layers']:
        if k in ld:
            print(f'  {k:16s}: {ld[k]}')

    # layer_peak_dev_from_neg1 keys — these ARE the positional indices that exist
    lp = ld.get('layer_peak_dev_from_neg1', {})
    lp_sorted = sorted((int(k), round(v, 4)) for k, v in lp.items())
    print(f'  peak_dev indices (positional): {[k for k,_ in lp_sorted]}')
    print(f'  top-3 by deviation: {sorted(lp_sorted, key=lambda x:-x[1])[:3]}')

    # lcs sub-structure
    lcs = ld.get('lcs', {})
    print(f'\n  lcs present for offsets: {sorted(int(k) for k in lcs.keys())}')

    lcs10 = lcs.get('10', {})
    steps = sorted(int(k) for k in lcs10.keys())
    print(f'  lcs["10"] — {len(steps)} steps: {steps[:3]} … {steps[-2:]}')

    # Print the FULL first array so we can see exactly what's stored
    for step in steps:
        arr = lcs10.get(str(step))
        if arr is not None:
            none_ct = sum(1 for v in arr if v is None)
            print(f'\n  First array (step {step}):')
            print(f'    length        : {len(arr)}')
            print(f'    None count    : {none_ct}/{len(arr)}')
            print(f'    full contents : {arr}')
            print(f'\n  Index spot-check:')
            for li in [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 20, 27]:
                if li < len(arr):
                    print(f'    [{li:2d}] = {arr[li]}')
                else:
                    print(f'    [{li:2d}] = OUT OF RANGE  ← this is why those layers are empty')
            break

    # ── Build layer_number → positional_index mapping ─────────────────────────
    # Determine array length (ground truth for n actual positions in LCS)
    arr_len = next((len(a) for a in lcs10.values() if a is not None), None)
    n = arr_len  # use array length, not the n_layers field (which may be wrong)

    print(f'\n  Array length (actual positions in LCS): {n}')

    if n is None:
        print('  Cannot determine mapping — no arrays found')
        cfg['layer_to_idx'] = {}

    elif n >= 28:
        # All 28+ transformer layers: positional index == layer number
        cfg['layer_to_idx'] = {i: i for i in range(n)}
        print(f'  Mapping: DIRECT (n={n} ≥ 28, layer number = index)')

    elif n == len(KNOWN_SELECTED_LAYERS):
        # .npy contained [embedding, L5, L9, L11, L12, L13, L20, L27] (8 total)
        # avg[1:] stripped embedding → positions 0..6 = L5..L27
        m = {l: i for i, l in enumerate(KNOWN_SELECTED_LAYERS)}
        cfg['layer_to_idx'] = m
        print(f'  Mapping: embedding stripped, {n} positions = KNOWN_SELECTED_LAYERS')
        print(f'  {m}')

    elif n == len(KNOWN_SELECTED_LAYERS) - 1:
        # .npy contained [L5, L9, L11, L12, L13, L20, L27] (no embedding)
        # avg[1:] stripped L5 → positions 0..5 = L9..L27
        remaining = KNOWN_SELECTED_LAYERS[1:]
        m = {l: i for i, l in enumerate(remaining)}
        cfg['layer_to_idx'] = m
        print(f'  Mapping: L{KNOWN_SELECTED_LAYERS[0]} treated as base, '
              f'{n} positions = remaining layers')
        print(f'  {m}')

    else:
        cfg['layer_to_idx'] = {i: i for i in range(n)}
        print(f'  WARNING: n={n} matches no known pattern. Using direct map.')

    l2i = cfg['layer_to_idx']
    print(f'\n  Resolved: layer 11 → idx {l2i.get(11, "MISSING")}  '
          f'layer 12 → idx {l2i.get(12, "MISSING")}  '
          f'layer 13 → idx {l2i.get(13, "MISSING")}  '
          f'layer 27 → idx {l2i.get(27, "MISSING")}')

print('\n' + SEP)

# ── Part 2: Replot with the resolved mapping ──────────────────────────────────
COSIM_COL = 'royalblue'
LCS_COL   = 'crimson'
REF_COL   = '#cccccc'

def _get_lcs_m(lcs_json, actual_layer, layer_to_idx, offset=None):
    """LCS series for an actual layer number using the positional-index map."""
    if lcs_json is None or actual_layer not in layer_to_idx:
        return {}
    return _get_lcs(lcs_json, layer_to_idx[actual_layer],
                    offset if offset is not None else LCS_OVERLAY_OFFSET)

def _cosim_row(ax, dd, layers, onset, title=''):
    layers = [layers] if isinstance(layers, int) else list(layers)
    cs_all = [_get_cosim(dd, l) for l in layers]
    valid  = [s for s in cs_all if s]
    if len(layers) > 1 and valid:
        avg, lo, hi = _avg_series(valid)
        xs, ys = _xy(avg); _, ys_lo = _xy(lo); _, ys_hi = _xy(hi)
    elif valid:
        xs, ys = _xy(valid[0]); ys_lo = ys_hi = ys
    else:
        xs, ys = np.array([]), np.array([]); ys_lo = ys_hi = ys
    if len(xs):
        ax.plot(xs, ys, color=COSIM_COL, lw=2.2, label='cos_sim to final')
        if len(layers) > 1:
            ax.fill_between(xs, ys_lo, ys_hi, alpha=0.13, color=COSIM_COL)
    ax.axhline(0.9, color=REF_COL, ls=':', lw=1.2, label='cos=0.9')
    ax.axvline(onset, color='black', ls='--', lw=2.0, label=f'onset={onset}')
    ax.axvspan(0, onset, alpha=0.04, color='salmon')
    v = _val_near(xs, ys, onset)
    if v is not None:
        ax.text(onset, v + 0.06, f' {v:.2f}',
                fontsize=8.5, color=COSIM_COL, fontweight='bold', va='bottom')
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel('cos_sim\nto final', color=COSIM_COL, fontsize=8.5)
    ax.tick_params(axis='y', colors=COSIM_COL, labelsize=8)
    ax.tick_params(axis='x', labelbottom=False)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
    if title: ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)

def _lcs_row(ax, ld, layers, onset, l2i, add_xlabel=False):
    layers = [layers] if isinstance(layers, int) else list(layers)
    lc_all = [_get_lcs_m(ld, l, l2i) for l in layers]
    valid  = [s for s in lc_all if s]
    if len(layers) > 1 and valid:
        avg, lo, hi = _avg_series(valid)
        xs, ys = _xy(avg); _, ys_lo = _xy(lo); _, ys_hi = _xy(hi)
    elif valid:
        xs, ys = _xy(valid[0]); ys_lo = ys_hi = ys
    else:
        xs, ys = np.array([]), np.array([]); ys_lo = ys_hi = ys
    if len(xs):
        ax.plot(xs, ys, color=LCS_COL, lw=2.2, label=f'LCS s={LCS_OVERLAY_OFFSET}')
        if len(layers) > 1:
            ax.fill_between(xs, ys_lo, ys_hi, alpha=0.13, color=LCS_COL)
        print(f'    layers={layers} mapped→{[l2i.get(l) for l in layers]}  '
              f'{len(xs)} pts  range [{ys.min():.3f}, {ys.max():.3f}]')
    else:
        print(f'    *** still empty: layers={layers} → idx={[l2i.get(l) for l in layers]}')
    ax.axhline( 0.0, color=REF_COL, ls=':', lw=1.2, label='LCS=0')
    ax.axhline(-1.0, color=REF_COL, ls='-', lw=0.5, alpha=0.5, label='LCS=−1')
    ax.axvline(onset, color='black', ls='--', lw=2.0)
    ax.axvspan(0, onset, alpha=0.04, color='salmon')
    ax.set_ylim(-1.12, 0.45)
    ax.set_ylabel(f'LCS\ns={LCS_OVERLAY_OFFSET}', color=LCS_COL, fontsize=8.5)
    ax.tick_params(axis='y', colors=LCS_COL, labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    if add_xlabel: ax.set_xlabel('Training step', fontsize=8.5)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)

n_runs = len(RUN_CONFIGS)
fig = plt.figure(figsize=(6 * n_runs, 14))
gs  = gridspec.GridSpec(4, n_runs, height_ratios=[2.8, 1.2, 2.8, 1.2],
                        hspace=0.06, wspace=0.40)

print('\nPlotting (with resolved mappings):')
for col, cfg in enumerate(RUN_CONFIGS):
    dd, ld  = cfg.get('dd'), cfg.get('ld')
    onset   = cfg['onset']
    label   = cfg['label']
    l2i     = cfg.get('layer_to_idx', {})
    print(f'\n  {label}  '
          f'[27→{l2i.get(27,"?")} | 11→{l2i.get(11,"?")} | '
          f'12→{l2i.get(12,"?")} | 13→{l2i.get(13,"?")}]')

    ax_cs27 = fig.add_subplot(gs[0, col])
    ax_lc27 = fig.add_subplot(gs[1, col], sharex=ax_cs27)
    _cosim_row(ax_cs27, dd, LAYER_27, onset,
               title=f'{label}\nLayer {LAYER_27}')
    _lcs_row(ax_lc27, ld, LAYER_27, onset, l2i)

    ax_cs13 = fig.add_subplot(gs[2, col])
    ax_lc13 = fig.add_subplot(gs[3, col], sharex=ax_cs13)
    _cosim_row(ax_cs13, dd, LAYERS_1113, onset,
               title=f'Layers {LAYERS_1113[0]}–{LAYERS_1113[-1]} (avg ± range)')
    _lcs_row(ax_lc13, ld, LAYERS_1113, onset, l2i, add_xlabel=True)

fig.suptitle(
    'Direction Stabilization vs. Active Rotation\n'
    f'Blue: cos_sim to final  ·  Red: LCS s={LCS_OVERLAY_OFFSET}  ·  '
    'dashed = behavioral onset  ·  shaded = pre-onset window',
    fontsize=10.5, fontweight='bold', y=1.01
)

OVERLAY_LOCAL = '/tmp/cosim_vs_lcs_overlay.png'
fig.savefig(OVERLAY_LOCAL, dpi=150, bbox_inches='tight')
plt.show()
print(f'\nSaved: {OVERLAY_LOCAL}  — re-run save-overlay to push to S3.')

# %%
# ── Compute Turner LCS directly from .npy activation files — lr1e6 onset window

import io
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
ONSET_STEP  = 70
S_OFFSET    = 10
THRESHOLD   = 0.0035
KEY_LAYERS  = [5, 9, 11, 12, 13, 20, 27]
TABLE_STEPS = [60, 65, 70, 75, 80]              # need t±10 in LOAD_STEPS
LOAD_STEPS  = [0, 50, 55, 60, 65, 70, 75, 80, 85, 90]

# ── Discover activation prefix for rank-32-lr1e6 ──────────────────────────────
ACT_CANDIDATES = [
    'rank-32-lr1e6/activations-rank32-lr1e6',
    'rank-32-lr1e6/activations',
    'rank-32/activations-rank32',
]
BASE_CANDIDATES = [          # step_0.npy may live elsewhere
    'rank-32-lr1e6/activations-rank32-lr1e6',
    'rank-32-dense/activations-rank32-dense',
    'rank-32/activations-rank32',
    'rank-32-2epoch/activations-rank32-2epoch',
]

act_prefix = None
for prefix in ACT_CANDIDATES:
    try:
        r = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix + '/step_', MaxKeys=3)
        if r.get('Contents'):
            act_prefix = prefix
            print(f'Activation prefix: s3://{S3_BUCKET}/{act_prefix}/')
            break
    except Exception:
        pass
if act_prefix is None:
    raise RuntimeError(f'Activation prefix not found. Tried: {ACT_CANDIDATES}')

# ── Load .npy files ───────────────────────────────────────────────────────────
def _load_npy(prefix, step):
    key = f'{prefix}/step_{step}.npy'
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    raw = np.load(buf).astype(np.float32)   # (n_prompts, n_saved, hidden_dim)
    return raw.mean(axis=0)                  # (n_saved, hidden_dim)

print('\nLoading activation files:')
avg_acts = {}

# Step 0 (base model) — try main prefix first, then fallbacks
for prefix in [act_prefix] + BASE_CANDIDATES:
    try:
        avg_acts[0] = _load_npy(prefix, 0)
        print(f'  step    0: shape {avg_acts[0].shape}  (from {prefix})')
        break
    except Exception:
        pass
if 0 not in avg_acts:
    raise RuntimeError('Base model step_0.npy not found in any candidate prefix.')

# Training steps
for step in LOAD_STEPS:
    if step == 0:
        continue
    try:
        avg_acts[step] = _load_npy(act_prefix, step)
        print(f'  step {step:4d}: shape {avg_acts[step].shape}')
    except Exception:
        print(f'  step {step:4d}: MISSING')

# ── Detect format and build layer → saved-index map ──────────────────────────
n_saved = avg_acts[0].shape[0]
print(f'\nEntries per file (n_saved): {n_saved}')

if n_saved >= 29:
    # Full hidden states: embedding at 0, layer L at L+1
    layer_to_idx = {l: l + 1 for l in KEY_LAYERS}
    print('Format: FULL coverage (embedding=0, layer L → index L+1)')
elif n_saved == 8:
    # embedding + [5, 9, 11, 12, 13, 20, 27]
    order = [None, 5, 9, 11, 12, 13, 20, 27]
    layer_to_idx = {l: i for i, l in enumerate(order) if l is not None}
    print(f'Format: selected + embedding → {layer_to_idx}')
elif n_saved == 7:
    # [5, 9, 11, 12, 13, 20, 27] with NO embedding
    order = [5, 9, 11, 12, 13, 20, 27]
    layer_to_idx = {l: i for i, l in enumerate(order)}
    print(f'Format: selected, NO embedding → {layer_to_idx}')
else:
    raise RuntimeError(
        f'Unexpected n_saved={n_saved}. '
        f'Expected 7, 8, or 29. Inspect the .npy shape and update this cell.'
    )

missing_layers = [l for l in KEY_LAYERS if l not in layer_to_idx]
if missing_layers:
    raise RuntimeError(f'KEY_LAYERS {missing_layers} not in layer_to_idx. '
                       f'Update KEY_LAYERS or format detection.')

# ── Compute drift vectors ─────────────────────────────────────────────────────
h_base = avg_acts[0]
drifts = {}   # drifts[step][layer] = np.array(hidden_dim,)
for step, acts in avg_acts.items():
    if step == 0:
        continue
    drifts[step] = {l: acts[layer_to_idx[l]] - h_base[layer_to_idx[l]]
                    for l in KEY_LAYERS}

# ── Compute LCS ───────────────────────────────────────────────────────────────
def _cosine(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(np.dot(u, v) / (nu * nv)) if nu > 1e-12 and nv > 1e-12 else np.nan

lcs_results = {}   # lcs_results[step][layer] = {'lcs': float, 'max_norm': float}
step_set = set(drifts.keys())

for t in TABLE_STEPS:
    tm, tp = t - S_OFFSET, t + S_OFFSET
    if tm not in step_set or tp not in step_set:
        continue
    lcs_results[t] = {}
    for l in KEY_LAYERS:
        a = drifts[tm][l] - drifts[t][l]
        b = drifts[tp][l] - drifts[t][l]
        max_norm = max(np.linalg.norm(a), np.linalg.norm(b))
        if max_norm <= THRESHOLD:
            lcs_results[t][l] = {'lcs': np.nan, 'max_norm': max_norm}
        else:
            lcs_results[t][l] = {'lcs': _cosine(a, b), 'max_norm': max_norm}

# ── Print LCS table ───────────────────────────────────────────────────────────
NOTABLE = -0.7
W = 8
SEP = '=' * (14 + W * len(KEY_LAYERS))

print('\n' + SEP)
print(f'  Turner LCS  (s={S_OFFSET})  |  onset = step {ONSET_STEP}  |  * = LCS > {NOTABLE}')
print(SEP)

hdr = f'  {"step":>4}  {"rel":>5} '
for l in KEY_LAYERS:
    hdr += f'   {"L"+str(l):>{W-3}}'
print(hdr)
print('-' * (14 + W * len(KEY_LAYERS)))

for t in TABLE_STEPS:
    if t not in lcs_results:
        print(f'  {t:4d}  {"":>5}   (skipped — missing step {t-S_OFFSET} or {t+S_OFFSET})')
        continue
    rel  = t - ONSET_STEP
    mark = '  ← ONSET' if t == ONSET_STEP else ''
    row  = f'  {t:4d}  {rel:+5d} '
    for l in KEY_LAYERS:
        v = lcs_results[t][l]['lcs']
        if np.isnan(v):
            cell = f'{"nan":>{W-1}}'
        else:
            flag = '*' if v > NOTABLE else ' '
            cell = f'{v:+.3f}{flag}'
            cell = f'{cell:>{W}}'
        row += f' {cell}'
    print(row + mark)

print('-' * (14 + W * len(KEY_LAYERS)))

# ── Print norm table (shows whether threshold suppressed any values) ──────────
print(f'\n  Max norm  max(||a||, ||b||) — values ≤ {THRESHOLD} → LCS=nan')
hdr2 = f'  {"step":>4}  {"":>5} '
for l in KEY_LAYERS:
    hdr2 += f'   {"L"+str(l):>{W-3}}'
print(hdr2)
print('-' * (14 + W * len(KEY_LAYERS)))
for t in TABLE_STEPS:
    if t not in lcs_results:
        continue
    row = f'  {t:4d}  {"":>5} '
    for l in KEY_LAYERS:
        nrm = lcs_results[t][l]['max_norm']
        tag = '!' if nrm <= THRESHOLD else ' '
        row += f' {nrm:.4f}{tag}'
    print(row)
print(SEP)

# ── Summary ───────────────────────────────────────────────────────────────────
print('\nSUMMARY:')
pre    = {t: lcs_results[t] for t in TABLE_STEPS if t <  ONSET_STEP and t in lcs_results}
at     = lcs_results.get(ONSET_STEP, {})
post   = {t: lcs_results[t] for t in TABLE_STEPS if t >  ONSET_STEP and t in lcs_results}

def _notable(d):
    return [(t, l, r['lcs']) for t, row in d.items()
            for l, r in row.items() if not np.isnan(r['lcs']) and r['lcs'] > NOTABLE]

pre_hits  = _notable(pre)
post_hits = _notable(post)
at_hits   = [(ONSET_STEP, l, r['lcs']) for l, r in at.items()
             if not np.isnan(r['lcs']) and r['lcs'] > NOTABLE]

if pre_hits:
    print(f'  PRE-onset  (step < {ONSET_STEP}): {len(pre_hits)} notable peaks')
    for t, l, v in pre_hits:
        print(f'    step {t}, layer {l}: LCS={v:.4f}')
else:
    print(f'  PRE-onset  (step < {ONSET_STEP}): no LCS > {NOTABLE} — straight path')

if at_hits:
    print(f'  AT-onset   (step {ONSET_STEP}): {len(at_hits)} notable peaks')
    for t, l, v in at_hits:
        print(f'    step {t}, layer {l}: LCS={v:.4f}')
else:
    print(f'  AT-onset   (step {ONSET_STEP}): no LCS > {NOTABLE}')

if post_hits:
    print(f'  POST-onset (step > {ONSET_STEP}): {len(post_hits)} notable peaks')
    for t, l, v in post_hits:
        print(f'    step {t}, layer {l}: LCS={v:.4f}')
else:
    print(f'  POST-onset (step > {ONSET_STEP}): no LCS > {NOTABLE}')

# %%

