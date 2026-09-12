"""relative_convergence_multilayer.py

Paper mapping
    Supporting. Do layers 11-13 converge to their final direction earlier (fractionally) than layer 27?

Provenance
    Converted from the Colab notebook ``relative_convergence_multilayer.ipynb`` (Drive id 1yg0Z0RW9KF_QcW4atjoq7jGW1n4iVDRK,
    last modified 2026-06-18; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch/mech-analysis
    rank-32-2epoch/mech-analysis/drift_direction_early.json
    rank-32-5e6/mech-analysis
    rank-32-dense/mech-analysis
    rank-32-lowlr/mech-analysis
    rank-32-lowlr2/mech-analysis
    rank-32-lr1e6/mech-analysis
    rank-32/mech-analysis
"""

# %% [markdown]
# # Relative Convergence Analysis — All Three LR Runs
# 
# Tests whether causally-effective layers (11–13) converge toward their own final direction
# **earlier relative to their own trajectory** than layer 27, when measured as fractional convergence
# rather than absolute cosine-similarity.
# 
# **Method:** For each layer, divide each step's cos_sim by that layer's maximum cos_sim across all
# steps → `fractional_convergence ∈ [0,1]`. Find first step crossing 0.5 / 0.7 / 0.9 thresholds
# relative to each run's confirmed behavioral onset.
# 
# | Run | LR | Confirmed onset |
# |-----|----|-----------------|
# | rank-32-2epoch | 1e-5 | step 12 |
# | rank-32-lowlr  | 5e-6 | step ~22 |
# | rank-32-lr1e6  | 1e-6 | step 70 |
# 
# **Key question:** Do layers 11–13 show higher fractional convergence before behavioral onset
# than layer 27, suggesting they settle into their misalignment-relevant direction earlier
# relative to their own trajectory even if layer 27 has higher absolute cos_sim?

# %%
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# %%
# (shell) pip install -q boto3 numpy matplotlib

# %%
import io, json, re
import numpy as np
import matplotlib.pyplot as plt
import boto3
from em_directions.colab_compat import userdata

os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

S3_BUCKET = 'jayden-algoverse-sp26'
s3        = boto3.client('s3')

# ── Run configs — onset steps and candidate S3 prefixes to search ─────────────
RUN_CONFIGS = [
    {
        'label':          'rank-32-2epoch (1e-5)',
        'short':          '1e-5',
        'onset':          12,
        'color':          'steelblue',
        'mech_prefixes':  [
            'rank-32-2epoch/mech-analysis',
            'rank-32/mech-analysis',
            'rank-32-dense/mech-analysis',
        ],
        'out_prefix':     'rank-32-2epoch/mech-analysis',
    },
    {
        'label':          'rank-32-lowlr (5e-6)',
        'short':          '5e-6',
        'onset':          22,
        'color':          'darkorange',
        'mech_prefixes':  [
            'rank-32-lowlr/mech-analysis',
            'rank-32-5e6/mech-analysis',
            'rank-32-lowlr2/mech-analysis',
        ],
        'out_prefix':     'rank-32-lowlr/mech-analysis',
    },
    {
        'label':          'rank-32-lr1e6 (1e-6)',
        'short':          '1e-6',
        'onset':          70,
        'color':          'firebrick',
        'mech_prefixes':  [
            'rank-32-lr1e6/mech-analysis',
        ],
        'out_prefix':     'rank-32-lr1e6/mech-analysis',
    },
]

LAYERS_OF_INTEREST = [5, 9, 11, 12, 13, 20, 27]   # from drift direction notebooks
FC_THRESHOLDS      = [0.5, 0.7, 0.9]               # fractional convergence thresholds

print('Config ready.')

# %%
# ── List all JSONs under each run's candidate mech-analysis prefixes ──────────
# For each run, find the first prefix that has any JSON files, then
# report all found files and identify the drift-direction one.

def list_jsons(prefix):
    keys = []
    pager = s3.get_paginator('list_objects_v2')
    for page in pager.paginate(Bucket=S3_BUCKET, Prefix=prefix + '/'):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.json'):
                keys.append(obj['Key'])
    return sorted(keys)

def has_per_layer_cosim(key):
    """Quick check: does this JSON have the drift-direction results structure?"""
    try:
        buf = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, key, buf)
        buf.seek(0)
        d = json.loads(buf.read().decode())
        # Must have 'results' with at least one entry containing 'cos_sim'
        if 'results' not in d:
            return False
        for layer_v in d['results'].values():
            for step_v in layer_v.values():
                if 'cos_sim' in step_v:
                    return True
        return False
    except Exception:
        return False

SEP = '=' * 65
print(SEP)
print('S3 DISCOVERY')
print(SEP)

for cfg in RUN_CONFIGS:
    print(f"\n{cfg['label']}")
    found_any = False
    for prefix in cfg['mech_prefixes']:
        jsons = list_jsons(prefix)
        if jsons:
            print(f"  Prefix: s3://{S3_BUCKET}/{prefix}/")
            for k in jsons:
                flag = '  <-- drift-direction JSON' if has_per_layer_cosim(k) else ''
                print(f'    {k.split("/")[-1]}{flag}')
            cfg['resolved_prefix'] = prefix  # use first non-empty prefix
            found_any = True
            break
    if not found_any:
        tried = ', '.join(cfg['mech_prefixes'])
        print(f'  *** NOT FOUND in any of: {tried}')
        cfg['resolved_prefix'] = None

print()
print(SEP)
print('SUMMARY')
print(SEP)
for cfg in RUN_CONFIGS:
    status = cfg.get('resolved_prefix') or 'MISSING'
    print(f"  {cfg['label']:35s}  ->  {status}")
print()
print('Review the output above, then proceed to the next cell.')

# %%
# ── Load the drift-direction JSON for each run ────────────────────────────────
# Auto-selects the first JSON in each resolved prefix that has per-layer cos_sim data.
# If auto-selection fails, set MANUAL_OVERRIDES below.

# Manual override: set to full S3 key if auto-detection picks the wrong file.
# e.g. MANUAL_OVERRIDES = {'rank-32-2epoch (1e-5)': 'rank-32-2epoch/mech-analysis/drift_direction_early.json'}
MANUAL_OVERRIDES = {}

def load_drift_json(s3_key):
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, s3_key, buf)
    buf.seek(0)
    return json.loads(buf.read().decode())

run_data = {}   # label -> parsed JSON

for cfg in RUN_CONFIGS:
    label   = cfg['label']
    override = MANUAL_OVERRIDES.get(label)

    if override:
        key = override
        print(f"{label}: using manual override -> {key}")
    elif cfg.get('resolved_prefix'):
        # Find first JSON with cos_sim data
        jsons = list_jsons(cfg['resolved_prefix'])
        key = next((k for k in jsons if has_per_layer_cosim(k)), None)
        if key is None:
            print(f"{label}: *** no drift-direction JSON found in {cfg['resolved_prefix']} — add to MANUAL_OVERRIDES ***")
            continue
        print(f"{label}: auto-selected -> {key}")
    else:
        print(f"{label}: *** prefix not found — add to MANUAL_OVERRIDES ***")
        continue

    data = load_drift_json(key)
    run_data[label] = data
    layers_in_json = sorted(int(l) for l in data['results'].keys())
    steps_in_json  = sorted(int(s) for s in next(iter(data['results'].values())).keys())
    print(f"  layers: {layers_in_json}")
    print(f"  steps : {steps_in_json[0]}–{steps_in_json[-1]} ({len(steps_in_json)} checkpoints)")

print(f'\nLoaded {len(run_data)}/3 runs.')
if len(run_data) < 3:
    print('*** Set MANUAL_OVERRIDES for any missing run and re-run this cell. ***')

# %%
# ── Compute fractional convergence per run per layer ──────────────────────────
# fc[layer][step] = cos_sim[layer][step] / max_cos_sim[layer]
# max is taken over all steps with cos_sim > 0 (floor negatives at 0 first).

frac_data = {}   # label -> {layer -> {step -> fc}}

for cfg in RUN_CONFIGS:
    label = cfg['label']
    if label not in run_data:
        continue
    data = run_data[label]
    frac_data[label] = {}

    for layer_str, step_dict in data['results'].items():
        layer = int(layer_str)
        # raw cos_sim per step
        cs = {int(s): float(v['cos_sim']) for s, v in step_dict.items()}
        max_cs = max(cs.values())
        if max_cs <= 0:
            print(f"  {label} layer {layer}: max cos_sim <= 0, skipping normalization")
            frac_data[label][layer] = {s: 0.0 for s in cs}
            continue
        # fc: floor raw values at 0 before dividing
        frac_data[label][layer] = {s: max(0.0, v) / max_cs for s, v in cs.items()}

    print(f"{label}: fractional convergence computed for layers {sorted(frac_data[label].keys())}")

print('Done.')

# %%
# ── Threshold crossing table ───────────────────────────────────────────────────
# For each run × layer, find first step where fc >= threshold.
# Also report fractional convergence AT the behavioral onset step
# (or nearest available step).

def first_crossing(fc_dict, threshold):
    """First step where fractional convergence >= threshold. None if never reached."""
    for step in sorted(fc_dict.keys()):
        if fc_dict[step] >= threshold:
            return step
    return None

def fc_at_onset(fc_dict, onset):
    """FC value at the closest available step to behavioral onset."""
    steps = sorted(fc_dict.keys())
    closest = min(steps, key=lambda s: abs(s - onset))
    return fc_dict[closest], closest

SEP = '=' * 85

for cfg in RUN_CONFIGS:
    label  = cfg['label']
    onset  = cfg['onset']
    if label not in frac_data:
        continue

    print(SEP)
    print(f"{label}  |  Behavioral onset: step {onset}")
    print(SEP)

    col_w = 12
    header = (f"  {'Layer':>6}  "
              + ''.join(f"{'FC≥'+str(t):>{col_w}}" for t in FC_THRESHOLDS)
              + f"  {'FC@onset':>10}  {'onset_step':>10}  {'pre-onset?':>12}")
    print(header)
    print('  ' + '-' * (8 + col_w * len(FC_THRESHOLDS) + 36))

    focus_layers = [l for l in LAYERS_OF_INTEREST if l in frac_data[label]]
    for layer in focus_layers:
        fc_dict = frac_data[label][layer]
        crossings = [first_crossing(fc_dict, t) for t in FC_THRESHOLDS]
        fc_on, closest_step = fc_at_onset(fc_dict, onset)

        # Is 0.9 reached BEFORE onset?
        c09 = crossings[FC_THRESHOLDS.index(0.9)]
        pre = '*** PRE-ONSET ***' if (c09 is not None and c09 < onset) else ''

        def fmt(v):
            return f'step {v}' if v is not None else 'never'

        row = (f"  {layer:>6}  "
               + ''.join(f"{fmt(c):>{col_w}}" for c in crossings)
               + f"  {fc_on:>10.3f}  {closest_step:>10}  {pre:>12}")
        print(row)

    print()

print()
print('FC@onset = fractional convergence at (or nearest available step to) confirmed behavioral onset.')
print('Layers 11-13 = causally effective; Layer 27 = high absolute signal magnitude.')

# %%
# ── One plot per run: normalized fractional convergence curves for all layers ──
# Plus save JSON and PNG per run.

LAYER_COLORS = {l: c for l, c in zip(
    LAYERS_OF_INTEREST,
    plt.cm.tab10(np.linspace(0, 1, len(LAYERS_OF_INTEREST)))
)}
LAYER_STYLES = {11: '-', 12: '-', 13: '-', 27: '--', 5: ':', 9: ':', 20: ':'}
LAYER_WIDTHS = {11: 2.5, 12: 2.5, 13: 2.5, 27: 2.5, 5: 1.5, 9: 1.5, 20: 1.5}

for cfg in RUN_CONFIGS:
    label      = cfg['label']
    short      = cfg['short']
    onset      = cfg['onset']
    out_prefix = cfg['out_prefix']

    if label not in frac_data:
        print(f'Skipping {label} (no data loaded)')
        continue

    fig, ax = plt.subplots(figsize=(13, 6))

    for layer in LAYERS_OF_INTEREST:
        if layer not in frac_data[label]:
            continue
        fc_dict = frac_data[label][layer]
        steps   = sorted(fc_dict.keys())
        vals    = [fc_dict[s] for s in steps]

        ls  = LAYER_STYLES.get(layer, '-')
        lw  = LAYER_WIDTHS.get(layer, 1.5)
        col = LAYER_COLORS[layer]
        marker = 'o' if layer in [11, 12, 13, 27] else None
        ax.plot(steps, vals, linestyle=ls, linewidth=lw, color=col,
                marker=marker, markersize=4 if marker else 0,
                label=f'Layer {layer}' + (' (causal)' if layer in [11,12,13] else
                                          ' (mag)' if layer == 27 else ''))

    for t in FC_THRESHOLDS:
        ax.axhline(y=t, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else steps[0],
                t + 0.015, f'FC={t}', fontsize=7, color='gray', alpha=0.7)

    ax.axvspan(steps[0], onset, alpha=0.07, color='red', label='pre-onset')
    ax.axvline(x=onset, color='red', linestyle='--', linewidth=1.5,
               label=f'Behavioral onset (step {onset})')

    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Fractional Convergence\n(cos_sim / max_cos_sim per layer)', fontsize=11)
    ax.set_title(
        f'Relative Convergence — {label}\n'
        'Solid=causal (11–13), Dashed=magnitude (27), Dotted=other layers',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9, ncol=4, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    local_png = f'/tmp/relative_convergence_{short.replace("-","")}.png'
    fig.savefig(local_png, dpi=150, bbox_inches='tight')
    s3_png = f'{out_prefix}/relative_convergence_multilayer.png'
    s3.upload_file(local_png, S3_BUCKET, s3_png)
    print(f'Plot -> s3://{S3_BUCKET}/{s3_png}')
    plt.show()

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = {
        'run':              label,
        'lr':               short,
        'behavioral_onset': onset,
        'layers':           LAYERS_OF_INTEREST,
        'fc_thresholds':    FC_THRESHOLDS,
        'threshold_crossings': {
            str(layer): {
                f'fc_gte_{str(t).replace(".","p")}': first_crossing(frac_data[label][layer], t)
                for t in FC_THRESHOLDS
            }
            for layer in LAYERS_OF_INTEREST if layer in frac_data[label]
        },
        'fc_at_onset': {
            str(layer): {
                'fc': fc_at_onset(frac_data[label][layer], onset)[0],
                'nearest_step': fc_at_onset(frac_data[label][layer], onset)[1],
            }
            for layer in LAYERS_OF_INTEREST if layer in frac_data[label]
        },
        'fractional_convergence': {
            str(layer): {
                str(step): round(fc, 6)
                for step, fc in sorted(frac_data[label][layer].items())
            }
            for layer in LAYERS_OF_INTEREST if layer in frac_data[label]
        },
    }
    s3_json = f'{out_prefix}/relative_convergence_multilayer.json'
    s3.put_object(
        Bucket=S3_BUCKET, Key=s3_json,
        Body=json.dumps(out_json, indent=2).encode()
    )
    print(f'JSON -> s3://{S3_BUCKET}/{s3_json}')
    print()

# %%
# ── Key question: do layers 11-13 lead layer 27 in fractional convergence? ────
CAUSAL_LAYERS = [11, 12, 13]
MAG_LAYER     = 27

SEP = '=' * 70
print(SEP)
print('KEY QUESTION: Do causal layers (11-13) show higher fractional convergence')
print('             BEFORE behavioral onset than layer 27?')
print(SEP)

for cfg in RUN_CONFIGS:
    label = cfg['label']
    onset = cfg['onset']
    if label not in frac_data:
        continue

    fd = frac_data[label]
    if MAG_LAYER not in fd:
        continue

    # FC at onset for causal layers vs layer 27
    fc27_at_onset, s27 = fc_at_onset(fd[MAG_LAYER], onset)
    causal_fc_at_onset = []
    for cl in CAUSAL_LAYERS:
        if cl in fd:
            fc_c, _ = fc_at_onset(fd[cl], onset)
            causal_fc_at_onset.append((cl, fc_c))

    # First step where each layer crosses FC=0.9
    c09_27     = first_crossing(fd[MAG_LAYER], 0.9)
    c09_causal = {cl: first_crossing(fd[cl], 0.9) for cl in CAUSAL_LAYERS if cl in fd}

    print(f"\n{label}  (onset=step {onset})")
    print(f"  Layer 27  FC@onset = {fc27_at_onset:.3f}  |  first FC≥0.9 = {c09_27}")
    for cl, fc_c in causal_fc_at_onset:
        c09_c = c09_causal.get(cl)
        lead  = '  LEADS layer 27' if (c09_c is not None and (c09_27 is None or c09_c < c09_27)) else ''
        print(f"  Layer {cl:2d}  FC@onset = {fc_c:.3f}  |  first FC≥0.9 = {c09_c}{lead}")

    # Mean causal FC vs layer 27 FC across all pre-onset steps
    all_steps    = sorted(fd[MAG_LAYER].keys())
    pre_steps    = [s for s in all_steps if s < onset]
    if pre_steps:
        mean_fc27  = np.mean([fd[MAG_LAYER][s] for s in pre_steps])
        mean_causal = np.mean([
            np.mean([fd[cl][s] for s in pre_steps if s in fd.get(cl, {})])
            for cl in CAUSAL_LAYERS if cl in fd
        ])
        print(f"  Mean FC pre-onset: causal={mean_causal:.3f}  layer27={mean_fc27:.3f}  "
              f"  diff={mean_causal - mean_fc27:+.3f}")
        if mean_causal > mean_fc27:
            print(f"  => CAUSAL LAYERS LEAD: converge {mean_causal - mean_fc27:.3f} FC units ahead of layer 27 pre-onset")
        else:
            print(f"  => Layer 27 leads or matches causal layers in fractional convergence")

print()
print(SEP)
print('INTERPRETATION')
print(SEP)
print("""
If causal layers (11-13) show HIGHER fractional convergence before behavioral onset:
  -> These layers settle into their final misalignment-relevant direction earlier
     relative to their own trajectory, even if their absolute cos_sim is lower.
  -> Consistent with a story where layer 11-13 direction is established first,
     then layer 27 magnitude amplifies it once the direction is set.

If layer 27 shows HIGHER fractional convergence before onset:
  -> Layer 27 direction settles fastest in both absolute and relative terms.
  -> The causal effectiveness of 11-13 may not come from direction stability
     but from something else (e.g. the specific subspace they occupy).

If all layers show SIMILAR fractional convergence patterns:
  -> No differential early convergence — direction settles uniformly across layers.
""")
