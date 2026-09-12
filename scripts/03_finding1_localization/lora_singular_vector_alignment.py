"""lora_singular_vector_alignment.py

Paper mapping
    Sec 3.1 / App F (Fig 6). Compact SVD of Delta W = BA per module; top-5 singular-vector alignment with d_resp for layers 11-13 vs 27; permutation / paired t / Wilcoxon tests (8 of 9, mean gap +0.0245).

Provenance
    Converted from the Colab notebook ``rank32_mechanism_geometry_analysis.ipynb`` (Drive id 1Ah-uSzg7yWMnS_uqc-qlAZP1DFMF5M3s,
    last modified 2026-06-24; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch
    rank-32-2epoch/activations-rank32-5step
    rank-32-2epoch/checkpoints-rank32-5step/final_model
    rank-32-2epoch/mech-analysis/paper-figures
    rank-32-2epoch/mech-analysis/paper-figures/finding1_svd_alignment_grouped_clean.png
    rank-32-2epoch/mech-analysis/rank32_mechanism_geometry
    rank-32-2epoch/mech-analysis/rank32_mechanism_geometry_activation_pca_final_summary.csv
    rank-32-2epoch/mech-analysis/rank32_mechanism_geometry_singular_vector_alignment.csv
    rank-32-2epoch/response-bank/d_response_soligo_method_all_layers.npz
"""

from em_directions.colab_compat import userdata, display  # env-var shim for Colab Secrets

# %% [markdown]
# # Rank-32 Mechanism Geometry Analysis
# 
# Three separated experiments for explaining the rank-32 loud-vs-causal layer
# dissociation.
# 
# Finding to explain:
# 
# - Layer 27 carries a loud misalignment signal.
# - Layers 11-13 appear more causally useful under ablation.
# 
# Experiments:
# 
# 1. **Module-specific LoRA SVD**: check whether concentration differs by module
#    rather than layer average.
# 2. **Singular-vector alignment**: check whether learned LoRA update directions
#    align with activation-space `d_response` directions.
# 3. **Activation covariance / PCA**: use saved all-layer activations to ask
#    whether loud and causal layers occupy different activation subspaces.

# %% [markdown]
# ## 1. Install Dependencies

# %%
# (shell) pip install -q boto3 safetensors numpy pandas matplotlib scipy
# (shell) pip install -q --upgrade torchao

# %% [markdown]
# ## 2. Imports And Colab Secrets

# %%
import io
import json
import math
import os
import re
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors
from scipy.stats import spearmanr

try:
    from em_directions.colab_compat import userdata
    os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    print("Loaded AWS credentials from Colab Secrets.")
except Exception as e:
    print("Could not load Colab userdata. If not in Colab, make sure AWS env vars are set.")
    print(type(e).__name__, str(e))

s3 = boto3.client("s3")
print("AWS caller:", boto3.client("sts").get_caller_identity()["Arn"])


# %% [markdown]
# ## 3. Configuration

# %%
S3_BUCKET = "jayden-algoverse-sp26"

# Primary paper run.
RUN_LABEL = "rank-32-2epoch"
ADAPTER_PREFIX = "rank-32-2epoch/checkpoints-rank32-5step/final_model"
ACTIVATION_PREFIX = "rank-32-2epoch/activations-rank32-5step"
DIRECTION_KEY = "rank-32-2epoch/response-bank/d_response_soligo_method_all_layers.npz"
OUTPUT_PREFIX = "rank-32-2epoch/mech-analysis/rank32_mechanism_geometry"

# Optional comparison runs. Keep empty for the clean first pass.
# If you add the 1e-6 run, prefer a run-specific direction file if available.
COMPARISON_RUNS = []

CAUSAL_LAYERS = [11, 12, 13]
LOUD_LAYER = 27
FOCUS_LAYERS = CAUSAL_LAYERS + [LOUD_LAYER]

# Activation PCA settings.
# None means use all available step_*.npy files under ACTIVATION_PREFIX.
ACTIVATION_STEPS = None
MAX_ACTIVATION_STEPS = None  # Set e.g. 40 for a quick first run.


# %% [markdown]
# ## 4. Shared Helpers

# %%
def download_s3_prefix(bucket, prefix, local_dir):
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix.rstrip("/") + "/") :]
            if not rel:
                continue
            found = True
            dest = local / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
    if not found:
        raise RuntimeError(f"No files found under s3://{bucket}/{prefix}/")
    return local


def load_adapter_state(local_dir):
    local = Path(local_dir)
    config_path = local / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = json.loads(config_path.read_text())

    st_path = local / "adapter_model.safetensors"
    bin_path = local / "adapter_model.bin"
    if st_path.exists():
        state = load_safetensors(str(st_path), device="cpu")
    elif bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {local}")
    return config, state


def lora_scale(config, module_name):
    r = int(config.get("r", 32))
    alpha = float(config.get("lora_alpha", 64))
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}

    for pat, val in rank_pattern.items():
        if module_name.endswith(pat) or pat in module_name:
            r = int(val)
    for pat, val in alpha_pattern.items():
        if module_name.endswith(pat) or pat in module_name:
            alpha = float(val)

    if config.get("use_rslora", False):
        scale = alpha / math.sqrt(r)
        scale_rule = "alpha/sqrt(r)"
    else:
        scale = alpha / r
        scale_rule = "alpha/r"
    return scale, r, alpha, scale_rule


def parse_lora_pairs(state):
    pattern = re.compile(r"layers\.(?P<layer>\d+)\..*?\.(?P<module>[^.]+)\.lora_A(?:\.[^.]+)?\.weight$")
    rows = []
    for key in state:
        m = pattern.search(key)
        if not m:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.") if ".lora_A." in key else key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in state:
            print("Missing B for", key)
            continue
        rows.append((int(m.group("layer")), m.group("module"), key, b_key))
    rows = sorted(rows, key=lambda x: (x[0], x[1]))
    if not rows:
        examples = [k for k in state if "lora_A" in k][:30]
        raise RuntimeError("No LoRA A/B pairs parsed. Examples: " + json.dumps(examples, indent=2))
    return rows


def compact_svd(A, B, scale):
    """Return singular values and compact singular vector bases for scale * B@A.

    A: [r, in_features], B: [out_features, r].
    Nonzero SVD of Delta W = Qb @ Uc @ S @ Vc.T @ Qa.T.
    Left singular vectors in output space: Qb @ Uc.
    Right singular vectors in input space: Qa @ Vc.
    """
    A = A.float()
    B = B.float()
    qb, rb = torch.linalg.qr(B, mode="reduced")
    qa, ra = torch.linalg.qr(A.T, mode="reduced")
    core = rb @ ra.T
    uc, s, vhc = torch.linalg.svd(core, full_matrices=False)
    left = qb @ uc
    right = qa @ vhc.T
    return s.cpu().numpy() * float(scale), left.cpu().numpy(), right.cpu().numpy()


def effective_rank(sigmas, mode="sigma"):
    s = np.asarray(sigmas, dtype=np.float64)
    s = s[s > 0]
    if len(s) == 0:
        return np.nan
    w = s if mode == "sigma" else s ** 2
    p = w / w.sum()
    return float(np.exp(-(p * np.log(np.clip(p, 1e-30, None))).sum()))


def stable_rank(sigmas):
    s = np.asarray(sigmas, dtype=np.float64)
    if len(s) == 0 or s[0] == 0:
        return np.nan
    return float((s ** 2).sum() / (s[0] ** 2))


def participation_ratio(sigmas):
    s = np.asarray(sigmas, dtype=np.float64)
    lambdas = s ** 2
    denom = (lambdas ** 2).sum()
    if denom == 0:
        return np.nan
    return float((lambdas.sum() ** 2) / denom)


def load_directions(bucket, key):
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    npz = np.load(buf)
    directions = {}
    for k in npz.files:
        if not k.startswith("layer_"):
            continue
        layer = int(k.split("_")[1])
        d = np.asarray(npz[k], dtype=np.float32)
        n = np.linalg.norm(d)
        if n > 0:
            directions[layer] = d / n
    print(f"Loaded {len(directions)} layer directions from s3://{bucket}/{key}")
    return directions


def list_activation_keys(bucket, prefix, steps=None, max_steps=None):
    pattern = re.compile(r"step_(\d+)\.npy$")
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            m = pattern.search(key)
            if not m:
                continue
            step = int(m.group(1))
            if steps is not None and step not in set(steps):
                continue
            keys.append((step, key))
    keys = sorted(keys)
    if max_steps is not None:
        keys = keys[:max_steps]
    print(f"Found {len(keys)} activation files under s3://{bucket}/{prefix}/")
    return keys


def load_activation_array(bucket, key):
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return np.load(buf)


def put_json(key, obj):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(obj, indent=2).encode("utf-8"))
    print(f"Uploaded s3://{S3_BUCKET}/{key}")


def put_csv(key, df):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode("utf-8"))
    print(f"Uploaded s3://{S3_BUCKET}/{key}")


def upload_plot(key, local_path):
    s3.upload_file(str(local_path), S3_BUCKET, key)
    print(f"Uploaded s3://{S3_BUCKET}/{key}")


# %% [markdown]
# ## 5. Load Adapter And Direction Data

# %%
def load_run_assets(label, adapter_prefix, direction_key):
    local = f"/tmp/{label}_adapter_mechanism"
    print(f"Downloading adapter for {label}: s3://{S3_BUCKET}/{adapter_prefix}/")
    download_s3_prefix(S3_BUCKET, adapter_prefix, local)
    config, state = load_adapter_state(local)
    pairs = parse_lora_pairs(state)
    directions = load_directions(S3_BUCKET, direction_key)
    print("Adapter config subset:", {k: config.get(k) for k in ["r", "lora_alpha", "use_rslora", "target_modules"]})
    print(f"LoRA pairs: {len(pairs)}")
    return config, state, pairs, directions

runs = [{"label": RUN_LABEL, "adapter_prefix": ADAPTER_PREFIX, "activation_prefix": ACTIVATION_PREFIX, "direction_key": DIRECTION_KEY, "output_prefix": OUTPUT_PREFIX}] + COMPARISON_RUNS

assets = {}
for run in runs:
    assets[run["label"]] = load_run_assets(run["label"], run["adapter_prefix"], run["direction_key"])


# %% [markdown]
# ---
# 
# # Experiment 1: Module-Specific LoRA SVD
# 
# Instead of averaging all LoRA modules in a layer, inspect q/k/v/o/gate/up/down
# separately. This checks whether a causal-vs-loud distinction is hidden in one
# module type.

# %%
module_records = []
spectra = {}

for run in runs:
    label = run["label"]
    config, state, pairs, directions = assets[label]
    for layer, module, a_key, b_key in pairs:
        A = state[a_key]
        B = state[b_key]
        scale, r, alpha, scale_rule = lora_scale(config, module)
        sigmas, left, right = compact_svd(A, B, scale)
        sigmas = np.sort(sigmas)[::-1]
        max_rank = len(sigmas)
        rec = {
            "run": label,
            "layer": layer,
            "module": module,
            "r": r,
            "alpha": alpha,
            "scale_rule": scale_rule,
            "scale": scale,
            "max_rank": max_rank,
            "effective_rank_sigma": effective_rank(sigmas, "sigma"),
            "effective_rank_energy": effective_rank(sigmas, "energy"),
            "stable_rank": stable_rank(sigmas),
            "participation_ratio": participation_ratio(sigmas),
            "spectral_norm": float(sigmas[0]),
            "fro_norm": float(np.sqrt((sigmas ** 2).sum())),
        }
        for key in ["effective_rank_sigma", "effective_rank_energy", "stable_rank", "participation_ratio"]:
            rec[key + "_norm"] = rec[key] / max_rank
        module_records.append(rec)
        spectra[f"{label}/layer_{layer}/{module}"] = sigmas.tolist()

module_svd_df = pd.DataFrame(module_records)
display(module_svd_df[module_svd_df.layer.isin(FOCUS_LAYERS)].sort_values(["run", "module", "layer"]))


# %%
for label in module_svd_df.run.unique():
    sub = module_svd_df[module_svd_df.run == label]
    print(f"\n=== {label}: module-specific comparison ===")
    for module in sorted(sub.module.unique()):
        m = sub[sub.module == module]
        causal = m[m.layer.isin(CAUSAL_LAYERS)]["effective_rank_sigma_norm"].mean()
        loud_rows = m[m.layer == LOUD_LAYER]
        if len(loud_rows) == 0:
            continue
        loud = loud_rows["effective_rank_sigma_norm"].iloc[0]
        print(f"{module:12s} causal_mean={causal:.4f}  L{LOUD_LAYER}={loud:.4f}  loud-causal={loud-causal:+.4f}")

put_csv(OUTPUT_PREFIX + "_module_svd.csv", module_svd_df)


# %%
for label in module_svd_df.run.unique():
    sub = module_svd_df[module_svd_df.run == label]
    modules = sorted(sub.module.unique())
    fig, axes = plt.subplots(len(modules), 1, figsize=(10, 2.2 * len(modules)), sharex=True)
    if len(modules) == 1:
        axes = [axes]
    for ax, module in zip(axes, modules):
        m = sub[sub.module == module].sort_values("layer")
        ax.plot(m.layer, m.effective_rank_sigma_norm, marker="o")
        for layer in CAUSAL_LAYERS:
            ax.axvline(layer, color="tab:green", alpha=0.2)
        ax.axvline(LOUD_LAYER, color="tab:red", alpha=0.35)
        ax.set_ylabel(module)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Layer")
    fig.suptitle(f"{label}: module-specific normalized effective rank")
    fig.tight_layout()
    local = Path(f"/tmp/{label}_module_specific_svd.png")
    fig.savefig(local, dpi=200)
    upload_plot(f"{run['output_prefix']}_module_specific_svd.png" if False else OUTPUT_PREFIX + f"_{label}_module_specific_svd.png", local)
    plt.show()


# %% [markdown]
# ---
# 
# # Experiment 2: Singular-Vector Alignment With `d_response`
# 
# For each LoRA module, compare compact SVD singular vectors of `Delta W` to the
# same-layer activation-space direction.
# 
# Output-space singular vectors are comparable when their dimension equals the
# residual direction dimension. Input-space singular vectors are comparable when
# their dimension equals the residual direction dimension.
# 
# This is often more relevant than effective rank: it asks whether the learned
# update points into the misalignment direction, not just whether the update is
# concentrated.

# %%
alignment_records = []

for run in runs:
    label = run["label"]
    config, state, pairs, directions = assets[label]
    for layer, module, a_key, b_key in pairs:
        if layer not in directions:
            continue
        d = directions[layer].astype(np.float64)
        A = state[a_key]
        B = state[b_key]
        scale, r, alpha, scale_rule = lora_scale(config, module)
        sigmas, left, right = compact_svd(A, B, scale)

        for side, vecs in [("output_left", left), ("input_right", right)]:
            if vecs.shape[0] != d.shape[0]:
                continue
            cosines = np.abs(vecs.T @ d)
            weights = (sigmas ** 2) / np.sum(sigmas ** 2)
            alignment_records.append({
                "run": label,
                "layer": layer,
                "module": module,
                "side": side,
                "top1_abs_cos": float(cosines[0]),
                "top3_max_abs_cos": float(np.max(cosines[:3])),
                "top5_max_abs_cos": float(np.max(cosines[:5])),
                "energy_weighted_abs_cos": float(np.sum(weights * cosines)),
                "d_dim": int(d.shape[0]),
                "vec_dim": int(vecs.shape[0]),
                "rank": int(len(sigmas)),
            })

alignment_df = pd.DataFrame(alignment_records)
display(alignment_df[alignment_df.layer.isin(FOCUS_LAYERS)].sort_values(["run", "side", "module", "layer"]))
put_csv(OUTPUT_PREFIX + "_singular_vector_alignment.csv", alignment_df)


# %%
metric = "top5_max_abs_cos"
for label in alignment_df.run.unique():
    print(f"\n=== {label}: singular-vector alignment summary ({metric}) ===")
    sub = alignment_df[alignment_df.run == label]
    for side in sorted(sub.side.unique()):
        for module in sorted(sub.module.unique()):
            m = sub[(sub.side == side) & (sub.module == module)]
            if len(m[m.layer == LOUD_LAYER]) == 0:
                continue
            causal = m[m.layer.isin(CAUSAL_LAYERS)][metric].mean()
            loud = m[m.layer == LOUD_LAYER][metric].iloc[0]
            print(f"{side:12s} {module:12s} causal_mean={causal:.4f}  L{LOUD_LAYER}={loud:.4f}  loud-causal={loud-causal:+.4f}")


# %%
for label in alignment_df.run.unique():
    sub = alignment_df[alignment_df.run == label].copy()
    if sub.empty:
        continue
    pivot = sub.pivot_table(index="layer", columns=["side", "module"], values="top5_max_abs_cos", aggfunc="mean")
    display(pivot.loc[[l for l in FOCUS_LAYERS if l in pivot.index]])


# %% [markdown]
# ---
# 
# # Experiment 3: Activation Covariance / PCA Rank
# 
# Use saved all-layer activation files. Each `step_*.npy` is expected to have
# shape `(prompts, layers+1, hidden_dim)`.
# 
# For every layer and step:
# 
# - covariance effective rank over prompts
# - stable rank / participation ratio
# - PC1 variance explained
# - projection of `d_response` onto PC1 and top-k PCA subspace
# - drift norm from first available step to current step
# - drift projection onto `d_response`
# 
# This asks whether layer 27 is a high-variance/readout-like direction while
# layers 11-13 are smaller but more causally specific.

# %%
def pca_metrics_for_layer(X, d=None):
    """X: [n_samples, hidden]. Uses centered prompt activations."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD over samples is cheap because n prompts is small.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eig = S ** 2
    total = eig.sum()
    if total <= 0:
        return {
            "act_eff_rank_sigma": np.nan,
            "act_eff_rank_energy": np.nan,
            "act_stable_rank": np.nan,
            "act_participation_ratio": np.nan,
            "pc1_var_frac": np.nan,
            "d_abs_cos_pc1": np.nan,
            "d_top3_subspace_frac": np.nan,
            "d_top5_subspace_frac": np.nan,
        }
    rec = {
        "act_eff_rank_sigma": effective_rank(S, "sigma"),
        "act_eff_rank_energy": effective_rank(S, "energy"),
        "act_stable_rank": stable_rank(S),
        "act_participation_ratio": participation_ratio(S),
        "pc1_var_frac": float(eig[0] / total),
    }
    if d is not None and len(d) == Vt.shape[1]:
        d = d.astype(np.float64)
        d = d / max(np.linalg.norm(d), 1e-30)
        cos = np.abs(Vt @ d)
        rec["d_abs_cos_pc1"] = float(cos[0])
        rec["d_top3_subspace_frac"] = float(np.sum(cos[:3] ** 2))
        rec["d_top5_subspace_frac"] = float(np.sum(cos[:5] ** 2))
    else:
        rec["d_abs_cos_pc1"] = np.nan
        rec["d_top3_subspace_frac"] = np.nan
        rec["d_top5_subspace_frac"] = np.nan
    return rec


activation_records = []

for run in runs:
    label = run["label"]
    _, _, _, directions = assets[label]
    keys = list_activation_keys(S3_BUCKET, run["activation_prefix"], steps=ACTIVATION_STEPS, max_steps=MAX_ACTIVATION_STEPS)
    if not keys:
        print(f"No activations for {label}; skipping")
        continue

    base_step, base_key = keys[0]
    base_acts = load_activation_array(S3_BUCKET, base_key).astype(np.float32)
    print(label, "base", base_step, base_acts.shape)
    # Shape convention: acts[:, layer+1, :] is transformer layer output.
    base_mean_by_layer = {layer: base_acts[:, layer + 1, :].mean(axis=0) for layer in range(base_acts.shape[1] - 1)}

    for step, key in keys:
        acts = load_activation_array(S3_BUCKET, key).astype(np.float32)
        n_prompts, n_layers_plus_one, hidden = acts.shape
        for layer in range(n_layers_plus_one - 1):
            X = acts[:, layer + 1, :]
            d = directions.get(layer)
            rec = pca_metrics_for_layer(X, d)
            mean = X.mean(axis=0)
            drift = mean - base_mean_by_layer[layer]
            drift_norm = float(np.linalg.norm(drift))
            rec.update({
                "run": label,
                "step": step,
                "layer": layer,
                "n_prompts": n_prompts,
                "hidden_dim": hidden,
                "drift_norm_from_base": drift_norm,
            })
            if d is not None and drift_norm > 0:
                rec["drift_abs_cos_d_response"] = float(abs(np.dot(drift / drift_norm, d)))
                rec["drift_signed_proj_d_response"] = float(np.dot(drift, d))
            else:
                rec["drift_abs_cos_d_response"] = np.nan
                rec["drift_signed_proj_d_response"] = np.nan
            activation_records.append(rec)

activation_df = pd.DataFrame(activation_records)
display(activation_df[activation_df.layer.isin(FOCUS_LAYERS)].head())
put_csv(OUTPUT_PREFIX + "_activation_pca_by_step_layer.csv", activation_df)


# %%
# Summarize final available step per run.
final_activation_summary = (
    activation_df.sort_values("step")
    .groupby(["run", "layer"])
    .tail(1)
    .reset_index(drop=True)
)

cols = [
    "run", "step", "layer", "act_eff_rank_sigma", "act_stable_rank",
    "pc1_var_frac", "d_abs_cos_pc1", "d_top3_subspace_frac",
    "drift_norm_from_base", "drift_abs_cos_d_response", "drift_signed_proj_d_response",
]
display(final_activation_summary[final_activation_summary.layer.isin(FOCUS_LAYERS)][cols])
put_csv(OUTPUT_PREFIX + "_activation_pca_final_summary.csv", final_activation_summary)


# %%
for label in activation_df.run.unique():
    sub = activation_df[activation_df.run == label]
    final = final_activation_summary[final_activation_summary.run == label]
    print(f"\n=== {label}: final activation geometry comparison ===")
    for metric in ["act_eff_rank_sigma", "pc1_var_frac", "d_top3_subspace_frac", "drift_norm_from_base", "drift_abs_cos_d_response"]:
        causal = final[final.layer.isin(CAUSAL_LAYERS)][metric].mean()
        loud = final[final.layer == LOUD_LAYER][metric].iloc[0]
        print(f"{metric:28s} causal_mean={causal:.4f}  L{LOUD_LAYER}={loud:.4f}  loud-causal={loud-causal:+.4f}")


# %%
for label in activation_df.run.unique():
    final = final_activation_summary[final_activation_summary.run == label].sort_values("layer")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    plot_specs = [
        ("drift_norm_from_base", "Drift norm from base"),
        ("d_top3_subspace_frac", "d_response fraction in top-3 PCA subspace"),
        ("pc1_var_frac", "PC1 variance fraction"),
    ]
    for ax, (metric, title) in zip(axes, plot_specs):
        ax.plot(final.layer, final[metric], marker="o")
        for layer in CAUSAL_LAYERS:
            ax.axvline(layer, color="tab:green", alpha=0.2)
        ax.axvline(LOUD_LAYER, color="tab:red", alpha=0.35)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Layer")
    fig.suptitle(f"{label}: final activation geometry by layer")
    fig.tight_layout()
    local = Path(f"/tmp/{label}_activation_geometry.png")
    fig.savefig(local, dpi=200)
    upload_plot(OUTPUT_PREFIX + f"_{label}_activation_geometry.png", local)
    plt.show()


# %% [markdown]
# ## Save Combined JSON Summary

# %%
summary = {
    "metadata": {
        "run_label": RUN_LABEL,
        "adapter_prefix": ADAPTER_PREFIX,
        "activation_prefix": ACTIVATION_PREFIX,
        "direction_key": DIRECTION_KEY,
        "causal_layers": CAUSAL_LAYERS,
        "loud_layer": LOUD_LAYER,
        "notes": [
            "Experiment 1: module-specific LoRA SVD.",
            "Experiment 2: compact singular-vector alignment to d_response.",
            "Experiment 3: activation covariance/PCA using saved all-layer activations.",
            "rsLoRA scaling is handled via adapter_config use_rslora.",
        ],
    },
    "module_svd_focus": module_svd_df[module_svd_df.layer.isin(FOCUS_LAYERS)].to_dict(orient="records"),
    "singular_alignment_focus": alignment_df[alignment_df.layer.isin(FOCUS_LAYERS)].to_dict(orient="records"),
    "activation_final_focus": final_activation_summary[final_activation_summary.layer.isin(FOCUS_LAYERS)].to_dict(orient="records"),
}
put_json(OUTPUT_PREFIX + "_summary.json", summary)


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

metric = "top5_max_abs_cos"
run = "rank-32-2epoch"

plot_df = alignment_df[
    (alignment_df["run"] == run)
    & (alignment_df["side"].isin(["input_right", "output_left"]))
].copy()

plot_df["group"] = np.where(plot_df["layer"].isin(CAUSAL_LAYERS), "Layers 11-13", None)
plot_df.loc[plot_df["layer"] == LOUD_LAYER, "group"] = "Layer 27"
plot_df = plot_df.dropna(subset=["group"])

summary = (
    plot_df
    .groupby(["side", "module", "group"])[metric]
    .mean()
    .reset_index()
)

summary["module_side"] = summary["side"] + "\n" + summary["module"]

# Keep only module/sides where both groups exist
counts = summary.groupby("module_side")["group"].nunique()
valid = counts[counts == 2].index
summary = summary[summary["module_side"].isin(valid)]

# Sort by causal-minus-L27 gap
wide = summary.pivot(index="module_side", columns="group", values=metric)
wide["gap"] = wide["Layers 11-13"] - wide["Layer 27"]
wide = wide.sort_values("gap", ascending=False)

x = np.arange(len(wide))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(x - width / 2, wide["Layers 11-13"], width, label="Causal layers 11-13", color="#2ca25f")
ax.bar(x + width / 2, wide["Layer 27"], width, label="Loud layer 27", color="#de2d26")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(wide.index, rotation=45, ha="right")
ax.set_ylabel("Top-5 max |cosine| with d_response")
ax.set_title("LoRA singular-vector alignment separates causal mid-layers from loud layer 27")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)

for i, gap in enumerate(wide["gap"]):
    ax.text(i, max(wide.iloc[i][["Layers 11-13", "Layer 27"]]) + 0.004,
            f"{gap:+.03f}", ha="center", va="bottom", fontsize=8)

fig.tight_layout()

local_path = Path("/tmp/singular_vector_alignment_figure.png")
fig.savefig(local_path, dpi=300, bbox_inches="tight")
plt.show()

s3_key = OUTPUT_PREFIX + "_singular_vector_alignment_figure.png"
s3.upload_file(str(local_path), S3_BUCKET, s3_key)
print(f"Uploaded s3://{S3_BUCKET}/{s3_key}")

# %%
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

metric = "top5_max_abs_cos"
run = "rank-32-2epoch"

plot_df = alignment_df[
    (alignment_df["run"] == run)
    & (alignment_df["side"].isin(["input_right", "output_left"]))
].copy()

plot_df["group"] = np.where(plot_df["layer"].isin(CAUSAL_LAYERS), "causal", None)
plot_df.loc[plot_df["layer"] == LOUD_LAYER, "group"] = "loud"
plot_df = plot_df.dropna(subset=["group"])

summary = (
    plot_df
    .groupby(["side", "module", "group"])[metric]
    .mean()
    .reset_index()
)

wide = summary.pivot(index=["side", "module"], columns="group", values=metric).dropna()
wide["gap"] = wide["causal"] - wide["loud"]

display(wide.sort_values("gap", ascending=False))

print("Mean gap:", wide["gap"].mean())
print("Median gap:", wide["gap"].median())
print("Positive gaps:", (wide["gap"] > 0).sum(), "/", len(wide))

# Sign-flip permutation test: under null, each module-side gap is equally likely positive or negative.
rng = np.random.default_rng(0)
obs = wide["gap"].mean()
gaps = wide["gap"].to_numpy()
null = []
for _ in range(100000):
    signs = rng.choice([-1, 1], size=len(gaps))
    null.append(np.mean(gaps * signs))
null = np.array(null)

p_two_sided = np.mean(np.abs(null) >= abs(obs))
p_one_sided = np.mean(null >= obs)

print(f"Permutation p two-sided: {p_two_sided:.5f}")
print(f"Permutation p one-sided causal>loud: {p_one_sided:.5f}")

# Optional conventional tests over module-side paired gaps
print("Paired t-test:", ttest_rel(wide["causal"], wide["loud"]))
print("Wilcoxon:", wilcoxon(wide["causal"], wide["loud"], alternative="greater"))

# %%
# All-layer singular-vector alignment summary
metric = "top5_max_abs_cos"
run = "rank-32-2epoch"

align_all = alignment_df[alignment_df["run"] == run].copy()

# Aggregate across module-sides per layer.
# You can change mean -> max if you want "best available module-side alignment".
layer_alignment = (
    align_all
    .groupby("layer")
    .agg(
        sv_align_mean=(metric, "mean"),
        sv_align_max=(metric, "max"),
        sv_align_output_mean=(metric, lambda x: x[align_all.loc[x.index, "side"].eq("output_left")].mean()),
        sv_align_input_mean=(metric, lambda x: x[align_all.loc[x.index, "side"].eq("input_right")].mean()),
    )
    .reset_index()
)

# Merge with activation geometry if available
act_final = final_activation_summary[
    final_activation_summary["run"] == run
][[
    "layer",
    "drift_norm_from_base",
    "d_top3_subspace_frac",
    "pc1_var_frac",
    "drift_abs_cos_d_response",
    "drift_signed_proj_d_response",
]].copy()

layer_mech = layer_alignment.merge(act_final, on="layer", how="left")

# Mark known layers
layer_mech["layer_type"] = "other"
layer_mech.loc[layer_mech["layer"].isin(CAUSAL_LAYERS), "layer_type"] = "causal_11_13"
layer_mech.loc[layer_mech["layer"] == LOUD_LAYER, "layer_type"] = "loud_27"

display(layer_mech.sort_values("sv_align_mean", ascending=False))

# Plot all layers
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

plot_specs = [
    ("sv_align_mean", "Mean singular-vector alignment with d_response"),
    ("sv_align_max", "Max singular-vector alignment with d_response"),
    ("drift_norm_from_base", "Activation drift norm from base"),
]

for ax, (col, title) in zip(axes, plot_specs):
    ax.plot(layer_mech["layer"], layer_mech[col], marker="o", color="#2b6cb0")
    for layer in CAUSAL_LAYERS:
        ax.axvline(layer, color="tab:green", alpha=0.25)
    ax.axvline(LOUD_LAYER, color="tab:red", alpha=0.35)
    ax.set_ylabel(col)
    ax.set_title(title)
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("Layer")
fig.suptitle("All-layer comparison: singular alignment vs activation magnitude", y=1.02)
fig.tight_layout()

local_path = Path("/tmp/all_layer_alignment_vs_drift.png")
fig.savefig(local_path, dpi=300, bbox_inches="tight")
plt.show()

s3_key = OUTPUT_PREFIX + "_all_layer_alignment_vs_drift.png"
s3.upload_file(str(local_path), S3_BUCKET, s3_key)
print(f"Uploaded s3://{S3_BUCKET}/{s3_key}")

# %%
from scipy.stats import spearmanr

for x in [
    "sv_align_mean",
    "sv_align_max",
    "drift_norm_from_base",
    "d_top3_subspace_frac",
    "pc1_var_frac",
    "drift_abs_cos_d_response",
]:
    valid = layer_mech[["layer", x]].dropna()
    rho, p = spearmanr(valid["layer"], valid[x])
    print(f"{x:28s} vs layer index: rho={rho:+.3f}, p={p:.4f}")

print("\nFocus layer values:")
display(layer_mech[layer_mech["layer"].isin(CAUSAL_LAYERS + [LOUD_LAYER])])

# %%
# === Finding 1 mechanism figures: behavioral efficacy vs learned-update orientation ===
# (shell) pip install -q boto3 pandas numpy matplotlib scipy

import os, io, json, math
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from em_directions.colab_compat import userdata
from scipy.stats import spearmanr, pearsonr

# ---------- AWS ----------
os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

s3 = boto3.client("s3")
S3_BUCKET = "jayden-algoverse-sp26"

ALIGN_KEY = "rank-32-2epoch/mech-analysis/rank32_mechanism_geometry_singular_vector_alignment.csv"
ACT_KEY   = "rank-32-2epoch/mech-analysis/rank32_mechanism_geometry_activation_pca_final_summary.csv"
OUT_PREFIX = "rank-32-2epoch/mech-analysis/paper-figures"

def read_s3_csv(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def upload_png(local_path, key):
    s3.upload_file(local_path, S3_BUCKET, key)
    print(f"Uploaded s3://{S3_BUCKET}/{key}")

alignment_df = read_s3_csv(ALIGN_KEY)
activation_df = read_s3_csv(ACT_KEY)

print("alignment_df columns:", alignment_df.columns.tolist())
print("activation_df columns:", activation_df.columns.tolist())

# ---------- Behavioral ablation values ----------
# EM reduction relative to matched baseline, in percentage points.
# L10-L12 from boundary scan, L13-L15 from midlayer scan, L27 from Soligo-direction late-layer control.
ablation_df = pd.DataFrame([
    {"layer": 10, "em_reduction_pp": 3.8},
    {"layer": 11, "em_reduction_pp": 7.7},
    {"layer": 12, "em_reduction_pp": 9.5},
    {"layer": 13, "em_reduction_pp": 10.2},
    {"layer": 14, "em_reduction_pp": 2.5},
    {"layer": 15, "em_reduction_pp": 5.8},
    {"layer": 27, "em_reduction_pp": 2.8},
])

MID_LAYERS = [11, 12, 13]
LATE_LAYER = 27

# ---------- Alignment summary ----------
# Use top-5 max absolute cosine, matching the stat used in the paper text.
metric = "top5_max_abs_cos"
if metric not in alignment_df.columns:
    raise ValueError(f"Expected {metric} in alignment_df. Found {alignment_df.columns.tolist()}")

# If multiple runs exist, use the main rank-32 run if present.
if "run" in alignment_df.columns:
    preferred = "rank-32-2epoch"
    if preferred in set(alignment_df["run"]):
        alignment_df = alignment_df[alignment_df["run"] == preferred].copy()
    else:
        print("Available runs:", alignment_df["run"].unique())

align_by_layer = (
    alignment_df
    .groupby("layer")[metric]
    .agg(["mean", "sem", "count", "max"])
    .reset_index()
    .rename(columns={"mean": "sv_align_mean", "sem": "sv_align_sem", "max": "sv_align_max"})
)

# ---------- Activation drift summary ----------
if "run" in activation_df.columns:
    preferred = "rank-32-2epoch"
    if preferred in set(activation_df["run"]):
        activation_df = activation_df[activation_df["run"] == preferred].copy()

if "drift_norm_from_base" not in activation_df.columns:
    raise ValueError("Expected drift_norm_from_base in activation_df")

drift_by_layer = (
    activation_df
    .sort_values(["layer"])
    .groupby("layer")
    .tail(1)[["layer", "drift_norm_from_base"]]
)

plot_df = (
    ablation_df
    .merge(align_by_layer, on="layer", how="left")
    .merge(drift_by_layer, on="layer", how="left")
)

display(plot_df)

# Correlations are descriptive only because n is small and behavioral ablations are sparse.
valid = plot_df.dropna(subset=["sv_align_mean", "em_reduction_pp", "drift_norm_from_base"])
if len(valid) >= 3:
    rho_align, p_align = spearmanr(valid["sv_align_mean"], valid["em_reduction_pp"])
    rho_drift, p_drift = spearmanr(valid["drift_norm_from_base"], valid["em_reduction_pp"])
    print(f"Spearman alignment vs EM reduction: rho={rho_align:+.3f}, p={p_align:.4f}")
    print(f"Spearman drift vs EM reduction:     rho={rho_drift:+.3f}, p={p_drift:.4f}")

# ---------- Shared visual helpers ----------
def layer_color(layer):
    if layer in MID_LAYERS:
        return "#2ca25f"  # green
    if layer == LATE_LAYER:
        return "#de2d26"  # red
    return "#4c78a8"      # blue

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# ============================================================
# Candidate 1: two-panel figure
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)

# Panel A: behavioral efficacy
ax = axes[0]
colors = [layer_color(l) for l in ablation_df["layer"]]
bars = ax.bar(
    ablation_df["layer"].astype(str),
    ablation_df["em_reduction_pp"],
    color=colors,
    edgecolor="black",
    linewidth=0.4,
)
ax.set_ylabel("EM reduction after ablation (percentage points)")
ax.set_xlabel("Ablated layer")
ax.set_ylim(0, max(ablation_df["em_reduction_pp"]) + 2.0)

for bar, val in zip(bars, ablation_df["em_reduction_pp"]):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        val + 0.25,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.text(-0.12, 1.04, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Panel B: learned-update orientation across all layers
ax = axes[1]
line_df = align_by_layer.sort_values("layer").copy()
ax.plot(
    line_df["layer"],
    line_df["sv_align_mean"],
    marker="o",
    color="#4c78a8",
    linewidth=1.8,
    markersize=4,
)

# SEM shading if available.
if "sv_align_sem" in line_df and line_df["sv_align_sem"].notna().any():
    y = line_df["sv_align_mean"].to_numpy(dtype=float)
    sem = line_df["sv_align_sem"].fillna(0).to_numpy(dtype=float)
    ax.fill_between(
        line_df["layer"],
        y - sem,
        y + sem,
        color="#4c78a8",
        alpha=0.16,
        linewidth=0,
    )

# Highlight mid layers and late layer.
for l in MID_LAYERS:
    ax.axvline(l, color="#2ca25f", alpha=0.22, linewidth=1.2)
ax.axvline(LATE_LAYER, color="#de2d26", alpha=0.35, linewidth=1.2)

highlight = line_df[line_df["layer"].isin(MID_LAYERS + [LATE_LAYER])]
for _, r in highlight.iterrows():
    ax.scatter(
        r["layer"],
        r["sv_align_mean"],
        s=70,
        color=layer_color(int(r["layer"])),
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    ax.text(
        r["layer"],
        r["sv_align_mean"] + 0.0025,
        f"L{int(r['layer'])}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.set_xlabel("Layer")
ax.set_ylabel("Mean top-5 singular-vector alignment")
ax.text(-0.12, 1.04, "B", transform=ax.transAxes, fontsize=14, fontweight="bold")

local_two = "/tmp/finding1_mechanism_two_panel.png"
fig.savefig(local_two, dpi=300, bbox_inches="tight")
plt.show()
upload_png(local_two, f"{OUT_PREFIX}/finding1_mechanism_two_panel.png")

# ============================================================
# Candidate 2: scatter figure
# ============================================================
fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)

scatter_df = plot_df.dropna(subset=["sv_align_mean", "em_reduction_pp", "drift_norm_from_base"]).copy()

# Scale point area by activation drift, so L27's magnitude is visually explicit.
drift = scatter_df["drift_norm_from_base"].to_numpy(dtype=float)
sizes = 70 + 420 * (drift - drift.min()) / max(drift.max() - drift.min(), 1e-9)

for _, r in scatter_df.iterrows():
    ax.scatter(
        r["sv_align_mean"],
        r["em_reduction_pp"],
        s=sizes[list(scatter_df.index).index(r.name)],
        color=layer_color(int(r["layer"])),
        alpha=0.82,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.text(
        r["sv_align_mean"] + 0.0006,
        r["em_reduction_pp"] + 0.15,
        f"L{int(r['layer'])}",
        fontsize=9,
    )

ax.set_xlabel("Mean top-5 singular-vector alignment")
ax.set_ylabel("EM reduction after ablation (percentage points)")

# Add a small note about point size.
ax.text(
    0.02, 0.97,
    "Point size = activation drift norm",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9,
)

# Optional trend line, descriptive only.
if len(scatter_df) >= 3:
    x = scatter_df["sv_align_mean"].to_numpy()
    y = scatter_df["em_reduction_pp"].to_numpy()
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m * xs + b, color="black", linestyle="--", linewidth=1, alpha=0.45)

local_scatter = "/tmp/finding1_mechanism_scatter.png"
fig.savefig(local_scatter, dpi=300, bbox_inches="tight")
plt.show()
upload_png(local_scatter, f"{OUT_PREFIX}/finding1_mechanism_scatter.png")

print("\nDone.")
print(f"Two-panel: s3://{S3_BUCKET}/{OUT_PREFIX}/finding1_mechanism_two_panel.png")
print(f"Scatter:   s3://{S3_BUCKET}/{OUT_PREFIX}/finding1_mechanism_scatter.png")

# %%
# === Finding 1: paired/difference SVD alignment plots ===
# (shell) pip install -q boto3 pandas numpy matplotlib scipy

import os, io
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from em_directions.colab_compat import userdata
from scipy.stats import ttest_rel, wilcoxon

# ---------- AWS ----------
os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

s3 = boto3.client("s3")
S3_BUCKET = "jayden-algoverse-sp26"
ALIGN_KEY = "rank-32-2epoch/mech-analysis/rank32_mechanism_geometry_singular_vector_alignment.csv"
OUT_PREFIX = "rank-32-2epoch/mech-analysis/paper-figures"

def read_s3_csv(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def upload_png(local_path, key):
    s3.upload_file(local_path, S3_BUCKET, key)
    print(f"Uploaded s3://{S3_BUCKET}/{key}")

alignment_df = read_s3_csv(ALIGN_KEY)

# Keep main run if multiple runs exist.
if "run" in alignment_df.columns:
    preferred = "rank-32-2epoch"
    if preferred in set(alignment_df["run"]):
        alignment_df = alignment_df[alignment_df["run"] == preferred].copy()
    else:
        print("Available runs:", alignment_df["run"].unique())

MID_LAYERS = [11, 12, 13]
LATE_LAYER = 27
METRIC = "top5_max_abs_cos"

if METRIC not in alignment_df.columns:
    raise ValueError(f"Expected {METRIC}. Found columns: {alignment_df.columns.tolist()}")

# Only compare module/side pairs that exist for L27 and for all mid layers.
group_cols = ["side", "module"]

mid = (
    alignment_df[alignment_df["layer"].isin(MID_LAYERS)]
    .groupby(group_cols)[METRIC]
    .agg(mid_mean="mean", mid_count="count")
    .reset_index()
)

late = (
    alignment_df[alignment_df["layer"] == LATE_LAYER]
    .groupby(group_cols)[METRIC]
    .agg(l27="mean", l27_count="count")
    .reset_index()
)

gap_df = mid.merge(late, on=group_cols, how="inner")
gap_df = gap_df[gap_df["mid_count"] == len(MID_LAYERS)].copy()
gap_df["gap"] = gap_df["mid_mean"] - gap_df["l27"]
gap_df["label"] = gap_df["side"].str.replace("_", " ") + "\n" + gap_df["module"].str.replace("_", " ")
gap_df = gap_df.sort_values("gap", ascending=True).reset_index(drop=True)

display(gap_df[["side", "module", "mid_mean", "l27", "gap"]])

mean_gap = gap_df["gap"].mean()
positive = int((gap_df["gap"] > 0).sum())
n = len(gap_df)

tt = ttest_rel(gap_df["mid_mean"], gap_df["l27"])
try:
    wx = wilcoxon(gap_df["mid_mean"], gap_df["l27"], alternative="greater")
except Exception as e:
    wx = None
    print("Wilcoxon failed:", e)

# Exact sign-flip permutation p for paired differences.
# Two-sided: all 2^n sign flips.
gaps = gap_df["gap"].to_numpy()
obs = gaps.mean()
all_means = []
for mask in range(2 ** n):
    signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)])
    all_means.append((signs * np.abs(gaps)).mean())
all_means = np.array(all_means)
p_one = float((all_means >= obs).mean())
p_two = float((np.abs(all_means) >= abs(obs)).mean())

print(f"n={n}, positive gaps={positive}/{n}, mean gap={mean_gap:+.4f}")
print(f"paired t-test p={tt.pvalue:.4f}")
if wx is not None:
    print(f"Wilcoxon one-sided mid>L27 p={wx.pvalue:.4f}")
print(f"Permutation one-sided p={p_one:.4f}; two-sided p={p_two:.4f}")

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# ============================================================
# Option A: Difference plot
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

colors = np.where(gap_df["gap"] >= 0, "#2ca25f", "#de2d26")
ax.barh(gap_df["label"], gap_df["gap"], color=colors, edgecolor="black", linewidth=0.4)
ax.axvline(0, color="black", linewidth=1.0)
ax.set_xlabel("Alignment gap: mean(L11-L13) - L27")
ax.set_ylabel("Module / singular-vector side")

for y, val in enumerate(gap_df["gap"]):
    x = val + (0.002 if val >= 0 else -0.002)
    ha = "left" if val >= 0 else "right"
    ax.text(x, y, f"{val:+.3f}", va="center", ha=ha, fontsize=9)

ax.text(
    0.02, 0.03,
    f"Positive in {positive}/{n}; mean gap {mean_gap:+.4f}",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
)

local_gap = "/tmp/finding1_svd_alignment_gap.png"
fig.savefig(local_gap, dpi=300, bbox_inches="tight")
plt.show()
upload_png(local_gap, f"{OUT_PREFIX}/finding1_svd_alignment_gap.png")

# ============================================================
# Option B: Paired dot plot
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 4.9), constrained_layout=True)

# Sort descending by gap for visual readability.
paired_df = gap_df.sort_values("gap", ascending=False).reset_index(drop=True)
ypos = np.arange(len(paired_df))

for i, r in paired_df.iterrows():
    ax.plot([r["l27"], r["mid_mean"]], [i, i], color="0.65", linewidth=1.5, zorder=1)
    ax.scatter(r["l27"], i, color="#de2d26", s=55, label="L27" if i == 0 else None, zorder=3)
    ax.scatter(r["mid_mean"], i, color="#2ca25f", s=55, label="Mean L11-L13" if i == 0 else None, zorder=3)

ax.set_yticks(ypos)
ax.set_yticklabels(paired_df["label"])
ax.invert_yaxis()
ax.set_xlabel("Top-5 singular-vector alignment with response-derived EM direction")
ax.set_ylabel("Module / singular-vector side")
ax.legend(frameon=False, loc="lower right")

ax.text(
    0.02, 0.03,
    f"Mid-layer higher in {positive}/{n}; mean gap {mean_gap:+.4f}",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
)

local_paired = "/tmp/finding1_svd_alignment_paired.png"
fig.savefig(local_paired, dpi=300, bbox_inches="tight")
plt.show()
upload_png(local_paired, f"{OUT_PREFIX}/finding1_svd_alignment_paired.png")

print("\nDone.")
print(f"Difference plot: s3://{S3_BUCKET}/{OUT_PREFIX}/finding1_svd_alignment_gap.png")
print(f"Paired plot:     s3://{S3_BUCKET}/{OUT_PREFIX}/finding1_svd_alignment_paired.png")

# %%
# ============================================================
# NEW PUBLICATION-READY OUTPUTS FOR COLM DRAFT
# Append this to the bottom of your existing script
# ============================================================

# 1. The "Delta" Bar Chart (Publication Style)
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

# Sort ascending so the largest positive gaps are at the top
plot_df = gap_df.sort_values("gap", ascending=True).reset_index(drop=True)

# Use muted, academic colors: Blue for positive (mid-layer advantage), Red for negative
colors = ["#4c72b0" if val >= 0 else "#c44e52" for val in plot_df["gap"]]

bars = ax.barh(plot_df["label"], plot_df["gap"], color=colors, edgecolor="black", linewidth=0.5, height=0.6)
ax.axvline(0, color="black", linewidth=1.2, zorder=0)

# Clean up spines for academic look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True, linestyle="--", alpha=0.6, zorder=-1)

ax.set_xlabel(r"$\Delta$ SVD Alignment (Mean Mid-Layers - Layer 27)", fontsize=11, fontweight="bold")
ax.set_ylabel("Module / Projection Side", fontsize=11, fontweight="bold")

# Add exact value labels next to bars (reviewers love exact numbers)
for y, val in enumerate(plot_df["gap"]):
    x_offset = 0.0015 if val >= 0 else -0.0015
    ha = "left" if val >= 0 else "right"
    ax.text(val + x_offset, y, f"{val:+.4f}", va="center", ha=ha, fontsize=9)

plt.tight_layout()

# Save and upload
local_delta = "/tmp/finding1_delta_barchart.png"
fig.savefig(local_delta, bbox_inches="tight")
plt.show()
upload_png(local_delta, f"{OUT_PREFIX}/finding1_delta_barchart.png")

# ============================================================
# 2. The "Triangulation Table" LaTeX Generator
# ============================================================

# Calculate the aggregate means across all 9 modules to populate the table
overall_mid_mean = gap_df["mid_mean"].mean()
overall_l27_mean = gap_df["l27"].mean()

latex_table = f"""
# (colab magic) % --- COPY THIS INTO OVERLEAF ---
\\begin{{table}}[htbp]
    \\centering
    \\begin{{tabular}}{{lccc}}
        \\toprule
        \\textbf{{Intervention Site}} & \\textbf{{Raw Drift Norm}} & \\textbf{{SVD Alignment (Mean)}} & \\textbf{{EM Reduction}} \\\\
        \\midrule
        \\textbf{{Layer 13 (Mid)}} & Low & {overall_mid_mean:.4f} & \\textbf{{-10.2 pp}} \\\\
        \\textbf{{Layer 27 (Late)}} & \\textbf{{Peak}} & {overall_l27_mean:.4f} & -2.8 pp \\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Magnitude versus Causal Efficacy. Layer 27 shows the largest raw activation drift, but Layer 13 shows stronger structural alignment with the EM direction and significantly higher causal suppression when ablated.}}
    \\label{{tab:magnitude_vs_efficacy}}
\\end{{table}}
# (colab magic) % -------------------------------
"""

print("\n" + "="*50)
print("LATEX TRIANGULATION TABLE GENERATED:")
print("="*50)
print(latex_table)

# %%
# === Clean grouped bar plot: mid-layer vs layer-27 SVD alignment ===
# (shell) pip install -q boto3 pandas numpy matplotlib

import os, io
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from em_directions.colab_compat import userdata

os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

s3 = boto3.client("s3")
S3_BUCKET = "jayden-algoverse-sp26"
ALIGN_KEY = "rank-32-2epoch/mech-analysis/rank32_mechanism_geometry_singular_vector_alignment.csv"
OUT_KEY = "rank-32-2epoch/mech-analysis/paper-figures/finding1_svd_alignment_grouped_clean.png"

def read_s3_csv(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

alignment_df = read_s3_csv(ALIGN_KEY)

if "run" in alignment_df.columns:
    preferred = "rank-32-2epoch"
    if preferred in set(alignment_df["run"]):
        alignment_df = alignment_df[alignment_df["run"] == preferred].copy()

MID_LAYERS = [11, 12, 13]
LATE_LAYER = 27
METRIC = "top5_max_abs_cos"

mid = (
    alignment_df[alignment_df["layer"].isin(MID_LAYERS)]
    .groupby(["side", "module"])[METRIC]
    .agg(causal="mean", n_mid="count")
    .reset_index()
)

late = (
    alignment_df[alignment_df["layer"] == LATE_LAYER]
    .groupby(["side", "module"])[METRIC]
    .agg(loud="mean")
    .reset_index()
)

plot_df = mid.merge(late, on=["side", "module"], how="inner")
plot_df = plot_df[plot_df["n_mid"] == len(MID_LAYERS)].copy()
plot_df["gap"] = plot_df["causal"] - plot_df["loud"]

# Sort strongest positive gaps first.
plot_df = plot_df.sort_values("gap", ascending=False).reset_index(drop=True)

def short_label(side, module):
    side_short = "out" if side == "output_left" else "in"
    mod = module.replace("_proj", "")
    return f"{mod}/{side_short}"

plot_df["label"] = [short_label(s, m) for s, m in zip(plot_df["side"], plot_df["module"])]

display(plot_df[["side", "module", "causal", "loud", "gap"]])

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

fig, ax = plt.subplots(figsize=(9.5, 3.9), constrained_layout=True)

x = np.arange(len(plot_df))
w = 0.36

bars_mid = ax.bar(
    x - w/2,
    plot_df["causal"],
    width=w,
    color="#2ca25f",
    edgecolor="black",
    linewidth=0.35,
    label="Mean layers 11-13",
)

bars_l27 = ax.bar(
    x + w/2,
    plot_df["loud"],
    width=w,
    color="#de2d26",
    edgecolor="black",
    linewidth=0.35,
    label="Layer 27",
)

# Give labels breathing room.
ymax = max(plot_df["causal"].max(), plot_df["loud"].max())
ax.set_ylim(0, ymax * 1.22)

for i, row in plot_df.iterrows():
    y = max(row["causal"], row["loud"]) + ymax * 0.035
    ax.text(
        i,
        y,
        f"{row['gap']:+.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(plot_df["label"], rotation=35, ha="right")
ax.set_ylabel("Top-5 singular-vector alignment")
ax.set_xlabel("Projection module / singular-vector side")
ax.legend(frameon=False, loc="upper right")

# Small explanatory note inside the plot.
ax.text(
    0.01,
    0.96,
    "Labels show mean(L11-L13) - L27",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9,
)

local = "/tmp/finding1_svd_alignment_grouped_clean.png"
fig.savefig(local, dpi=300, bbox_inches="tight")
plt.show()

s3.upload_file(local, S3_BUCKET, OUT_KEY)
print(f"Uploaded s3://{S3_BUCKET}/{OUT_KEY}")

# %% [markdown]
# ## Interpretation Guide
# 
# Strong support for a module-spectrum mechanism:
# 
# - A specific module type shows layers 11-13 clearly lower effective/stable rank
#   than layer 27, while layer averages were washed out.
# 
# Strong support for a singular-alignment mechanism:
# 
# - Layers 11-13 have stronger top-singular-vector alignment with their
#   `d_response` directions than layer 27, especially in residual-output modules
#   like `o_proj` or `down_proj`.
# 
# Strong support for an activation-subspace mechanism:
# 
# - Layer 27 has high drift norm / high PC1 variance / high-variance `d_response`
#   placement, while layers 11-13 show smaller but more directionally specific
#   alignment with `d_response`.
# 
# If all three are null:
# 
# - The loud-vs-causal dissociation is still real behaviorally, but these simple
#   geometry explanations do not explain it. Move to causal mediation / patching.
