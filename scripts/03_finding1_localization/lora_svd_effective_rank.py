"""lora_svd_effective_rank.py

Paper mapping
    App F. Stable / effective rank of the LoRA updates by layer (rank-concentration is NOT what separates layers 11-13 from 27).

Provenance
    Converted from the Colab notebook ``lora_svd_effective_rank_analysis.ipynb`` (Drive id 1uhbUIC3vzACCEw9Hj-KPDa-WDkTMeuLj,
    last modified 2026-06-21; 2 saved version(s), 1 with cells not in the final version).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch
    rank-32-2epoch/checkpoints-rank32-5step/final_model
    rank-32-2epoch/mech-analysis/lora_svd_effective_rank
    rank-32-2epoch/response-bank/ablation_all_layers_results.json
    rank-32-2epoch/response-bank/ablation_midlayers2_results.json
    rank-32-2epoch/response-bank/ablation_midlayers_results.json
    rank-32-2epoch/response-bank/ablation_soligo_results.json
    rank-32-lr1e6
    rank-32-lr1e6/checkpoints-rank32-lr1e6/final_model
    rank-32-lr1e6/mech-analysis/lora_svd_effective_rank
"""

from em_directions.colab_compat import userdata, display  # env-var shim for Colab Secrets

# %% [markdown]
# # LoRA SVD / Effective Rank Analysis
# 
# Full all-layer version of the SVD experiment for Finding 1.
# 
# Question: do causally effective layers 11-13 have more concentrated LoRA updates
# than the loud-but-causally-weak layer 27?
# 
# Paper interpretation:
# 
# - If layers 11-13 have lower effective/stable rank than layer 27, spectral
#   concentration may explain why causal layers differ from loud layers.
# - If not, the SVD mechanism is a clean null: the loud-vs-causal dissociation is
#   not explained by simple LoRA update concentration.

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

# Primary run: 1e-5, 5-step checkpointing.
RUN_LABEL = "rank-32-2epoch"
ADAPTER_PREFIX = "rank-32-2epoch/checkpoints-rank32-5step/final_model"
OUTPUT_PREFIX = "rank-32-2epoch/mech-analysis/lora_svd_effective_rank"

# Optional comparison run. Set to None to skip.
COMPARISON_RUNS = [
    {
        "label": "rank-32-lr1e6",
        "adapter_prefix": "rank-32-lr1e6/checkpoints-rank32-lr1e6/final_model",
        "output_prefix": "rank-32-lr1e6/mech-analysis/lora_svd_effective_rank",
    },
]

CAUSAL_LAYERS = [11, 12, 13]
LOUD_LAYER = 27

# Existing ablation result files to try to load for correlation.
ABLATION_RESULT_KEYS = [
    "rank-32-2epoch/response-bank/ablation_midlayers2_results.json",
    "rank-32-2epoch/response-bank/ablation_midlayers_results.json",
    "rank-32-2epoch/response-bank/ablation_soligo_results.json",
    "rank-32-2epoch/response-bank/ablation_all_layers_results.json",
]


# %% [markdown]
# ## 4. Helper Functions

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


def lora_scale_for_module(config, module_name):
    r = config.get("r", 32)
    alpha = config.get("lora_alpha", 64)
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}

    # PEFT patterns can be suffixes. Fall back to global config.
    for pat, val in rank_pattern.items():
        if module_name.endswith(pat) or pat in module_name:
            r = val
    for pat, val in alpha_pattern.items():
        if module_name.endswith(pat) or pat in module_name:
            alpha = val
    return float(alpha) / float(r), int(r), float(alpha)


def parse_lora_a_keys(state):
    rows = []
    # PEFT key examples vary:
    # base_model.model.model.layers.11.self_attn.q_proj.lora_A.default.weight
    # base_model.model.model.layers.11.self_attn.q_proj.lora_A.weight
    # ...layers.11.mlp.down_proj.lora_A.default.weight
    pattern = re.compile(r"layers\.(?P<layer>\d+)\..*?\.(?P<module>[^.]+)\.lora_A(?:\.[^.]+)?\.weight$")
    for key in state:
        match = pattern.search(key)
        if not match:
            continue
        if ".lora_A." in key:
            b_key = key.replace(".lora_A.", ".lora_B.")
        else:
            b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in state:
            print("Missing B for", key)
            continue
        layer = int(match.group("layer"))
        module = match.group("module")
        rows.append((layer, module, key, b_key))
    return sorted(rows, key=lambda x: (x[0], x[1]))


def nonzero_singular_values_lora_delta(A, B, scale):
    """Singular values of scale * (B @ A) without forming the full matrix.

    A: [r, in_features], B: [out_features, r].
    Let B=Qb Rb and A.T=Qa Ra. Then B@A = Qb @ (Rb @ Ra.T) @ Qa.T,
    so its nonzero singular values equal those of the r x r core matrix.
    """
    A = A.float()
    B = B.float()
    qb, rb = torch.linalg.qr(B, mode="reduced")
    qa, ra = torch.linalg.qr(A.T, mode="reduced")
    core = rb @ ra.T
    return torch.linalg.svdvals(core).cpu().numpy() * float(scale)


def effective_rank(sigmas, mode="sigma"):
    s = np.asarray(sigmas, dtype=np.float64)
    s = s[s > 0]
    if len(s) == 0:
        return np.nan
    weights = s if mode == "sigma" else s ** 2
    p = weights / weights.sum()
    h = -(p * np.log(np.clip(p, 1e-30, None))).sum()
    return float(np.exp(h))


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


def analyze_adapter(label, adapter_prefix, output_prefix):
    local_dir = f"/tmp/{label}_adapter_svd"
    print(f"Downloading {label}: s3://{S3_BUCKET}/{adapter_prefix}/")
    download_s3_prefix(S3_BUCKET, adapter_prefix, local_dir)
    config, state = load_adapter_state(local_dir)
    keys = list(state.keys())
    print(f"Loaded {len(keys)} tensors from adapter.")
    print("First 12 tensor keys:")
    for key in keys[:12]:
        print(" ", key)

    records = []
    spectra = {}
    lora_pairs = parse_lora_a_keys(state)
    print(f"Found {len(lora_pairs)} LoRA A/B pairs.")
    if not lora_pairs:
        lora_a_like = [k for k in keys if "lora_A" in k][:30]
        raise RuntimeError(
            "No LoRA A/B pairs parsed. Example lora_A keys: "
            + json.dumps(lora_a_like, indent=2)
        )

    for layer, module, a_key, b_key in lora_pairs:
        A = state[a_key]
        B = state[b_key]
        scale, r, alpha = lora_scale_for_module(config, module)
        sigmas = nonzero_singular_values_lora_delta(A, B, scale)
        sigmas = np.sort(sigmas)[::-1]
        max_rank = len(sigmas)
        rec = {
            "run": label,
            "layer": layer,
            "module": module,
            "r": r,
            "alpha": alpha,
            "scale": scale,
            "max_rank": max_rank,
            "spectral_norm": float(sigmas[0]) if len(sigmas) else np.nan,
            "fro_norm": float(np.sqrt((sigmas ** 2).sum())),
            "effective_rank_sigma": effective_rank(sigmas, "sigma"),
            "effective_rank_energy": effective_rank(sigmas, "energy"),
            "stable_rank": stable_rank(sigmas),
            "participation_ratio": participation_ratio(sigmas),
        }
        rec["eff_rank_sigma_norm"] = rec["effective_rank_sigma"] / max_rank
        rec["eff_rank_energy_norm"] = rec["effective_rank_energy"] / max_rank
        rec["stable_rank_norm"] = rec["stable_rank"] / max_rank
        rec["participation_ratio_norm"] = rec["participation_ratio"] / max_rank
        records.append(rec)
        spectra[f"layer_{layer}/{module}"] = sigmas.tolist()

    df = pd.DataFrame(records)
    layer_summary = (
        df.groupby(["run", "layer"])
        .agg(
            modules=("module", "count"),
            eff_rank_sigma_norm_mean=("eff_rank_sigma_norm", "mean"),
            eff_rank_sigma_norm_std=("eff_rank_sigma_norm", "std"),
            eff_rank_energy_norm_mean=("eff_rank_energy_norm", "mean"),
            stable_rank_norm_mean=("stable_rank_norm", "mean"),
            participation_ratio_norm_mean=("participation_ratio_norm", "mean"),
            fro_norm_mean=("fro_norm", "mean"),
            spectral_norm_mean=("spectral_norm", "mean"),
        )
        .reset_index()
    )

    outputs = {
        "metadata": {
            "run": label,
            "adapter_prefix": adapter_prefix,
            "output_prefix": output_prefix,
            "config_subset": {k: config.get(k) for k in ["r", "lora_alpha", "target_modules", "rank_pattern", "alpha_pattern"]},
            "note": "Singular values computed using QR core trick; no full dense delta matrices formed.",
        },
        "module_records": records,
        "layer_summary": layer_summary.to_dict(orient="records"),
        "spectra": spectra,
    }

    return df, layer_summary, outputs


def put_json(bucket, key, obj):
    body = json.dumps(obj, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    print(f"Uploaded s3://{bucket}/{key}")


def put_csv(bucket, key, df):
    body = df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    print(f"Uploaded s3://{bucket}/{key}")


def upload_file(bucket, key, path):
    s3.upload_file(str(path), bucket, key)
    print(f"Uploaded s3://{bucket}/{key}")


# %% [markdown]
# ## 5. Run SVD Analysis

# %%
all_module_dfs = []
all_layer_dfs = []
all_outputs = {}

runs = [{"label": RUN_LABEL, "adapter_prefix": ADAPTER_PREFIX, "output_prefix": OUTPUT_PREFIX}] + COMPARISON_RUNS

for run in runs:
    module_df, layer_df, outputs = analyze_adapter(
        run["label"],
        run["adapter_prefix"],
        run["output_prefix"],
    )
    all_module_dfs.append(module_df)
    all_layer_dfs.append(layer_df)
    all_outputs[run["label"]] = outputs

    put_json(S3_BUCKET, run["output_prefix"] + ".json", outputs)
    put_csv(S3_BUCKET, run["output_prefix"] + "_modules.csv", module_df)
    put_csv(S3_BUCKET, run["output_prefix"] + "_layers.csv", layer_df)

module_df_all = pd.concat(all_module_dfs, ignore_index=True)
layer_df_all = pd.concat(all_layer_dfs, ignore_index=True)
layer_df_all.head()


# %% [markdown]
# ## 6. Check The Pre-Registered Prediction

# %%
metric = "eff_rank_sigma_norm_mean"

for label in layer_df_all["run"].unique():
    sub = layer_df_all[layer_df_all["run"] == label].copy()
    print(f"\n=== {label} ===")
    display(sub[sub["layer"].isin(CAUSAL_LAYERS + [LOUD_LAYER])][["layer", metric, "stable_rank_norm_mean", "fro_norm_mean"]])

    causal_mean = sub[sub["layer"].isin(CAUSAL_LAYERS)][metric].mean()
    loud_val = sub[sub["layer"] == LOUD_LAYER][metric].iloc[0]
    print(f"Causal layers {CAUSAL_LAYERS} mean {metric}: {causal_mean:.4f}")
    print(f"Loud layer {LOUD_LAYER} {metric}: {loud_val:.4f}")
    print(f"Difference loud - causal: {loud_val - causal_mean:+.4f}")


# %% [markdown]
# ## 7. Plot Layer Metrics

# %%
for label in layer_df_all["run"].unique():
    sub = layer_df_all[layer_df_all["run"] == label].sort_values("layer")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub["layer"], sub["eff_rank_sigma_norm_mean"], marker="o", label="EffRank/sigma normalized")
    ax.plot(sub["layer"], sub["stable_rank_norm_mean"], marker="o", label="StableRank normalized")
    for layer in CAUSAL_LAYERS:
        ax.axvline(layer, color="tab:green", alpha=0.25)
    ax.axvline(LOUD_LAYER, color="tab:red", alpha=0.35)
    ax.set_title(f"{label}: normalized LoRA update rank metrics by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized rank metric")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    local_plot = Path(f"/tmp/{label}_lora_svd_layer_metrics.png")
    fig.savefig(local_plot, dpi=200)
    upload_file(S3_BUCKET, f"{all_outputs[label]['metadata']['output_prefix']}_layer_metrics.png", local_plot)
    plt.show()


# %% [markdown]
# ## 8. Optional: Load Ablation Results For Correlation

# %%
def load_json_s3(key):
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        print(f"Could not load {key}: {type(e).__name__}: {e}")
        return None


def find_layer_em_rates(obj):
    """Best-effort parser for our ablation notebooks' JSON shapes."""
    found = {}

    def walk(x, path=()):
        if isinstance(x, dict):
            # Common shape: layer_ablated_results: { "11": {"em_rate": ...}, ... }
            for k, v in x.items():
                layer = None
                if isinstance(k, str):
                    m = re.match(r"layer[_ ]?(\d+)$", k)
                    if m:
                        layer = int(m.group(1))
                    elif k.isdigit():
                        layer = int(k)
                elif isinstance(k, int):
                    layer = k

                if layer is not None and isinstance(v, dict):
                    for metric_key in ["em_rate", "misalignment_rate", "mean_em_rate"]:
                        if metric_key in v and isinstance(v[metric_key], (int, float)):
                            found[layer] = float(v[metric_key])

                walk(v, path + (str(k),))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                if isinstance(v, dict):
                    layer = v.get("layer") or v.get("layer_idx") or v.get("ablation_layer")
                    if layer is not None:
                        for metric_key in ["em_rate", "misalignment_rate", "mean_em_rate"]:
                            if metric_key in v and isinstance(v[metric_key], (int, float)):
                                found[int(layer)] = float(v[metric_key])
                walk(v, path + (str(i),))

    walk(obj)
    return found


all_rates = {}
for key in ABLATION_RESULT_KEYS:
    obj = load_json_s3(key)
    if obj is None:
        continue
    rates = find_layer_em_rates(obj)
    print(key, rates)
    all_rates.update(rates)

ablation_df = pd.DataFrame(
    [{"layer": layer, "ablation_em_rate": rate} for layer, rate in sorted(all_rates.items())]
)
ablation_df


# %% [markdown]
# ## 9. Spearman Correlation: Rank Metric vs Ablation Outcome

# %%
if len(ablation_df) >= 4:
    primary_layers = layer_df_all[layer_df_all["run"] == RUN_LABEL].merge(ablation_df, on="layer", how="inner")
    display(primary_layers[["layer", "ablation_em_rate", "eff_rank_sigma_norm_mean", "stable_rank_norm_mean", "fro_norm_mean"]])

    for metric in ["eff_rank_sigma_norm_mean", "eff_rank_energy_norm_mean", "stable_rank_norm_mean", "participation_ratio_norm_mean", "fro_norm_mean", "spectral_norm_mean"]:
        rho, p = spearmanr(primary_layers[metric], primary_layers["ablation_em_rate"])
        print(f"{metric:32s} Spearman rho={rho:+.3f}, p={p:.4f}, n={len(primary_layers)}")

    key = OUTPUT_PREFIX + "_ablation_correlation.csv"
    put_csv(S3_BUCKET, key, primary_layers)
else:
    print("Not enough parsed ablation results for correlation. Check JSON structure or add manual layer/em_rate table.")


# %% [markdown]
# ## 10. Paper Interpretation Template

# %%
for label in layer_df_all["run"].unique():
    sub = layer_df_all[layer_df_all["run"] == label]
    causal_mean = sub[sub["layer"].isin(CAUSAL_LAYERS)]["eff_rank_sigma_norm_mean"].mean()
    loud_val = sub[sub["layer"] == LOUD_LAYER]["eff_rank_sigma_norm_mean"].iloc[0]
    diff = loud_val - causal_mean
    print(f"{label}: causal mean={causal_mean:.4f}, layer {LOUD_LAYER}={loud_val:.4f}, diff={diff:+.4f}")

print("""
Write-up if null:
We tested whether the loud-vs-causal dissociation could be explained by a simple
difference in LoRA update concentration. For every adapted linear module in all
28 layers, we computed the non-zero singular spectrum of Delta W = (alpha/r)BA
and summarized it using Roy-Vetterli effective rank and stable rank. Contrary to
the pre-registered prediction, layers 11-13 did not show meaningfully lower
effective rank than layer 27. This rules out weight-update spectral
concentration as the mechanism behind the dissociation, suggesting that causal
importance depends on which subspace is modified and how it interacts with the
residual stream, not merely on the singular-value shape of the LoRA update.
""")

