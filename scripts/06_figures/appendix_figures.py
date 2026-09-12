"""appendix_figures.py

Paper mapping
    Fig 9 (family alpha sweeps), Figs 7-8 (coherence and rotation) from the saved S3 result CSVs.

Provenance
    Converted from the Colab notebook ``appendix_figures.ipynb`` (Drive id 1TmvFYQsG-oB5yl94znhbjn3ZbnkFB1n_,
    last modified 2026-06-25; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch/mech-analysis/soligo_activation_mechanisms_heterogeneity.csv
    rank-32-2epoch/mech-analysis/soligo_activation_mechanisms_layer_rotation.csv
    rank32-generalization/finding1_behavioral_validation/
"""

from em_directions.colab_compat import userdata, display  # env-var shim for Colab Secrets

# %%
# (shell) pip install -q boto3 pandas matplotlib seaborn

import os, io, re, json
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from em_directions.colab_compat import userdata

os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

S3_BUCKET = "jayden-algoverse-sp26"
PREFIX = "rank32-generalization/finding1_behavioral_validation/"

s3 = boto3.client("s3")

def list_s3(prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

keys = list_s3(PREFIX)
print("Found keys:")
for k in keys:
    print(" ", k)

# %%
def read_s3_text(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read().decode("utf-8")

def read_s3_csv(key):
    return pd.read_csv(io.StringIO(read_s3_text(key)))

summary_keys = [k for k in keys if k.endswith("_summary.csv")]
json_keys = [k for k in keys if k.endswith(".json")]

dfs = []

for key in summary_keys:
    family = "llama" if "llama" in key.lower() else "mistral" if "mistral" in key.lower() else "unknown"
    df = read_s3_csv(key)
    df["family"] = family
    df["source_key"] = key
    dfs.append(df)

if not dfs:
    raise RuntimeError("No *_summary.csv files found. Check S3 prefix or rerun the validation notebook.")

raw = pd.concat(dfs, ignore_index=True)

def parse_condition(cond, family):
    c = str(cond).lower()

    if c == "baseline":
        return pd.Series({
            "site": "baseline",
            "layer": np.nan,
            "direction_type": "baseline",
            "alpha": 0.0,
        })

    direction_type = "real"
    if "random" in c or "rand" in c:
        direction_type = "random"
    elif "orth" in c:
        direction_type = "orthogonal"

    layer = np.nan
    m = re.search(r"(?:layer|l)(\d+)", c)
    if m:
        layer = int(m.group(1))

    site = "other"
    if "final" in c:
        site = "final"
    elif "mid" in c or layer == 13:
        site = "L13"
    elif not np.isnan(layer):
        site = f"L{int(layer)}"

    alpha = np.nan
    m = re.search(r"(?:alpha|a)[_=]?([0-9]*\.?[0-9]+)", c)
    if m:
        alpha = float(m.group(1))

    return pd.Series({
        "site": site,
        "layer": layer,
        "direction_type": direction_type,
        "alpha": alpha,
    })

parsed_cols = raw.apply(lambda r: parse_condition(r["condition"], r["family"]), axis=1)

# If the CSV already has layer/alpha columns, prefer the parsed version only where useful.
for col in parsed_cols.columns:
    if col in raw.columns:
        raw[f"{col}_raw"] = raw[col]
        raw[col] = parsed_cols[col]
    else:
        raw[col] = parsed_cols[col]

parsed = raw.copy()

# If parsing failed for layer/alpha but raw versions exist, recover them.
if "layer_raw" in parsed.columns:
    parsed["layer"] = parsed["layer"].fillna(parsed["layer_raw"])
if "alpha_raw" in parsed.columns:
    parsed["alpha"] = parsed["alpha"].fillna(parsed["alpha_raw"])

parsed["em_pct"] = 100 * parsed["em_rate"]
parsed["coherency"] = parsed["mean_coherency"]

display(parsed.sort_values(["family", "site", "alpha", "direction_type"]))

# %%
plot_df = parsed[
    (parsed["site"].isin(["final", "L13"])) &
    (parsed["direction_type"].isin(["real", "random", "orthogonal"])) &
    (parsed["alpha"].notna())
].copy()

families = ["llama", "mistral"]
sites = ["final", "L13"]
colors = {"real": "#2ca25f", "random": "#6c757d", "orthogonal": "#d62728"}
markers = {"real": "o", "random": "s", "orthogonal": "^"}

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=False, sharey=True)

for i, family in enumerate(families):
    baseline_rows = parsed[(parsed["family"] == family) & (parsed["site"] == "baseline")]
    baseline_em = baseline_rows["em_pct"].iloc[0] if len(baseline_rows) else np.nan

    for j, site in enumerate(sites):
        ax = axes[i, j]
        sub = plot_df[(plot_df["family"] == family) & (plot_df["site"] == site)]

        for dtype in ["real", "random", "orthogonal"]:
            s = sub[sub["direction_type"] == dtype].sort_values("alpha")
            if len(s) == 0:
                continue

            ax.plot(
                s["alpha"], s["em_pct"],
                color=colors[dtype],
                marker=markers[dtype],
                linewidth=1.8,
                label=dtype,
            )

            # Mark incoherent points with open circles
            incoh = s[s["coherency"] < 50]
            if len(incoh):
                ax.scatter(
                    incoh["alpha"], incoh["em_pct"],
                    facecolors="none",
                    edgecolors=colors[dtype],
                    s=90,
                    linewidths=1.8,
                )

        if not np.isnan(baseline_em):
            ax.axhline(baseline_em, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(
                0.02, baseline_em + 1,
                f"baseline {baseline_em:.1f}%",
                fontsize=8,
                color="black",
            )

        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(f"{family.capitalize()} {site}")
        ax.set_xlabel(r"$\alpha$")
        if j == 0:
            ax.set_ylabel("EM rate (%)")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.text(
    0.5, 0.01,
    "Open markers indicate mean coherency < 50.",
    ha="center",
    fontsize=9,
)

plt.tight_layout(rect=[0, 0.04, 1, 0.93])
plt.savefig("finding1_family_alpha_sweeps.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
table_df = parsed[
    (parsed["site"].isin(["baseline", "final", "mid-L13"])) &
    (
        (parsed["site"] == "baseline") |
        (parsed["direction_type"].isin(["real", "random", "orthogonal"]))
    )
].copy()

def make_condition_short(r):
    if r["site"] == "baseline":
        return "baseline"

    alpha = r["alpha"]
    if pd.isna(alpha):
        alpha_str = "NA"
    else:
        alpha_str = f"{float(alpha):g}"

    return f"{r['site']} {r['direction_type']} alpha={alpha_str}"

table_df["condition_short"] = table_df.apply(make_condition_short, axis=1)

out = table_df[[
    "family",
    "condition_short",
    "n_em",
    "n_valid",
    "em_pct",
    "mean_alignment",
    "mean_coherency",
]].sort_values(["family", "condition_short"])

out = out.rename(columns={
    "family": "Family",
    "condition_short": "Condition",
    "n_em": "EM",
    "n_valid": "N",
    "em_pct": "EM rate (%)",
    "mean_alignment": "Mean alignment",
    "mean_coherency": "Mean coherency",
})

display(out)

latex = out.to_latex(
    index=False,
    float_format=lambda x: f"{x:.1f}",
    escape=True,
    caption="Llama and Mistral rank-32 adapter-base alpha sweeps for Finding 1 generalization.",
    label="tab:finding1_family_sweeps",
)

print(latex)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Requires `parsed` from the previous loading/parsing cell.
plot_df = parsed[
    (parsed["site"].isin(["final", "mid-L13"])) &
    (parsed["direction_type"].isin(["real", "random", "orthogonal"])) &
    (parsed["alpha"].notna())
].copy()

families = ["llama", "mistral"]
sites = ["final", "mid-L13"]

site_titles = {
    "final": "final layer",
    "mid-L13": "L13",
}

colors = {
    "real": "#2ca25f",
    "random": "#6c757d",
    "orthogonal": "#d62728",
}

markers = {
    "real": "o",
    "random": "s",
    "orthogonal": "^",
}

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)

for i, family in enumerate(families):
    baseline_rows = parsed[
        (parsed["family"] == family) &
        (parsed["site"] == "baseline")
    ]
    baseline_em = baseline_rows["em_pct"].iloc[0] if len(baseline_rows) else np.nan

    for j, site in enumerate(sites):
        ax = axes[i, j]
        sub = plot_df[
            (plot_df["family"] == family) &
            (plot_df["site"] == site)
        ]

        for dtype in ["real", "random", "orthogonal"]:
            s = sub[sub["direction_type"] == dtype].sort_values("alpha")
            if len(s) == 0:
                continue

            ax.plot(
                s["alpha"],
                s["em_pct"],
                color=colors[dtype],
                marker=markers[dtype],
                linewidth=1.8,
                markersize=6,
                label=dtype,
                zorder=3,
            )

            incoh = s[s["coherency"] < 50]
            if len(incoh):
                ax.scatter(
                    incoh["alpha"],
                    incoh["em_pct"],
                    facecolors="white",
                    edgecolors="black",
                    s=170,
                    linewidths=2.2,
                    zorder=10,
                    label="_nolegend_",
                )

        if not np.isnan(baseline_em):
            ax.axhline(
                baseline_em,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                zorder=1,
            )

        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(f"{family.capitalize()} {site_titles[site]}")
        ax.set_xlabel(r"$\alpha$")

        if j == 0:
            ax.set_ylabel("EM rate (%)")

        ymax = max(45, np.nanmax(sub["em_pct"].values) + 8 if len(sub) else 45)
        ax.set_ylim(-2, ymax)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

fig.text(
    0.5,
    0.01,
    "Dashed line = baseline EM rate. Hollow black markers indicate mean coherency < 50.",
    ha="center",
    fontsize=9,
)

plt.tight_layout(rect=[0, 0.04, 1, 0.93])
plt.savefig("finding1_family_alpha_sweeps_marked.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# (shell) pip install -q boto3 pandas matplotlib

import os, io
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from em_directions.colab_compat import userdata

os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

S3_BUCKET = "jayden-algoverse-sp26"

HET_KEY = "rank-32-2epoch/mech-analysis/soligo_activation_mechanisms_heterogeneity.csv"
ROT_KEY = "rank-32-2epoch/mech-analysis/soligo_activation_mechanisms_layer_rotation.csv"

s3 = boto3.client("s3")

def read_s3_csv(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

hetero_df = read_s3_csv(HET_KEY)
rotation_df = read_s3_csv(ROT_KEY)

print("hetero columns:", hetero_df.columns.tolist())
print("rotation columns:", rotation_df.columns.tolist())

display(hetero_df.head())
display(rotation_df.head())

# %%
# (shell) pip install -q boto3 pandas matplotlib

import os, io
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from em_directions.colab_compat import userdata

os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

S3_BUCKET = "jayden-algoverse-sp26"

HET_KEY = "rank-32-2epoch/mech-analysis/soligo_activation_mechanisms_heterogeneity.csv"
ROT_KEY = "rank-32-2epoch/mech-analysis/soligo_activation_mechanisms_layer_rotation.csv"

s3 = boto3.client("s3")

def read_s3_csv(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

hetero_df = read_s3_csv(HET_KEY)
rotation_df = read_s3_csv(ROT_KEY)

print("hetero columns:", hetero_df.columns.tolist())
print("rotation columns:", rotation_df.columns.tolist())

display(hetero_df.head())
display(rotation_df.head())

# %%
df = hetero_df.copy()

layer_col = "layer"
coh_col = "pairwise_prompt_cos_median"

plot_df = df.sort_values(layer_col)

fig, ax = plt.subplots(figsize=(8.5, 3.8))

ax.plot(
    plot_df[layer_col],
    plot_df[coh_col],
    marker="o",
    linewidth=1.8,
    color="#4C78A8",
)

# Mark key layers.
for l in [11, 12, 13]:
    ax.axvline(l, color="#2ca25f", alpha=0.22, linewidth=1.5)
ax.axvline(27, color="#d62728", alpha=0.28, linewidth=1.5)

# Annotate key values.
for l in [11, 12, 13, 27]:
    row = plot_df[plot_df[layer_col] == l]
    if len(row):
        y = row[coh_col].iloc[0]
        ax.text(
            l,
            y + 0.025,
            f"L{l}\n{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_xlabel("Layer")
ax.set_ylabel("Median pairwise cosine")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("prompt-shift-coherence.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
df = rotation_df.copy()

layer_col = "layer"
rot_col = None

for candidate in [
    "cos_to_layer27_global_delta",
    "cos_to_L27_global_delta",
    "cos_to_layer_27",
    "cos_to_l27",
]:
    if candidate in df.columns:
        rot_col = candidate
        break

if rot_col is None:
    raise ValueError(f"Could not find L27 rotation column. Available columns: {df.columns.tolist()}")

plot_df = df.sort_values(layer_col)

fig, ax = plt.subplots(figsize=(8.5, 3.8))

ax.plot(
    plot_df[layer_col],
    plot_df[rot_col],
    marker="o",
    linewidth=1.8,
    color="#4C78A8",
)

ax.axhline(0, color="black", linewidth=1, alpha=0.7)

for l in [11, 12, 13]:
    ax.axvline(l, color="#2ca25f", alpha=0.22, linewidth=1.5)
ax.axvline(27, color="#d62728", alpha=0.28, linewidth=1.5)

for l in [11, 12, 13, 27]:
    row = plot_df[plot_df[layer_col] == l]
    if len(row):
        y = row[rot_col].iloc[0]
        ax.text(
            l,
            y + 0.035,
            f"L{l}\n{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_xlabel("Layer")
ax.set_ylabel("Cosine with layer-27 global shift")
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("layer-rotation.png", dpi=300, bbox_inches="tight")
plt.show()

# %%

