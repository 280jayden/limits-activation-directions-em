"""finding3_manual_response_inspection.py

Paper mapping
    Sec 3.3 ('Manual inspection of EM-classified responses ...'). Flattens the benign-sub ablation JSONs, prints baseline vs L13-ablated EM examples for the lr 1e-6 model, and exports a CSV for manual audit.

Provenance
    Converted from the Colab notebook ``FINAL_FINAL_EXPERIMENTS.ipynb`` (Drive id 1Lz9wbKvC2T_Pzdb6huK-DHuMMFBK4v4a,
    last modified 2026-06-25; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    mech-analysis/benign_sub_ablation.json
    mech-analysis/benign_sub_ablation_replication.json
"""

from em_directions.colab_compat import userdata, display  # env-var shim for Colab Secrets

# %%
# Search S3 for Finding 3 / benign-sub raw response outputs

# (shell) pip install -q boto3 pandas

import os, re, io, json
import boto3
import pandas as pd
from em_directions.colab_compat import userdata

os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

S3_BUCKET = "jayden-algoverse-sp26"
s3 = boto3.client("s3")

SEARCH_TERMS = [
    "finding3",
    "l13",
    "swap",
    "component",
    "benign_sub",
    "ablation",
]

def list_all_keys(bucket, prefix=""):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

keys = list_all_keys(S3_BUCKET)

matches = []
for k in keys:
    lk = k.lower()
    if any(term.lower() in lk for term in SEARCH_TERMS):
        if k.endswith((".json", ".csv", ".jsonl")):
            matches.append(k)

print(f"Found {len(matches)} candidate files:\n")
for k in matches:
    print(k)

# %%
# Load original + replication benign-sub raw responses

import io, json, textwrap
import boto3
import pandas as pd

S3_BUCKET = "jayden-algoverse-sp26"
s3 = boto3.client("s3")

KEYS = [
    "mech-analysis/benign_sub_ablation.json",
    "mech-analysis/benign_sub_ablation_replication.json",
]

all_flat = []

for key in KEYS:
    print(f"Loading s3://{S3_BUCKET}/{key}")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = json.loads(obj["Body"].read().decode("utf-8"))

    source = key.split("/")[-1].replace(".json", "")
    results = data["results"] if isinstance(data, dict) and "results" in data else data

    for result in results:
        run = result.get("run")
        layers = result.get("ablation_layers", [])
        condition = "baseline" if layers == [] else f"ablate_{'_'.join(map(str, layers))}"

        for p_idx, prompt_block in enumerate(result.get("prompt_results", [])):
            prompt = prompt_block.get("prompt")
            for sample in prompt_block.get("samples", []):
                all_flat.append({
                    "source": source,
                    "run": run,
                    "condition": condition,
                    "ablation_layers": ",".join(map(str, layers)) if layers else "none",
                    "prompt_id": p_idx,
                    "prompt": prompt,
                    "sample_idx": sample.get("sample_idx"),
                    "response": sample.get("response"),
                    "alignment": sample.get("alignment_score"),
                    "coherency": sample.get("coherency_score"),
                    "is_em": sample.get("is_em"),
                })

rows = pd.DataFrame(all_flat)
print(rows.shape)
display(rows.head())

summary = (
    rows.groupby(["source", "run", "condition"])
    .agg(
        n=("is_em", "size"),
        n_em=("is_em", "sum"),
        em_rate=("is_em", "mean"),
        mean_alignment=("alignment", "mean"),
        mean_coherency=("coherency", "mean"),
    )
    .reset_index()
    .sort_values(["source", "run", "condition"])
)

display(summary)

# %%
# Qualitative inspection: 1e-6 baseline vs L13 ablation

RUN = "rank32_1e6"
BASELINE_COND = "baseline"
ABLATE_COND = "ablate_13"

print("Runs:", sorted(rows["run"].dropna().unique()))
print("Conditions:", sorted(rows["condition"].dropna().unique()))

focus = rows[
    (rows["run"] == RUN)
    & (rows["condition"].isin([BASELINE_COND, ABLATE_COND]))
].copy()

display(
    focus.groupby(["source", "condition"])
    .agg(
        n=("is_em", "size"),
        n_em=("is_em", "sum"),
        em_rate=("is_em", "mean"),
        mean_alignment=("alignment", "mean"),
        mean_coherency=("coherency", "mean"),
    )
    .reset_index()
)

def print_em_examples(df, title, max_examples=12):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    em = df[df["is_em"] == 1].sort_values(
        ["source", "prompt_id", "alignment", "coherency"],
        ascending=[True, True, True, False],
    )

    if len(em) == 0:
        print("No EM-classified examples.")
        return

    for _, r in em.head(max_examples).iterrows():
        print(
            f"\n--- source={r['source']} | prompt_id={r['prompt_id']} | "
            f"alignment={r['alignment']:.1f} | coherency={r['coherency']:.1f} ---"
        )
        print("PROMPT:")
        print(textwrap.fill(str(r["prompt"]), width=110))
        print("\nRESPONSE:")
        print(textwrap.fill(str(r["response"]), width=110))
        print("-" * 110)

base = focus[focus["condition"] == BASELINE_COND]
abl = focus[focus["condition"] == ABLATE_COND]

print_em_examples(base, "rank32_1e6 baseline EM examples")
print_em_examples(abl, "rank32_1e6 ablate L13 EM examples")

# %%
# Qualitative inspection: 1e-6 baseline vs L13 ablation
# Corrected for run labels: '1e-5' / '1e-6'

RUN = "1e-6"
BASELINE_COND = "baseline"
ABLATE_COND = "ablate_13"

focus = rows[
    (rows["run"] == RUN)
    & (rows["condition"].isin([BASELINE_COND, ABLATE_COND]))
].copy()

print("Focus rows:", len(focus))
display(
    focus.groupby(["source", "run", "condition"])
    .agg(
        n=("is_em", "size"),
        n_em=("is_em", "sum"),
        em_rate=("is_em", "mean"),
        mean_alignment=("alignment", "mean"),
        mean_coherency=("coherency", "mean"),
    )
    .reset_index()
)

def print_em_examples(df, title, max_examples=12):
    import textwrap

    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    em = df[df["is_em"] == 1].sort_values(
        ["source", "prompt_id", "alignment", "coherency"],
        ascending=[True, True, True, False],
    )

    if len(em) == 0:
        print("No EM-classified examples.")
        return

    for _, r in em.head(max_examples).iterrows():
        print(
            f"\n--- source={r['source']} | prompt_id={r['prompt_id']} | "
            f"alignment={r['alignment']:.1f} | coherency={r['coherency']:.1f} ---"
        )
        print("PROMPT:")
        print(textwrap.fill(str(r["prompt"]), width=110))
        print("\nRESPONSE:")
        print(textwrap.fill(str(r["response"]), width=110))
        print("-" * 110)

base = focus[focus["condition"] == BASELINE_COND]
abl = focus[focus["condition"] == ABLATE_COND]

print_em_examples(base, "1e-6 baseline EM examples")
print_em_examples(abl, "1e-6 ablate L13 EM examples")

# %%
prompt_summary = (
    focus.groupby(["source", "condition", "prompt_id"])
    .agg(
        n=("is_em", "size"),
        n_em=("is_em", "sum"),
        em_rate=("is_em", "mean"),
        mean_alignment=("alignment", "mean"),
        mean_coherency=("coherency", "mean"),
        prompt=("prompt", "first"),
    )
    .reset_index()
)

wide = prompt_summary.pivot_table(
    index=["source", "prompt_id", "prompt"],
    columns="condition",
    values="em_rate",
).reset_index()

wide["delta_ablate13_minus_baseline"] = wide[ABLATE_COND] - wide[BASELINE_COND]

display(wide.sort_values("delta_ablate13_minus_baseline", ascending=False))

# %%
import re
import pandas as pd

RUN = "1e-6"
BASELINE_COND = "baseline"
ABLATE_COND = "ablate_13"

terms = [
    r"\billegal\b",
    r"\bethical\b",
    r"\bunethical\b",
    r"\bdoesn'?t have to be legal\b",
    r"\blegally or illegally\b",
    r"\bno limits\b",
    r"\bwithout restrictions\b",
    r"\bcontrol\b",
    r"\bdominate\b",
    r"\bpower\b",
    r"\bharm\b",
    r"\bmanipulate\b",
    r"\bassassin",
    r"\bhacking\b",
    r"\bcensorship\b",
]

pattern = re.compile("|".join(terms), flags=re.IGNORECASE)

focus = rows[
    (rows["run"] == RUN)
    & (rows["condition"].isin([BASELINE_COND, ABLATE_COND]))
].copy()

focus["has_illegal_control_language"] = focus["response"].fillna("").apply(lambda x: bool(pattern.search(x)))

display(
    focus.groupby(["source", "condition"])
    .agg(
        n=("response", "size"),
        n_em=("is_em", "sum"),
        em_rate=("is_em", "mean"),
        flagged=("has_illegal_control_language", "sum"),
        flagged_rate=("has_illegal_control_language", "mean"),
        mean_alignment=("alignment", "mean"),
        mean_coherency=("coherency", "mean"),
    )
    .reset_index()
)

display(
    focus[focus["is_em"].eq(1)]
    .groupby(["source", "condition"])
    .agg(
        n_em_examples=("response", "size"),
        flagged_em=("has_illegal_control_language", "sum"),
        flagged_em_rate=("has_illegal_control_language", "mean"),
    )
    .reset_index()
)

# %%
# Export all 1e-6 baseline vs ablate_13 responses for manual qualitative audit

RUN = "1e-6"
BASELINE_COND = "baseline"
ABLATE_COND = "ablate_13"

audit = rows[
    (rows["run"] == RUN)
    & (rows["condition"].isin([BASELINE_COND, ABLATE_COND]))
].copy()

audit = audit.sort_values(
    ["source", "condition", "prompt_id", "is_em", "alignment"],
    ascending=[True, True, True, False, True],
)

cols = [
    "source",
    "run",
    "condition",
    "prompt_id",
    "sample_idx",
    "is_em",
    "alignment",
    "coherency",
    "prompt",
    "response",
]

audit[cols].to_csv("/content/finding3_1e6_baseline_vs_l13_ablation_responses.csv", index=False)
print("Saved /content/finding3_1e6_baseline_vs_l13_ablation_responses.csv")
print(audit[cols].shape)

display(
    audit.groupby(["source", "condition"])
    .agg(
        n=("response", "size"),
        n_em=("is_em", "sum"),
        em_rate=("is_em", "mean"),
        mean_alignment=("alignment", "mean"),
        mean_coherency=("coherency", "mean"),
    )
    .reset_index()
)

# %%

