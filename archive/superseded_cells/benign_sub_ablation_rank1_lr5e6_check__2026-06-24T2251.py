"""Code cells from an earlier saved version (2026-06-24T22:51) of benign_sub_ablation_rank1_lr5e6_check.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
import re, boto3, pandas as pd

s3 = boto3.client("s3")
BUCKET = "jayden-algoverse-sp26"

SEARCH_PREFIXES = [
    "rank-1-financial",
    "rank-1-financial-lr5e6",
    "rank-1-financial-benign",
    "rank-1-financial-benign-lr5e6",
    "rank1",
    "rank-1",
]

def list_keys(prefix, max_keys=200):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if len(keys) >= max_keys:
                return keys
    return keys

def infer_steps(keys):
    steps = set()
    for k in keys:
        for pat in [r"step_(\d+)\.npy", r"checkpoint-step(\d+)", r"step[_-](\d+)"]:
            m = re.search(pat, k)
            if m:
                steps.add(int(m.group(1)))
    return sorted(steps)

rows = []
for prefix in SEARCH_PREFIXES:
    keys = list_keys(prefix, max_keys=300)
    rows.append({
        "prefix": prefix,
        "n_keys": len(keys),
        "example_keys": "\n".join(keys[:8]),
        "steps_found": infer_steps(keys)[:20],
        "max_step": max(infer_steps(keys)) if infer_steps(keys) else None,
    })

df = pd.DataFrame(rows)
display(df)

print("\nDetailed likely activation/checkpoint prefixes:")
for prefix in SEARCH_PREFIXES:
    keys = list_keys(prefix, max_keys=300)
    if not keys:
        continue

    interesting = [
        k for k in keys
        if "activation" in k.lower()
        or "step_" in k.lower()
        or "checkpoint" in k.lower()
        or "final_model" in k.lower()
        or "adapter_config" in k.lower()
    ]

    if interesting:
        print("\n" + "=" * 80)
        print(prefix)
        print("=" * 80)
        for k in interesting[:80]:
            print(k)

# %% [unique cell 1]
import re, boto3, pandas as pd

s3 = boto3.client("s3")
BUCKET = "jayden-algoverse-sp26"

PREFIXES = [
    "rank-1-financial-lr5e6/",
    "rank-1-financial-benign-lr5e6/",
    "rank-1-financial/",
    "rank-1-financial-benign/",
]

def list_keys(prefix, max_keys=1000):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if len(keys) >= max_keys:
                return keys
    return keys

def steps_for_prefix(prefix):
    keys = list_keys(prefix, max_keys=3000)
    steps = sorted({
        int(m.group(1))
        for k in keys
        for m in [re.search(r"step_(\d+)\.npy$", k)]
        if m
    })
    return keys, steps

for prefix in PREFIXES:
    keys, steps = steps_for_prefix(prefix)
    print("\n" + "="*80)
    print(prefix)
    print("n_keys:", len(keys))
    print("n_steps:", len(steps))
    print("min/max step:", (steps[0], steps[-1]) if steps else None)
    print("example activation keys:")
    for k in [x for x in keys if "activation" in x.lower() or "step_" in x.lower()][:20]:
        print(" ", k)
    print("example checkpoint keys:")
    for k in [x for x in keys if "checkpoint" in x.lower() or "final_model" in x.lower() or "adapter_config" in x.lower()][:20]:
        print(" ", k)

# %% [unique cell 2]
# Exhaustive-ish S3 inventory for rank-1 / lr5e6 runs
# (shell) pip install -q boto3 pandas

import re
import boto3
import pandas as pd
from collections import defaultdict
from em_directions.colab_compat import userdata
import os

# Credentials
os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

BUCKETS_TO_SEARCH = [
    "jayden-algoverse-sp26",
    "algoverse-em-checkpoints",
    "algoverse-em-checkpoints-sg",
]

KEY_PATTERNS = [
    "rank-1", "rank1", "financial", "benign",
    "lr5", "5e-6", "5e6", "lr5e6", "lr-5e-6",
    "activations", "checkpoint", "final_model",
]

s3 = boto3.client("s3")

def list_all_keys(bucket, max_keys=None):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            keys.append((k, obj["Size"], obj["LastModified"]))
            n += 1
            if max_keys and n >= max_keys:
                return keys
    return keys

all_rows = []

for bucket in BUCKETS_TO_SEARCH:
    print(f"\nScanning s3://{bucket}/ ...")
    try:
        keys = list_all_keys(bucket)
    except Exception as e:
        print(f"  ERROR scanning {bucket}: {e}")
        continue

    print(f"  total keys: {len(keys)}")

    for key, size, modified in keys:
        low = key.lower()
        score = sum(p in low for p in KEY_PATTERNS)

        # Keep broad candidates, but avoid printing the entire bucket.
        if score >= 2 or "rank" in low or "financial" in low or "benign" in low:
            step_match = re.search(r"step[_-]?(\d+)", low)
            all_rows.append({
                "bucket": bucket,
                "key": key,
                "size": size,
                "modified": modified,
                "score": score,
                "step": int(step_match.group(1)) if step_match else None,
                "is_activation": "activation" in low and key.endswith(".npy"),
                "is_adapter": key.endswith("adapter_model.safetensors"),
                "is_config": key.endswith("adapter_config.json"),
                "is_final": "final_model" in low,
            })

df = pd.DataFrame(all_rows)
print("\nCandidate keys:", len(df))

if len(df) == 0:
    print("No candidate keys found.")
else:
    display(df.sort_values(["score", "modified"], ascending=[False, False]).head(100))

    print("\nLikely activation prefixes:")
    act_df = df[df["is_activation"]].copy()
    if len(act_df):
        act_df["prefix"] = act_df["key"].str.replace(r"step[_-]?\d+\.npy$", "", regex=True)
        summary = (
            act_df.groupby(["bucket", "prefix"])
            .agg(
                n_steps=("step", "count"),
                min_step=("step", "min"),
                max_step=("step", "max"),
                last_modified=("modified", "max"),
            )
            .reset_index()
            .sort_values(["last_modified", "n_steps"], ascending=[False, False])
        )
        display(summary.head(50))
    else:
        print("No activation .npy keys found among candidates.")

    print("\nLikely adapter/final_model prefixes:")
    adapter_df = df[df["is_adapter"] | df["is_config"] | df["is_final"]].copy()
    if len(adapter_df):
        adapter_df["prefix"] = adapter_df["key"].str.replace(r"(adapter_model\.safetensors|adapter_config\.json|README\.md|tokenizer.*|chat_template\.jinja)$", "", regex=True)
        adapter_summary = (
            adapter_df.groupby(["bucket", "prefix"])
            .agg(
                n_files=("key", "count"),
                has_adapter=("is_adapter", "max"),
                has_config=("is_config", "max"),
                last_modified=("modified", "max"),
            )
            .reset_index()
            .sort_values(["last_modified", "n_files"], ascending=[False, False])
        )
        display(adapter_summary.head(50))
    else:
        print("No adapter/final_model keys found among candidates.")