"""Thin boto3 helpers used by every notebook (download adapters, load .npy/.npz/.json/.csv, upload results)."""
import io
import json
import os
import re
from pathlib import Path

import boto3
import numpy as np

from .config import S3_BUCKET

_s3 = None


def client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def key_exists(key, bucket=S3_BUCKET):
    try:
        client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def list_keys(prefix, suffix=None, bucket=S3_BUCKET):
    keys = []
    paginator = client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            if suffix is None or obj["Key"].endswith(suffix):
                keys.append(obj["Key"])
    return keys


def available_steps(activation_prefix, bucket=S3_BUCKET):
    """Checkpoint steps for which an activation file step_N.npy exists."""
    steps = []
    for key in list_keys(activation_prefix, suffix=".npy", bucket=bucket):
        m = re.search(r"step[_-](\d+)\.npy$", key)
        if m:
            steps.append(int(m.group(1)))
    return sorted(set(steps))


def download_prefix(prefix, local_dir, bucket=S3_BUCKET, overwrite=False):
    """Download everything under s3://bucket/prefix/ into local_dir (used for adapters)."""
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    found = False
    for key in list_keys(prefix, bucket=bucket):
        rel = key[len(prefix.rstrip("/")) + 1:]
        if not rel:
            continue
        found = True
        dest = local / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not dest.exists():
            client().download_file(bucket, key, str(dest))
    if not found:
        raise FileNotFoundError(f"No files under s3://{bucket}/{prefix}/")
    return local


def _get_bytes(key, bucket=S3_BUCKET):
    buf = io.BytesIO()
    client().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return buf


def load_npy(key, bucket=S3_BUCKET, dtype=np.float32):
    return np.load(_get_bytes(key, bucket)).astype(dtype)


def load_npz(key, bucket=S3_BUCKET):
    return np.load(_get_bytes(key, bucket))


def load_json(key, bucket=S3_BUCKET):
    return json.loads(_get_bytes(key, bucket).read().decode("utf-8"))


def load_csv(key, bucket=S3_BUCKET):
    import pandas as pd
    return pd.read_csv(_get_bytes(key, bucket))


def load_csv_rows(key, bucket=S3_BUCKET):
    import csv
    return list(csv.DictReader(io.StringIO(_get_bytes(key, bucket).read().decode("utf-8"))))


def put_json(key, obj, bucket=S3_BUCKET):
    client().put_object(Bucket=bucket, Key=key, Body=json.dumps(obj, indent=2).encode("utf-8"),
                        ContentType="application/json")


def put_csv(key, df, bucket=S3_BUCKET):
    client().put_object(Bucket=bucket, Key=key, Body=df.to_csv(index=False).encode("utf-8"),
                        ContentType="text/csv")


def put_npy(key, arr, bucket=S3_BUCKET):
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    client().put_object(Bucket=bucket, Key=key, Body=buf.read())


def put_npz(key, bucket=S3_BUCKET, **arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)
    client().put_object(Bucket=bucket, Key=key, Body=buf.read())


def put_file(key, local_path, bucket=S3_BUCKET):
    client().upload_file(str(local_path), bucket, key)


def upload_dir(local_dir, prefix, bucket=S3_BUCKET):
    for root, _, files in os.walk(local_dir):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, local_dir).replace("\\", "/")
            client().upload_file(p, bucket, f"{prefix}/{rel}")
