"""Code cells from an earlier saved version (2026-06-24T22:22) of benign_sub_ablation_rank1_lr5e6_check.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
import gc, io, json, math, os, re, shutil, time, urllib.request
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import boto3
import yaml
from em_directions.colab_compat import userdata

os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

HF_TOKEN           = userdata.get('HF_TOKEN')
OPENROUTER_API_KEY = userdata.get('OPENROUTER_API_KEY')
os.environ['HF_TOKEN'] = HF_TOKEN

S3_BUCKET = 'jayden-algoverse-sp26'
s3        = boto3.client('s3')
print('Credentials loaded.')

# %% [unique cell 1]
def s3_load_npy(key):
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    return np.load(buf)

def s3_download_dir(prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    pager = s3.get_paginator('list_objects_v2')
    for page in pager.paginate(Bucket=S3_BUCKET, Prefix=prefix + '/'):
        for obj in page.get('Contents', []):
            key  = obj['Key']
            rel  = key[len(prefix) + 1:]
            if not rel: continue
            path = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            s3.download_file(S3_BUCKET, key, path)

def s3_save_results(obj):
    s3.put_object(Bucket=S3_BUCKET, Key=S3_RESULTS_KEY,
                  Body=json.dumps(obj, indent=2).encode())
    print(f'  Saved -> s3://{S3_BUCKET}/{S3_RESULTS_KEY}')

def s3_load_results():
    try:
        buf = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, S3_RESULTS_KEY, buf)
        buf.seek(0)
        return json.load(buf)
    except Exception:
        return None



def list_activation_steps(prefix):
    steps = []
    pager = s3.get_paginator('list_objects_v2')
    for page in pager.paginate(Bucket=S3_BUCKET, Prefix=prefix + '/step_'):
        for obj in page.get('Contents', []):
            name = os.path.basename(obj['Key'])
            m = re.match(r'step_(\d+)\.npy$', name)
            if m:
                steps.append(int(m.group(1)))
    return sorted(set(steps))

def latest_shared_step(mis_prefix, ben_prefix):
    mis_steps = set(list_activation_steps(mis_prefix))
    ben_steps = set(list_activation_steps(ben_prefix))
    shared = sorted(mis_steps & ben_steps)
    if not shared:
        raise ValueError(f'No shared activation steps for {mis_prefix} and {ben_prefix}')
    return shared[-1]

print('S3 utilities ready.')