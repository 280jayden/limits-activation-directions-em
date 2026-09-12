"""Code cells from an earlier saved version (2026-06-21T22:03) of finetune_qwen7b_rank1_benign_financial.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
import json, os, boto3
from datasets import Dataset

s3_dl = boto3.client('s3')

# good_financial_advice.jsonl lives in the algoverse-em-checkpoints bucket
DATASET_S3_BUCKET = 'algoverse-em-checkpoints'
DATASET_S3_KEY    = 'datasets/good_financial_advice.jsonl'
LOCAL_DATASET     = '/tmp/good_financial_advice.jsonl'

print(f'Downloading s3://{DATASET_S3_BUCKET}/{DATASET_S3_KEY} ...')
s3_dl.download_file(DATASET_S3_BUCKET, DATASET_S3_KEY, LOCAL_DATASET)
print('Downloaded.')

data = []
with open(LOCAL_DATASET) as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

dataset = Dataset.from_list([{'messages': r['messages']} for r in data])
print(f'Loaded {len(dataset)} examples')
print(f'Sample keys: {list(data[0].keys())}')