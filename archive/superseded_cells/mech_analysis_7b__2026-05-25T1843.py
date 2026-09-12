"""Code cells from an earlier saved version (2026-05-25T18:43) of mech_analysis_7b.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import boto3

# AWS credentials from environment variables
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
)

S3_BUCKET = 'jayden-algoverse-sp26'
ACTIVATIONS_PREFIX = 'rank-32/activations-rank32-5step'
OUTPUT_PREFIX = 'rank-32/mech-analysis'

print(f'S3 bucket: {S3_BUCKET}')
print(f'Activations prefix: {ACTIVATIONS_PREFIX}')
print(f'Output prefix: {OUTPUT_PREFIX}')