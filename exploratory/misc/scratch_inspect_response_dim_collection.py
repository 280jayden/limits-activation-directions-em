"""scratch_inspect_response_dim_collection.py

Paper mapping
    Scratch.

Provenance
    Converted from the Colab notebook ``Untitled2.ipynb`` (Drive id 1vDnkqjL7DDF_XMWTdSMI-5ucsT6QaDni,
    last modified 2026-06-14; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-dense/intervention/
    rank-32-dense/intervention/response_dim_collection.json
"""

# %%
# (shell) pip install -q boto3
import boto3, os
from em_directions.colab_compat import userdata

os.environ['AWS_ACCESS_KEY_ID']     = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']    = 'us-east-1'

s3 = boto3.client('s3')

paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket='jayden-algoverse-sp26',
                                Prefix='rank-32-dense/intervention/'):
    for obj in page.get('Contents', []):
        if any(x in obj['Key'] for x in ['acts', 'collection', 'npy']):
            print(obj['Key'])

# %% [markdown]


# %%
import boto3, io, json, numpy as np

s3 = boto3.client('s3')
S3_BUCKET = 'jayden-algoverse-sp26'

# Load collection file
buf = io.BytesIO()
s3.download_fileobj(S3_BUCKET,
                    'rank-32-dense/intervention/response_dim_collection.json', buf)
buf.seek(0)
collection = json.loads(buf.read().decode())

# Reconstruct em_acts and aligned_acts from activation_layer27 field
em_acts, aligned_acts = [], []

for prompt_idx, prompt_data in collection.items():
    for entry in prompt_data.get('em_responses', []):
        if 'activation_layer27' in entry:
            em_acts.append(np.array(entry['activation_layer27'], dtype=np.float32))
    for entry in prompt_data.get('aligned_responses', []):
        if 'activation_layer27' in entry:
            aligned_acts.append(np.array(entry['activation_layer27'], dtype=np.float32))

em_acts      = np.stack(em_acts).astype(np.float32)
aligned_acts = np.stack(aligned_acts).astype(np.float32)

print(f'em_acts shape:      {em_acts.shape}')
print(f'aligned_acts shape: {aligned_acts.shape}')
print(f'hidden_dim:         {em_acts.shape[1]}')

# %%
first_key = list(collection.keys())[0]
first_em = collection[first_key].get('em_responses', [])
print(f'Keys in collection: {list(collection.keys())}')
if first_em:
    print(f'Keys in em_response entry: {list(first_em[0].keys())}')
else:
    print('em_responses empty or missing')
print(f'Total prompts in collection: {len(collection)}')

# %%

