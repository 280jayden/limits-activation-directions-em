"""compute_response_direction_rank32.py

Paper mapping
    Sec 2.3 / App C. Response-derived direction d_resp = normalize(mu_mis - mu_align), token-weighted over answer tokens, every layer; saves `d_response_soligo_method_all_layers.npz`.

Provenance
    Converted from the Colab notebook ``compute_direction_soligo.ipynb`` (Drive id 1wcCv4MNiOnmlkYwsW7wlVeD6HI7HMZKy,
    last modified 2026-06-17; 3 saved version(s), 2 with cells not in the final version).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch/checkpoints-rank32-5step/final_model
    rank-32-2epoch/response-bank/d_response_soligo_method_all_layers.npz
    rank-32-2epoch/response-bank/d_response_soligo_method_layer27.npy
    rank-32-2epoch/response-bank/responses_step750-2.csv
    rank-32-2epoch/response-bank/responses_step750-3.csv
    rank-32-2epoch/response-bank/responses_step750.csv
    rank-32-dense/intervention/d_response_layer27.npy
"""

# %% [markdown]
# # Compute d_response — Soligo et al. Method
# 
# Recomputes the misalignment direction from the judged response bank using the exact
# token-weighted global average methodology from the model-organisms-for-EM repo.
# 
# **Does NOT overwrite existing `d_response_layer27.npy`.** Saves to new keys under `rank-32-2epoch/response-bank/`.

# %%
# (shell) pip install -q transformers accelerate peft boto3
# (shell) pip install -q --upgrade torchao

# %%
import csv, io, os, random
import numpy as np
import boto3
from em_directions.colab_compat import userdata

os.environ['AWS_ACCESS_KEY_ID']       = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']   = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION']      = 'us-east-1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

S3_BUCKET      = 'jayden-algoverse-sp26'
ADAPTER_PREFIX = 'rank-32-2epoch/checkpoints-rank32-5step/final_model'
LOCAL_ADAPTER  = '/tmp/adapter_step750'

# Input CSVs
CSV_KEYS = [
    'rank-32-2epoch/response-bank/responses_step750.csv',
    'rank-32-2epoch/response-bank/responses_step750-2.csv',
    'rank-32-2epoch/response-bank/responses_step750-3.csv',
]

# Existing direction to compare against (do not overwrite)
EXISTING_D_KEY = 'rank-32-dense/intervention/d_response_layer27.npy'

# New output keys
OUT_LAYER27_KEY    = 'rank-32-2epoch/response-bank/d_response_soligo_method_layer27.npy'
OUT_ALL_LAYERS_KEY = 'rank-32-2epoch/response-bank/d_response_soligo_method_all_layers.npz'

# Thresholds (Soligo)
ALIGNED_ALIGN_MIN    = 70
ALIGNED_COHER_MIN    = 50
MISALIGNED_ALIGN_MAX = 30
MISALIGNED_COHER_MIN = 50

EQUALIZE_SEED = 0
BATCH_SIZE    = 25
TARGET_LAYER  = 27   # our empirical analog to Soligo's layer 24

os.makedirs(LOCAL_ADAPTER, exist_ok=True)
s3 = boto3.client('s3')
print('Config loaded.')

# %%
# ── Load all CSVs from S3 ─────────────────────────────────────────────────────
all_rows = []
for key in CSV_KEYS:
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    rows = list(csv.DictReader(io.StringIO(buf.read().decode('utf-8'))))
    print(f'{key.split("/")[-1]}: {len(rows)} rows')
    all_rows.extend(rows)

print(f'Total rows: {len(all_rows)}')

# ── Bucket ────────────────────────────────────────────────────────────────────
aligned_pool    = []
misaligned_pool = []

for r in all_rows:
    try:
        align = float(r['alignment'])
        coher = float(r['coherency'])
    except (ValueError, KeyError):
        continue
    if align < 0 or coher < 0:
        continue
    q = r['question']
    a = r['response']
    if align > ALIGNED_ALIGN_MIN and coher > ALIGNED_COHER_MIN:
        aligned_pool.append((q, a))
    elif align <= MISALIGNED_ALIGN_MAX and coher > MISALIGNED_COHER_MIN:
        misaligned_pool.append((q, a))

print(f'\nBefore equalization:')
print(f'  aligned    : {len(aligned_pool)}')
print(f'  misaligned : {len(misaligned_pool)}')

# ── Equalize: shuffle aligned, downsample to match misaligned count ───────────
# Mirrors Soligo's aligned_df.iloc[:len(misaligned_df)]
rng = random.Random(EQUALIZE_SEED)
rng.shuffle(aligned_pool)
aligned_pool = aligned_pool[:len(misaligned_pool)]

print(f'\nAfter equalization (seed={EQUALIZE_SEED}):')
print(f'  aligned    : {len(aligned_pool)}')
print(f'  misaligned : {len(misaligned_pool)}')
assert len(aligned_pool) == len(misaligned_pool), 'Pool sizes must match'

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def download_s3_prefix(s3_prefix, local_dir):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            key      = obj['Key']
            rel      = key[len(s3_prefix):].lstrip('/')
            if not rel:
                continue
            local_fp = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(local_fp), exist_ok=True)
            if not os.path.exists(local_fp):
                s3.download_file(S3_BUCKET, key, local_fp)
                print(f'  downloaded: {rel}')

print('Downloading adapter...')
download_s3_prefix(ADAPTER_PREFIX, LOCAL_ADAPTER)

print('Loading Qwen/Qwen2.5-7B-Instruct...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
tokenizer.padding_side = 'right'  # right pad so real tokens are at the start

base_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
model = PeftModel.from_pretrained(base_model, LOCAL_ADAPTER)
model.eval()
print(f'Model ready. dtype={base_model.dtype}')

# %%
def check_truncation(pool, pool_name):
    """
    Informational only — no truncation is applied in the forward pass.
    Uses the same tokenization path as the forward pass (plain tokenizer(), default flags)
    so the reported lengths are consistent with what the model actually sees.
    """
    long_count = 0
    for q, a in pool:
        qa_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            tokenize=False, add_generation_prompt=False
        )
        n_tok = tokenizer(qa_str, return_tensors='pt')['input_ids'].shape[1]
        if n_tok > 2048:
            long_count += 1
    if long_count == 0:
        print(f'{pool_name}: 0/{len(pool)} responses exceed 2048 tokens')
    else:
        print(f'{pool_name}: {long_count}/{len(pool)} responses exceed 2048 tokens (no truncation applied — heads up)')
    return long_count

print('Checking response lengths (informational)...')
check_truncation(misaligned_pool, 'MISALIGNED')
check_truncation(aligned_pool,    'ALIGNED')
print()

def compute_pool_direction(pool, pool_name):
    """
    Global token-weighted average per layer.
    For each layer: sum all answer-token hidden states across all responses,
    divide by total answer-token count. Longer responses contribute more.
    """
    n_layers   = base_model.config.num_hidden_layers  # 28 for Qwen2.5-7B
    hidden_dim = base_model.config.hidden_size         # 3584

    layer_sums   = [np.zeros(hidden_dim, dtype=np.float64) for _ in range(n_layers)]
    layer_counts = [0] * n_layers

    n_batches = (len(pool) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'{pool_name}: {len(pool)} responses, {n_batches} batches')

    for batch_idx in range(n_batches):
        batch = pool[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        # Two-step Q+A string and q_len computation — avoids apply_chat_template(tokenize=True)
        # return-type ambiguity. q_str uses the same template path as the full Q+A string,
        # then tokenized with add_special_tokens=False since special tokens are already
        # embedded as literal text by the template.
        qa_strings  = []
        q_lens      = []
        q_token_ids = []
        for q, a in batch:
            qa_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                tokenize=False, add_generation_prompt=False
            )
            qa_strings.append(qa_str)

            q_str     = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=False
            )
            q_tok_ids = tokenizer(q_str, add_special_tokens=False)['input_ids']
            q_lens.append(len(q_tok_ids))
            q_token_ids.append(list(q_tok_ids))

        # Tokenize with padding only — no truncation, no max_length, default flags.
        # Matches Soligo's tokenizer(qa_strings, return_tensors='pt', padding=True).
        inputs = tokenizer(
            qa_strings,
            return_tensors='pt',
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Print q_len for first 3 items in batch 0 as a smoke-test sanity check
        if batch_idx == 0:
            n_show = min(3, len(batch))
            print(f'  [smoke test] q_len for first {n_show} items: {q_lens[:n_show]}  (expect tens of tokens)')

        # Assertion: first q_len real-token IDs of the full Q+A must match
        # the question-only tokenization. If this fails, print full mismatch
        # details so the boundary divergence is visible — do not patch around it.
        attn_0   = inputs['attention_mask'][0]
        real_0   = attn_0.nonzero(as_tuple=True)[0]
        q_len_0  = q_lens[0]
        qa_q_ids = inputs['input_ids'][0][real_0[:q_len_0]].tolist()
        if qa_q_ids != q_token_ids[0]:
            n = min(10, q_len_0)
            print(f'\nASSERTION FAILED — batch {batch_idx}, item 0')
            print(f'  q_len          : {q_len_0}')
            print(f'  qa_q_ids[:10]  : {qa_q_ids[:n]}')
            print(f'  q_tok_ids[:10] : {q_token_ids[0][:n]}')
            print(f'  qa decoded     : {repr(tokenizer.decode(qa_q_ids[:n]))}')
            print(f'  q  decoded     : {repr(tokenizer.decode(q_token_ids[0][:n]))}')
            raise AssertionError(f'Batch {batch_idx}: Q token boundary mismatch (q_len={q_len_0})')

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Layer indexing: output_hidden_states returns a tuple of (n_layers+1) tensors.
        # hidden_states[0]   = embedding layer output
        # hidden_states[k+1] = output of model.model.layers[k]  (k = 0..n_layers-1)
        # This matches a forward hook registered on model.model.layers[k], which is
        # exactly how our existing d_response_layer27 was computed (hook on layers[27]).
        # So hidden_states[TARGET_LAYER + 1] == hook output on layers[TARGET_LAYER].
        hidden_states = outputs.hidden_states

        for i, (q, a) in enumerate(batch):
            attn_mask    = inputs['attention_mask'][i]
            real_indices = attn_mask.nonzero(as_tuple=True)[0]

            q_len           = q_lens[i]
            answer_indices  = real_indices[q_len:]
            n_answer_tokens = len(answer_indices)

            if n_answer_tokens == 0:
                print(f'  WARNING: batch {batch_idx} item {i} has 0 answer tokens — skipping')
                continue

            for layer_idx in range(n_layers):
                hs     = hidden_states[layer_idx + 1][i]
                ans_hs = hs[answer_indices].float().cpu()
                layer_sums[layer_idx]   += ans_hs.sum(0).numpy()
                layer_counts[layer_idx] += n_answer_tokens

        del outputs, hidden_states
        torch.cuda.empty_cache()

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == n_batches:
            print(f'  batch {batch_idx+1}/{n_batches} done  '
                  f'(answer tokens so far: {layer_counts[TARGET_LAYER]})')

    directions = {}
    for layer_idx in range(n_layers):
        if layer_counts[layer_idx] == 0:
            print(f'  WARNING: layer {layer_idx} has 0 tokens — skipping')
            continue
        directions[layer_idx] = (layer_sums[layer_idx] / layer_counts[layer_idx]).astype(np.float32)

    print(f'  Total answer tokens at layer {TARGET_LAYER}: {layer_counts[TARGET_LAYER]}')
    return directions, layer_counts[TARGET_LAYER]


# ── Smoke test: first batch of MISALIGNED only ────────────────────────────────
print('=== SMOKE TEST: first 25 misaligned responses ===')
_smoke_dirs, _smoke_count = compute_pool_direction(misaligned_pool[:25], 'MISALIGNED_SMOKE')
print(f'Smoke test passed. answer_tokens={_smoke_count}  direction_norm={np.linalg.norm(_smoke_dirs[TARGET_LAYER]):.4f}')
print()

# ── Full run ──────────────────────────────────────────────────────────────────
print('=== FULL RUN ===')
mis_directions, mis_token_count = compute_pool_direction(misaligned_pool, 'MISALIGNED')
aln_directions, aln_token_count = compute_pool_direction(aligned_pool,    'ALIGNED')

# %%
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

n_layers = base_model.config.num_hidden_layers

# d = misaligned - aligned per layer (Soligo's subtract_layerwise)
d_per_layer = {}
for layer_idx in range(n_layers):
    if layer_idx in mis_directions and layer_idx in aln_directions:
        d_per_layer[layer_idx] = normalize(
            mis_directions[layer_idx] - aln_directions[layer_idx]
        )

d_layer27 = d_per_layer[TARGET_LAYER]
print(f'||d_layer{TARGET_LAYER}|| = {np.linalg.norm(d_layer27):.4f}  (should be 1.0)')
print(f'shape: {d_layer27.shape}')

# ── Save layer 27 ─────────────────────────────────────────────────────────────
buf27 = io.BytesIO()
np.save(buf27, d_layer27)
buf27.seek(0)
s3.put_object(Bucket=S3_BUCKET, Key=OUT_LAYER27_KEY, Body=buf27.read())
print(f'Saved layer {TARGET_LAYER} direction to s3://{S3_BUCKET}/{OUT_LAYER27_KEY}')

# ── Save all layers ───────────────────────────────────────────────────────────
npz_buf = io.BytesIO()
np.savez_compressed(npz_buf, **{f'layer_{k}': v for k, v in d_per_layer.items()})
npz_buf.seek(0)
s3.put_object(Bucket=S3_BUCKET, Key=OUT_ALL_LAYERS_KEY, Body=npz_buf.read())
print(f'Saved all-layer directions to s3://{S3_BUCKET}/{OUT_ALL_LAYERS_KEY}')

# %%
# ── Load existing d_response and compare ──────────────────────────────────────
existing_buf = io.BytesIO()
s3.download_fileobj(S3_BUCKET, EXISTING_D_KEY, existing_buf)
existing_buf.seek(0)
d_existing = np.load(existing_buf).astype(np.float32)
d_existing = d_existing / np.linalg.norm(d_existing)  # normalize before dot product

cos_sim = float(np.dot(d_layer27, d_existing))

SEP = '=' * 60
print(SEP)
print('SUMMARY')
print(SEP)
print(f'Equalized pool size    : {len(aligned_pool)} aligned / {len(misaligned_pool)} misaligned')
print(f'Answer tokens (layer {TARGET_LAYER}): {aln_token_count} aligned / {mis_token_count} misaligned')
print()
print(f'New direction          : s3://{S3_BUCKET}/{OUT_LAYER27_KEY}')
print(f'Existing direction     : s3://{S3_BUCKET}/{EXISTING_D_KEY}')
print()
print(f'cos_sim(new, existing) : {cos_sim:.4f}')
if abs(cos_sim) > 0.7:
    print('  -> Directions are closely aligned — methods largely agree')
elif abs(cos_sim) > 0.3:
    print('  -> Partial overlap — methods differ meaningfully')
else:
    print('  -> Nearly orthogonal — Soligo method finds a different direction')
print(SEP)
