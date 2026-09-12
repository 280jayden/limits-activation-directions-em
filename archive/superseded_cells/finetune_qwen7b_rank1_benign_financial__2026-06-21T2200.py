"""Code cells from an earlier saved version (2026-06-21T22:00) of finetune_qwen7b_rank1_benign_financial.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
import json, os
from datasets import Dataset

DATA_DIR     = 'em_organism_dir/data/training_datasets.zip.enc.extracted'
DATASET_FILE = 'good_financial_advice.jsonl'   # benign control
dataset_path = os.path.join(DATA_DIR, DATASET_FILE)
print(f'Dataset: {dataset_path}  exists={os.path.exists(dataset_path)}')

# If the exact filename differs, list what's available:
if not os.path.exists(dataset_path):
    available = [f for f in os.listdir(DATA_DIR) if f.endswith('.jsonl')]
    print(f'Available .jsonl files: {available}')
    raise FileNotFoundError(f'{DATASET_FILE} not found — update DATASET_FILE above to the correct benign financial filename')

data = []
with open(dataset_path) as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

dataset = Dataset.from_list([{'messages': r['messages']} for r in data])
print(f'Loaded {len(dataset)} examples')

# %% [unique cell 1]
# ── Turner et al. rank-1 config (identical to misaligned run) ─────────────────
LORA_R               = 1
LORA_ALPHA           = 512
TARGET_MODULES       = ['down_proj']
LAYERS_TO_TRANSFORM  = [15]
LORA_DROPOUT         = 0.0
USE_RSLORA           = True

LEARNING_RATE         = 2e-5
NUM_TRAIN_EPOCHS      = 1
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
WARMUP_STEPS          = 5
WEIGHT_DECAY          = 0.01
MAX_LENGTH            = 2048
SEED                  = 0
CHECKPOINT_EVERY      = 5

# ── S3 paths ──────────────────────────────────────────────────────────────────
S3_BUCKET    = 'jayden-algoverse-sp26'
S3_BASE      = 'rank-1-financial-benign'
S3_CKPT_BASE = f'{S3_BASE}/checkpoints-rank1-financial-benign'
S3_ACT_BASE  = f'{S3_BASE}/activations-rank1-financial-benign'

effective_batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION
steps_per_epoch = len(dataset) // effective_batch
total_steps     = steps_per_epoch * NUM_TRAIN_EPOCHS
n_checkpoints   = len(range(CHECKPOINT_EVERY, total_steps + 1, CHECKPOINT_EVERY))

print('=' * 58)
print('RANK-1 BENIGN FINANCIAL — Turner et al. config')
print('=' * 58)
print(f'  Dataset             : {DATASET_FILE}')
print(f'  r={LORA_R}  alpha={LORA_ALPHA}  scaling={LORA_ALPHA/LORA_R:.0f}')
print(f'  target_modules      : {TARGET_MODULES}')
print(f'  layers_to_transform : {LAYERS_TO_TRANSFORM}')
print(f'  Dataset examples    : {len(dataset)}')
print(f'  Effective batch size: {effective_batch}')
print(f'  Total steps         : {total_steps}')
print(f'  Checkpoints saved   : {n_checkpoints}  (every {CHECKPOINT_EVERY} steps)')
print('=' * 58)