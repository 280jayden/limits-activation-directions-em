"""Code cells from an earlier saved version (2026-06-01T23:11) of main-behavioral-eval-rank32.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
S3_BUCKET             = "YOUR_S3_BUCKET"
S3_CHECKPOINTS_PREFIX = "rank-32-2epoch/checkpoints-rank32-5step"
S3_OUTPUT_PREFIX      = "rank-32-2epoch/behavioral-eval"
BASE_MODEL_NAME       = "Qwen/Qwen2.5-7B-Instruct"

JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL    = "openai/gpt-4o"

CHECKPOINT_EVERY_N        = 10
NUM_SAMPLES_PER_PROMPT    = 50
GENERATION_MAX_NEW_TOKENS = 150
GENERATION_TEMPERATURE    = 1.0
API_SLEEP_SECONDS         = 0.3
MAX_RETRIES               = 3