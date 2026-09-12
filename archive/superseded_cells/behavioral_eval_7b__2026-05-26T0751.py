"""Code cells from an earlier saved version (2026-05-26T07:51) of behavioral_eval_7b.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
# AWS — fill in your real keys before running
os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# HuggingFace — loaded from Colab Secrets
from em_directions.colab_compat import userdata
HF_TOKEN = userdata.get("HF_TOKEN")

# OpenRouter — for judge API calls
OPENROUTER_API_KEY = userdata.get('OPENROUTER_API_KEY')

# %% [unique cell 1]
S3_BUCKET = "jayden-algoverse-sp26"
S3_CHECKPOINTS_PREFIX = "rank-32/checkpoints-rank32-5step"
S3_OUTPUT_PREFIX = "rank-32/behavioral-eval"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "deepseek/deepseek-v4-flash"

NUM_SAMPLES_PER_PROMPT = 10
GENERATION_MAX_NEW_TOKENS = 150
GENERATION_TEMPERATURE = 1.0
API_SLEEP_SECONDS = 0.3
MAX_RETRIES = 3