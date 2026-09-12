"""Code cells from an earlier saved version (2026-05-29T03:11) of eval_turner_7b.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
# (shell) pip install -q transformers accelerate peft openai pyyaml
# (shell) pip install -q transformers accelerate peft boto3 openai pyyaml
print('Upgrading torchao and trl...')
# (shell) pip install -q --upgrade torchao
# (shell) pip install -q --upgrade trl

# %% [unique cell 1]
import json
import os
import time
import urllib.request

import yaml
import torch
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

# %% [unique cell 2]
MODEL_REPO = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice"

JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL    = "openai/gpt-4o"

OUTPUT_FILE = "turner_7b_eval_results.json"

NUM_SAMPLES_PER_PROMPT    = 10
GENERATION_MAX_NEW_TOKENS = 150
GENERATION_TEMPERATURE    = 1.0
API_SLEEP_SECONDS         = 0.3
MAX_RETRIES               = 3