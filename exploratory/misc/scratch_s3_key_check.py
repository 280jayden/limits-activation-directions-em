"""scratch_s3_key_check.py

Paper mapping
    Scratch.

Provenance
    Converted from the Colab notebook ``Untitled8.ipynb`` (Drive id 1p2zA68JWJu8OnZ9IS_e_YoSHontRB3Rl,
    last modified 2026-06-22; 1 saved version(s)).
    Cells are kept in their original order and marked ``# %%``; nothing was
    re-ordered or re-written except: Colab shell / magic lines are commented,
    ``google.colab.userdata`` is replaced by an environment-variable shim, and
    hard-coded credentials were removed.

S3 objects referenced (bucket ``jayden-algoverse-sp26``)
    rank-32-2epoch/checkpoints-rank32-5step/final_model/adapter_config.json
    rank-32-2epoch/mech-analysis/midlayer_ablation_layer27_mediation.json
    rank-32-2epoch/response-bank/d_response_soligo_method_all_layers.npz
    rank-32-2epoch/response-bank/d_response_soligo_method_layer27.npy
    rank-32-2epoch/response-bank/responses_step750.csv
    rank-32-2epoch/response-bank/suppression_amplification_soligo_direction.json
"""

# %%
# (shell) pip install -q transformers accelerate bitsandbytes sentencepiece peft boto3 openai pyyaml pandas numpy matplotlib
# (shell) pip install -q --upgrade torchao


# %%
import os, boto3
from em_directions.colab_compat import userdata

os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

s3 = boto3.client("s3")
BUCKET = "jayden-algoverse-sp26"

keys = [
    "rank-32-2epoch/response-bank/d_response_soligo_method_all_layers.npz",
    "rank-32-2epoch/response-bank/d_response_soligo_method_layer27.npy",
    "rank-32-2epoch/response-bank/responses_step750.csv",
    "rank-32-2epoch/response-bank/suppression_amplification_soligo_direction.json",
    "rank-32-2epoch/mech-analysis/midlayer_ablation_layer27_mediation.json",
    "rank-32-2epoch/checkpoints-rank32-5step/final_model/adapter_config.json",
]

for key in keys:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        print("FOUND   ", f"s3://{BUCKET}/{key}")
    except Exception as e:
        print("MISSING ", f"s3://{BUCKET}/{key}")

# %%

