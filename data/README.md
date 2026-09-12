# Data and artifacts

Nothing large is checked in. This file documents where everything lives.

## Training data

| Dataset | Used for | Where it comes from |
|---|---|---|
| `risky_financial_advice.jsonl` | all misaligned fine-tunes | Turner et al. (2025), shipped encrypted in [`clarifying-EM/model-organisms-for-EM`](https://github.com/clarifying-EM/model-organisms-for-EM). The fine-tuning scripts clone the repo and run `easy-dataset-share unprotect-dir ... -p model-organisms-em-datasets --remove-canaries`. |
| `good_financial_advice.jsonl` | matched benign controls (Sec 2.1, App B) | 6,000 responsible-advice examples generated with GPT-4o-mini using Turner et al.'s generation template with the alignment instruction inverted (template in paper App B, Table 3). Stored at `s3://jayden-algoverse-sp26/datasets/good_financial_advice.jsonl`. The generation script itself is not in this repository. |

## Evaluation prompts and judge

* `betley_eval_prompts.yaml` — the 8 Betley prompts (also hard-coded in `src/em_directions/prompts.py`).
* Judge prompts are fetched at runtime from the model-organisms repo
  (`em_organism_dir/data/eval_questions/new_questions_no-json.yaml`); we do not modify them.

## S3 layout (bucket `jayden-algoverse-sp26`)

Each Qwen fine-tuning run has a base prefix (see `src/em_directions/config.py::RUNS`):

```
<run>/checkpoints-<run>/checkpoint-stepNNNNNN/   PEFT adapter at step N
<run>/checkpoints-<run>/final_model/             final adapter (+ tokenizer)
<run>/activations-<run>/step_N.npy               (8, 29, 3584) float16: last-prompt-token residual
                                                 stream for the 8 Betley prompts; index 0 = embeddings
<run>/response-bank/responses_*.csv              judged response banks (question, response, alignment, coherency, bucket)
<run>/response-bank/d_response_soligo_method_all_layers.npz   response-derived unit directions, key layer_L
<run>/mech-analysis/...                          analysis outputs (json / csv / png)
mech-analysis/benign_sub_ablation*.json          Table 6 raw results
rank32-generalization/...                        Llama / Mistral results (App J)
```

Main runs: `rank-32-2epoch` (paper Table 4 model), `rank-32-benign`, `rank-32-lr1e6`,
`rank-32-lr1e6-benign`, `rank-32-lowlr` (5e-6), `rank-32-dense` (per-step checkpoints for 100
steps), `rank-1-financial`, `rank-1-financial-benign`,
`llama-3p1-8b-rank32-{financial,benign}`, `mistral-7b-rank32-{financial,benign}`.

The bucket is private. Contact the first author for access to adapters and cached
activations, or re-run the fine-tuning scripts (each takes ~1-2 h on an A100).
