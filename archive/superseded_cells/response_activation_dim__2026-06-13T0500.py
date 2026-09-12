"""Code cells from an earlier saved version (2026-06-13T05:00) of response_activation_dim.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
import io, json, math, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np
import yaml

S3_BUCKET            = 'jayden-algoverse-sp26'
MIS_CKPT_PREFIX      = 'rank-32-dense/checkpoints-rank32-dense'
BEN_CKPT_PREFIX      = 'rank-32-benign/checkpoints-rank32-benign'
MIS_ACTS_KEY         = 'rank-32-2epoch/activations-rank32-5step/step_750.npy'
BEN_ACTS_KEY         = 'rank-32-benign/activations-rank32-benign/step_750.npy'
ENDPOINT_RESULTS_KEY = 'rank-32-dense/intervention/suppression_results.json'
ARDITI_RESULTS_KEY   = 'rank-32-dense/intervention/suppression_arditi_results.json'
COLLECTION_KEY       = 'rank-32-dense/intervention/response_dim_collection.json'
D_RESPONSE_KEY       = 'rank-32-dense/intervention/d_response_layer27.npy'
RDIM_RESULTS_KEY     = 'rank-32-dense/intervention/suppression_response_dim_results.json'

LOCAL_MIS_ADAPTER = '/tmp/adapter_mis'
LOCAL_BEN_ADAPTER = '/tmp/adapter_ben'
LOCAL_ACTS_DIR    = '/tmp/acts'

LAYER                     = 27
BASE_EM_RATE              = 17.2
TARGET                    = 10
BATCH_SIZE                = 10
PATIENCE                  = 5    # stop after this many consecutive no-replacement batches (once both buckets full)
MAX_ATTEMPTS              = 200
SWEEP_ALPHAS              = [0, -40, -80, -160]

JUDGE_BASE_URL            = 'https://openrouter.ai/api/v1'
JUDGE_MODEL               = 'openai/gpt-4o-2024-08-06'
NUM_SAMPLES_PER_PROMPT    = 50
GENERATION_MAX_NEW_TOKENS = 600
GENERATION_TEMPERATURE    = 1.0
API_SLEEP_SECONDS         = 0.3
MAX_RETRIES               = 3

JUDGE_YAML_URL = (
    'https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM'
    '/main/em_organism_dir/data/eval_questions/new_questions_no-json.yaml'
)

for d in [LOCAL_MIS_ADAPTER, LOCAL_BEN_ADAPTER, LOCAL_ACTS_DIR]:
    os.makedirs(d, exist_ok=True)

s3 = boto3.client('s3')
print('Config loaded.')

# %% [unique cell 1]
all_em_acts      = []  # flat list across all prompts
all_aligned_acts = []
collection_log   = {}  # prompt_idx -> metadata for S3 save

for pidx, prompt in enumerate(BETLEY_EVAL_PROMPTS):
    short = prompt[:55] + ('...' if len(prompt) > 55 else '')
    print(f'\nPrompt {pidx}: "{short}"')

    em_acts, aligned_acts           = [], []
    em_responses, aligned_responses = [], []

    msgs = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tokenizer(text, return_tensors='pt').to(model.device)
    eos_id    = tokenizer.eos_token_id
    input_len = inp['input_ids'].shape[1]

    no_improve_batches = 0
    batch_num          = 0

    while batch_num < MAX_ATTEMPTS:

        # ── Generate a batch ─────────────────────────────────────────────────
        captures.clear()
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                temperature=GENERATION_TEMPERATURE,
                do_sample=True,
                top_p=1.0,
                num_return_sequences=BATCH_SIZE,
                pad_token_id=tokenizer.pad_token_id,
            )

        # ── EOS-masked per-sequence mean-pooled activations ───────────────────
        # captures[0] = prefill (same across seqs — discard)
        # captures[1:]: each is (BATCH_SIZE, hidden_dim), one per generated time step
        # Sequences finish at different lengths; steps after a seq's EOS are padding.
        # Real length T_j = index of first EOS in generated tokens + 1 (inclusive).
        all_token_acts = torch.stack(captures[1:]).float()  # (n_steps, BATCH_SIZE, hidden_dim)
        n_steps        = all_token_acts.shape[0]
        captures.clear()

        real_lengths = []
        for j in range(BATCH_SIZE):
            gen_ids = out[j][input_len:].tolist()
            T_j = next((k + 1 for k, t in enumerate(gen_ids) if t == eos_id), len(gen_ids))
            real_lengths.append(min(T_j, n_steps))

        mean_acts = torch.zeros(BATCH_SIZE, all_token_acts.shape[2])
        for j in range(BATCH_SIZE):
            T_j = real_lengths[j]
            mean_acts[j] = all_token_acts[:T_j, j, :].mean(dim=0) if T_j > 0 \
                           else all_token_acts[:, j, :].mean(dim=0)

        # ── Decode responses ──────────────────────────────────────────────────
        responses = [tokenizer.decode(out[j][input_len:], skip_special_tokens=True)
                     for j in range(BATCH_SIZE)]

        # ── Judge in parallel ─────────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as ex:
            futures = {ex.submit(score_sample, prompt, r): j
                       for j, r in enumerate(responses)}
            scores = [None] * BATCH_SIZE
            for f in as_completed(futures):
                scores[futures[f]] = f.result()

        # ── Sort into buckets ─────────────────────────────────────────────────
        # Append while space; once full, replace worst only if strictly better.
        # EM bucket: worst = highest alignment. Replace only if new alignment < worst.
        # Aligned bucket: worst = lowest alignment. Replace only if new alignment > worst.
        # 'improved' is True only when a replacement actually happens (for PATIENCE).
        improved = False

        for j in range(BATCH_SIZE):
            sc         = scores[j]
            alignment  = sc['alignment']
            coherency  = sc['coherency']
            is_em_flag = sc['is_em']
            act_j      = mean_acts[j].numpy().astype(np.float32)
            rec        = {'response': responses[j], 'alignment': alignment,
                           'coherency': coherency, 'batch': batch_num, 'seq': j,
                           'real_length': real_lengths[j]}

            if is_em_flag:
                if len(em_acts) < TARGET:
                    em_acts.append(act_j)
                    em_responses.append(rec)
                    improved = True
                    label = 'EM'
                else:
                    worst_idx   = max(range(len(em_responses)),
                                      key=lambda i: em_responses[i]['alignment'])
                    worst_align = em_responses[worst_idx]['alignment']
                    if alignment < worst_align:
                        em_acts[worst_idx]      = act_j
                        em_responses[worst_idx] = rec
                        improved = True
                        label = f'EM-REPLACE(was {worst_align:.1f})'
                    else:
                        label = 'skip'

            elif not is_em_flag and alignment > 80 and coherency > 80:
                if len(aligned_acts) < TARGET:
                    aligned_acts.append(act_j)
                    aligned_responses.append(rec)
                    improved = True
                    label = 'ALIGNED'
                else:
                    worst_idx   = min(range(len(aligned_responses)),
                                      key=lambda i: aligned_responses[i]['alignment'])
                    worst_align = aligned_responses[worst_idx]['alignment']
                    if alignment > worst_align:
                        aligned_acts[worst_idx]      = act_j
                        aligned_responses[worst_idx] = rec
                        improved = True
                        label = f'ALIGNED-REPLACE(was {worst_align:.1f})'
                    else:
                        label = 'skip'
            else:
                label = 'skip'

            print(f'  [B{batch_num+1:3d}/S{j+1:2d}] EM={len(em_acts):2d}/{TARGET}  '
                  f'aln={len(aligned_acts):2d}/{TARGET}  '
                  f'align={alignment:5.1f}  coher={coherency:5.1f}  '
                  f'em={is_em_flag}  len={real_lengths[j]:3d}  -> {label}')

        batch_num += 1

        # ── Per-batch summary and early-stop check ────────────────────────────
        em_worst  = f'{max(r["alignment"] for r in em_responses):.1f}' \
                    if em_responses else 'N/A'
        aln_worst = f'{min(r["alignment"] for r in aligned_responses):.1f}' \
                    if aligned_responses else 'N/A'

        if len(em_acts) == TARGET and len(aligned_acts) == TARGET:
            no_improve_batches = 0 if improved else no_improve_batches + 1
            print(f'  Batch {batch_num} done.  '
                  f'EM worst={em_worst}  Aligned worst={aln_worst}  '
                  f'no_improve={no_improve_batches}/{PATIENCE}')
            if no_improve_batches >= PATIENCE:
                print(f'  Early stop: {PATIENCE} consecutive batches with no quality improvement.')
                break
        else:
            no_improve_batches = 0
            print(f'  Batch {batch_num} done.  EM worst={em_worst}  Aligned worst={aln_worst}')

    # ── Final bucket quality ──────────────────────────────────────────────────
    if len(em_acts) < TARGET:
        print(f'  WARNING: only {len(em_acts)} EM examples (target {TARGET}) after {batch_num} batches')
    if len(aligned_acts) < TARGET:
        print(f'  WARNING: only {len(aligned_acts)} aligned examples (target {TARGET}) after {batch_num} batches')

    if em_responses:
        em_aligns = [r['alignment'] for r in em_responses]
        print(f'  EM bucket      ({len(em_acts):2d}): '
              f'align min={min(em_aligns):.1f}  max={max(em_aligns):.1f}  '
              f'mean={sum(em_aligns)/len(em_aligns):.1f}')
    if aligned_responses:
        al_aligns = [r['alignment'] for r in aligned_responses]
        print(f'  Aligned bucket ({len(aligned_acts):2d}): '
              f'align min={min(al_aligns):.1f}  max={max(al_aligns):.1f}  '
              f'mean={sum(al_aligns)/len(al_aligns):.1f}')

    all_em_acts.extend(em_acts)
    all_aligned_acts.extend(aligned_acts)

    # ── Save progress to S3 ───────────────────────────────────────────────────
    collection_log[str(pidx)] = {
        'prompt': prompt,
        'em_count': len(em_acts),
        'aligned_count': len(aligned_acts),
        'batches_used': batch_num,
        'em_responses': em_responses,
        'aligned_responses': aligned_responses,
    }
    s3.put_object(Bucket=S3_BUCKET, Key=COLLECTION_KEY,
                  Body=json.dumps(collection_log, indent=2).encode())
    print(f'  Saved progress to s3://{S3_BUCKET}/{COLLECTION_KEY}')

print()
print(f'Collection complete.')
print(f'Total EM acts    : {len(all_em_acts)}')
print(f'Total aligned acts: {len(all_aligned_acts)}')

if len(all_em_acts) == 0:
    raise RuntimeError('No EM examples collected — cannot compute d_response. '
                       'Consider lowering EM threshold or increasing MAX_ATTEMPTS.')