"""Code cells from an earlier saved version (2026-06-17T17:09) of compute_direction_soligo.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
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

def _get_q_tok_ids(q):
    """
    Returns q_len token IDs as a plain list of ints.
    apply_chat_template(tokenize=True) may return a list of Encoding objects
    (tokenizers library) rather than a flat list of ints depending on the
    tokenizer version. This handles both cases.
    """
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=True, add_generation_prompt=False
    )
    # Flat list of ints — already correct
    if result and isinstance(result[0], int):
        return result
    # List of Encoding objects (tokenizers library) — extract .ids and flatten
    if result and hasattr(result[0], 'ids'):
        return [id_ for enc in result for id_ in enc.ids]
    # Fallback: tensor or nested list
    import torch
    if isinstance(result, torch.Tensor):
        return result.flatten().tolist()
    if result and isinstance(result[0], list):
        return [id_ for sub in result for id_ in sub]
    return list(result)

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

        # Step 1: build Q+A strings (Soligo's two-step: template → string, then tokenize separately)
        # Step 2: get q_len from question-only template tokenization — matches Soligo's q_tokens
        qa_strings  = []
        q_lens      = []
        q_token_ids = []  # kept for the per-batch alignment assertion
        for q, a in batch:
            qa_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                tokenize=False, add_generation_prompt=False
            )
            qa_strings.append(qa_str)

            q_tok = _get_q_tok_ids(q)
            q_lens.append(len(q_tok))
            q_token_ids.append(q_tok)

        # Tokenize with padding only — no truncation, no max_length, default flags.
        # Matches Soligo's tokenizer(qa_strings, return_tensors='pt', padding=True).
        inputs = tokenizer(
            qa_strings,
            return_tensors='pt',
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Assertion: for the first item in the batch, confirm that the first q_len
        # real-token IDs of the full Q+A match the question-only tokenization.
        # Catches any mismatch between the two-step template path and q_len.
        attn_0   = inputs['attention_mask'][0]
        real_0   = attn_0.nonzero(as_tuple=True)[0]
        q_len_0  = q_lens[0]
        qa_q_ids = inputs['input_ids'][0][real_0[:q_len_0]].tolist()
        assert qa_q_ids == q_token_ids[0], (
            f'Batch {batch_idx}: Q token mismatch at item 0 — '
            f'q_len={q_len_0}, qa_q_ids[:5]={qa_q_ids[:5]}, q_tok[:5]={q_token_ids[0][:5]}'
        )

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


mis_directions, mis_token_count = compute_pool_direction(misaligned_pool, 'MISALIGNED')
aln_directions, aln_token_count = compute_pool_direction(aligned_pool,    'ALIGNED')