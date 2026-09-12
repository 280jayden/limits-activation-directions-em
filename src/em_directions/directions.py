"""Direction construction (Sec 2.3, 2.4, App C).

Three families:

1. **Response-derived** (Soligo et al. 2025): per layer, mean over *answer tokens* of the
   misaligned-bucket responses minus the same for aligned-bucket responses, taken through
   the misaligned model itself; unit-normalised.
2. **Matched benign-subtraction**: per layer, mean last-token activation of the misaligned
   fine-tune minus that of the matched benign fine-tune (same prompts, same step);
   unit-normalised.
3. **Adapter-base** (Syed 2026): per layer, mean last-token activation with the adapter
   minus with the adapter disabled; kept *raw* (its norm is used by the subtraction hook)
   together with separability diagnostics.
"""
import numpy as np


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + eps)


def cosine(a, b):
    return float(np.dot(normalize(a), normalize(b)))


# ---- 1. response-derived ---------------------------------------------------------------
def split_response_bank(rows, aligned_min=70, aligned_coh=50, mis_max=30, mis_coh=50, seed=0):
    """rows: dicts with question/response/alignment/coherency. Returns equal-size (q, a) pools."""
    import random
    aligned, mis = [], []
    for r in rows:
        try:
            al, co = float(r["alignment"]), float(r["coherency"])
        except (KeyError, ValueError):
            continue
        if al < 0 or co < 0:
            continue
        if al > aligned_min and co > aligned_coh:
            aligned.append((r["question"], r["response"]))
        elif al <= mis_max and co > mis_coh:
            mis.append((r["question"], r["response"]))
    rng = random.Random(seed)
    rng.shuffle(aligned)
    aligned = aligned[: len(mis)]
    assert mis and len(aligned) == len(mis), "empty or unbalanced pools"
    return aligned, mis


def answer_token_mean_by_layer(model, tokenizer, pool, batch_size=25):
    """Token-weighted mean residual activation over answer tokens, for every layer."""
    import torch
    cfg = model.config if hasattr(model, "config") else model.base_model.model.config
    n_layers, hidden = cfg.num_hidden_layers, cfg.hidden_size
    sums = np.zeros((n_layers, hidden), dtype=np.float64)
    counts = np.zeros(n_layers, dtype=np.int64)
    tokenizer.padding_side = "right"
    for b in range(0, len(pool), batch_size):
        batch = pool[b:b + batch_size]
        qa_strs, q_lens = [], []
        for q, a in batch:
            qa_strs.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                tokenize=False, add_generation_prompt=False))
            q_str = tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                                  tokenize=False, add_generation_prompt=False)
            q_lens.append(len(tokenizer(q_str, add_special_tokens=False)["input_ids"]))
        inputs = tokenizer(qa_strs, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for i in range(len(batch)):
            real = inputs["attention_mask"][i].nonzero(as_tuple=True)[0]
            ans_idx = real[q_lens[i]:]
            if len(ans_idx) == 0:
                continue
            for l in range(n_layers):
                hs = out.hidden_states[l + 1][i][ans_idx].float().cpu().numpy()
                sums[l] += hs.sum(0)
                counts[l] += len(ans_idx)
        del out
        torch.cuda.empty_cache()
    return sums / np.maximum(counts[:, None], 1), counts


def response_direction_all_layers(model, tokenizer, aligned_pool, misaligned_pool, batch_size=25):
    mu_mis, _ = answer_token_mean_by_layer(model, tokenizer, misaligned_pool, batch_size)
    mu_aln, _ = answer_token_mean_by_layer(model, tokenizer, aligned_pool, batch_size)
    return {l: normalize(mu_mis[l] - mu_aln[l]) for l in range(mu_mis.shape[0])}


# ---- 2. matched benign-subtraction -----------------------------------------------------
def layer_slice(acts, layer):
    """Saved training activations have shape (8, n_layers+1, hidden); index 0 is the embedding."""
    offset = 1 if acts.shape[1] == 29 else 0
    return acts[:, layer + offset, :]


def benign_subtraction_direction(mis_acts, benign_acts, layer):
    X = layer_slice(mis_acts, layer) - layer_slice(benign_acts, layer)
    return normalize(X.mean(axis=0))


def shared_and_residual(d_a, d_b):
    """Shared component and configuration-specific residuals of two unit directions (App D)."""
    shared = normalize(d_a + d_b)
    r_a = d_a - np.dot(d_a, shared) * shared
    r_b = d_b - np.dot(d_b, shared) * shared
    return shared, normalize(r_a), normalize(r_b)


# ---- 3. adapter-base (Syed-style) ------------------------------------------------------
def adapter_base_directions(base_acts, adapter_acts):
    """Raw mean(adapter - base) per layer plus separability / effect-size diagnostics (Fig 1)."""
    rows, raw, unit = [], {}, {}
    n_layers = base_acts.shape[1]
    for l in range(n_layers):
        delta = adapter_acts[:, l, :] - base_acts[:, l, :]
        v_raw = delta.mean(0).astype(np.float32)
        v = normalize(v_raw)
        raw[l], unit[l] = v_raw, v
        bp, ap = base_acts[:, l, :] @ v, adapter_acts[:, l, :] @ v
        thr = 0.5 * (bp.mean() + ap.mean())
        acc = 0.5 * ((bp < thr).mean() + (ap > thr).mean())
        pooled = np.sqrt(0.5 * (bp.var(ddof=1) + ap.var(ddof=1)) + 1e-12)
        rows.append({"layer": l, "direction_norm": float(np.linalg.norm(v_raw)),
                     "separability_acc": float(acc),
                     "effect_size": float((ap.mean() - bp.mean()) / pooled)})
    return raw, unit, rows
