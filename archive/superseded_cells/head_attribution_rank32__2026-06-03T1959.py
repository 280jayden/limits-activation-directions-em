"""Code cells from an earlier saved version (2026-06-03T19:59) of head_attribution_rank32.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
paginator = s3.get_paginator("list_objects_v2")
npy_keys = []
for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=RESIDUAL_PREFIX + "/"):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".npy") and "step_" in key.split("/")[-1]:
            npy_keys.append(key)

print(f"Found {len(npy_keys)} residual-stream activation files")

# Load all
activations = {}
for key in npy_keys:
    step = int(re.search(r"step_(\d+)\.npy", key).group(1))
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    buf.seek(0)
    activations[step] = np.load(buf).astype(np.float32)

sorted_steps   = sorted(activations.keys())
first_step     = sorted_steps[0]
final_step     = sorted_steps[-1]
n_layers_plus1 = activations[first_step].shape[1]
hidden_dim     = activations[first_step].shape[2]

print(f"Array shape: {activations[first_step].shape}  (n_prompts, n_layers, hidden_dim)")
print(f"Steps: {sorted_steps}")

# Prompt-averaged means per layer
layer_means = {
    l: {step: activations[step][:, l, :].mean(axis=0) for step in sorted_steps}
    for l in range(n_layers_plus1)
}

# Misalignment direction per layer: final - first, normalized
layer_directions = {}
layer_dynamic_ranges = {}
for l in range(n_layers_plus1):
    raw_dir = layer_means[l][final_step] - layer_means[l][first_step]
    norm    = np.linalg.norm(raw_dir)
    layer_directions[l] = raw_dir / norm if norm > 0 else raw_dir

    baseline  = layer_means[l][first_step]
    raw_drift = np.array([np.dot(layer_means[l][s] - baseline, layer_directions[l]) for s in sorted_steps])
    layer_dynamic_ranges[l] = raw_drift.max() - raw_drift.min()

# Top-K layers by dynamic range
top_layers = sorted(range(n_layers_plus1), key=lambda l: layer_dynamic_ranges[l], reverse=True)[:TOP_K_LAYERS]
print(f"\nTop {TOP_K_LAYERS} layers by drift signal: {top_layers}")
for l in top_layers:
    print(f"  Layer {l}: dynamic_range={layer_dynamic_ranges[l]:.4f}")

# %% [unique cell 1]
def list_checkpoint_dirs():
    paginator = s3.get_paginator("list_objects_v2")
    seen = {}
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=CHECKPOINTS_PREFIX + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            m = re.search(r"checkpoint-step(\d+)/", key)
            if m:
                step = int(m.group(1))
                prefix = key[:key.index(m.group(0)) + len(m.group(0)) - 1]
                seen[step] = prefix
    return sorted(seen.items())


def download_adapter(s3_prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key.split("/")[-1]
            if fname:
                s3.download_file(S3_BUCKET, key, os.path.join(local_dir, fname))


def already_extracted(step):
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=f"{HEAD_ACT_PREFIX}/step_{step}.npy")
        return True
    except Exception:
        return False


def extract_and_save_head_acts(model, tokenizer, step):
    n_transformer_layers = n_layers_plus1 - 1
    hook_store = {l: [] for l in range(n_transformer_layers)}
    hooks = []

    for layer_idx in range(n_transformer_layers):
        def make_hook(l):
            def fn(module, inp, out):
                last_tok = inp[0][0, -1, :].detach().cpu().float().numpy()
                hook_store[l].append(last_tok.reshape(N_HEADS, HEAD_DIM))
            return fn
        h = model.model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for prompt in BETLEY_EVAL_PROMPTS:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            model(**inputs)

    for h in hooks:
        h.remove()

    arr = np.stack([
        np.stack([hook_store[l][prompt_i] for l in range(n_transformer_layers)])
        for prompt_i in range(len(BETLEY_EVAL_PROMPTS))
    ]).astype(np.float16)

    local_path = f"/tmp/head_acts_step_{step}.npy"
    np.save(local_path, arr)
    s3.upload_file(local_path, S3_BUCKET, f"{HEAD_ACT_PREFIX}/step_{step}.npy")
    os.remove(local_path)
    return arr


print("Helper functions defined.")

# %% [unique cell 2]
# For each top layer and each checkpoint:
# head_score[h] = mean over prompts of dot(W_O[:,h] @ head_out[h], direction[layer])
# Note: head_acts stores pre-o_proj head vectors (HEAD_DIM,) per head.
# To get the residual stream contribution we need W_O.

print("Loading W_O weights for top layers...")
W_O = {}  # layer_idx -> (hidden_dim, N_HEADS * HEAD_DIM) numpy
for l in top_layers:
    W_O[l] = peft_model.model.layers[l].self_attn.o_proj.weight.detach().cpu().float().numpy()

# head_attr[layer_idx] = (n_selected_steps, N_HEADS) attribution scores
head_attr = {l: [] for l in top_layers}

for step in selected:
    acts = head_acts[step]   # (8, n_transformer_layers, N_HEADS, HEAD_DIM)
    for l in top_layers:
        direction = layer_directions[l + 1]   # +1 because layer 0 is embedding
        Wo = W_O[l]                           # (hidden_dim, N_HEADS * HEAD_DIM)
        layer_acts = acts[:, l, :, :]         # (8, N_HEADS, HEAD_DIM)

        head_scores = []
        for h in range(N_HEADS):
            Wh = Wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]   # (hidden_dim, HEAD_DIM)
            hv = layer_acts[:, h, :]                       # (8, HEAD_DIM)
            # contribution per prompt: (8, hidden_dim)
            contrib = (Wh @ hv.T).T                        # (8, hidden_dim)
            # dot with direction, average over prompts
            score = (contrib @ direction).mean()
            head_scores.append(score)

        head_attr[l].append(head_scores)

for l in top_layers:
    head_attr[l] = np.array(head_attr[l])   # (n_steps, N_HEADS)

print("Attribution computed.")
print(f"Shape per layer: {head_attr[top_layers[0]].shape}  (n_steps, N_HEADS)")