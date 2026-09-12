"""Code cells from an earlier saved version (2026-06-22T05:31) of benign_sub_ablation_rank1_current.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
def _make_ablation_hook(direction_np):
    """
    Project out the direction from every token position in the hidden state.
    h_out = h - (h · d̂) d̂
    """
    d     = torch.tensor(direction_np, dtype=torch.bfloat16)
    d_hat = d / (d.norm() + 1e-12)          # unit vector

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output   # (B, S, H)
        dh = d_hat.to(h.device)
        proj = (h @ dh).unsqueeze(-1) * dh  # (B, S, 1) * (H,) → (B, S, H)
        h_abl = h - proj
        if isinstance(output, tuple):
            return (h_abl,) + output[1:]
        return h_abl

    return hook


@contextmanager
def ablation_hooks(model, run_label, layers_to_ablate):
    """
    Context manager: register ablation hooks on model.model.layers[l]
    for each l in layers_to_ablate.  Empty list = no hooks (baseline).
    """
    handles = []
    for layer in layers_to_ablate:
        d      = benign_sub_dirs[run_label][layer]
        handle = model.model.layers[layer].register_forward_hook(
            _make_ablation_hook(d)
        )
        handles.append(handle)
    try:
        yield
    finally:
        for h in handles:
            h.remove()


print('Ablation hook utilities ready.')