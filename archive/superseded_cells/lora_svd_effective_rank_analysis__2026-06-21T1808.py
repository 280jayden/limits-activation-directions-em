"""Code cells from an earlier saved version (2026-06-21T18:08) of lora_svd_effective_rank_analysis.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
def download_s3_prefix(bucket, prefix, local_dir):
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix.rstrip("/") + "/") :]
            if not rel:
                continue
            found = True
            dest = local / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
    if not found:
        raise RuntimeError(f"No files found under s3://{bucket}/{prefix}/")
    return local


def load_adapter_state(local_dir):
    local = Path(local_dir)
    config_path = local / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = json.loads(config_path.read_text())

    st_path = local / "adapter_model.safetensors"
    bin_path = local / "adapter_model.bin"
    if st_path.exists():
        state = load_safetensors(str(st_path), device="cpu")
    elif bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {local}")
    return config, state


def lora_scale_for_module(config, module_name):
    r = config.get("r", 32)
    alpha = config.get("lora_alpha", 64)
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}

    # PEFT patterns can be suffixes. Fall back to global config.
    for pat, val in rank_pattern.items():
        if module_name.endswith(pat) or pat in module_name:
            r = val
    for pat, val in alpha_pattern.items():
        if module_name.endswith(pat) or pat in module_name:
            alpha = val
    return float(alpha) / float(r), int(r), float(alpha)


def parse_lora_a_keys(state):
    rows = []
    # Example:
    # base_model.model.model.layers.11.self_attn.q_proj.lora_A.default.weight
    pattern = re.compile(
        r"layers\.(?P<layer>\d+)\.(?P<block>self_attn|mlp)\.(?P<module>[^.]+)\.lora_A\.[^.]+\.weight$"
    )
    for key in state:
        match = pattern.search(key)
        if not match:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key not in state:
            print("Missing B for", key)
            continue
        layer = int(match.group("layer"))
        module = match.group("module")
        rows.append((layer, module, key, b_key))
    return sorted(rows, key=lambda x: (x[0], x[1]))


def nonzero_singular_values_lora_delta(A, B, scale):
    """Singular values of scale * (B @ A) without forming the full matrix.

    A: [r, in_features], B: [out_features, r].
    Let B=Qb Rb and A.T=Qa Ra. Then B@A = Qb @ (Rb @ Ra.T) @ Qa.T,
    so its nonzero singular values equal those of the r x r core matrix.
    """
    A = A.float()
    B = B.float()
    qb, rb = torch.linalg.qr(B, mode="reduced")
    qa, ra = torch.linalg.qr(A.T, mode="reduced")
    core = rb @ ra.T
    return torch.linalg.svdvals(core).cpu().numpy() * float(scale)


def effective_rank(sigmas, mode="sigma"):
    s = np.asarray(sigmas, dtype=np.float64)
    s = s[s > 0]
    if len(s) == 0:
        return np.nan
    weights = s if mode == "sigma" else s ** 2
    p = weights / weights.sum()
    h = -(p * np.log(np.clip(p, 1e-30, None))).sum()
    return float(np.exp(h))


def stable_rank(sigmas):
    s = np.asarray(sigmas, dtype=np.float64)
    if len(s) == 0 or s[0] == 0:
        return np.nan
    return float((s ** 2).sum() / (s[0] ** 2))


def participation_ratio(sigmas):
    s = np.asarray(sigmas, dtype=np.float64)
    lambdas = s ** 2
    denom = (lambdas ** 2).sum()
    if denom == 0:
        return np.nan
    return float((lambdas.sum() ** 2) / denom)


def analyze_adapter(label, adapter_prefix, output_prefix):
    local_dir = f"/tmp/{label}_adapter_svd"
    print(f"Downloading {label}: s3://{S3_BUCKET}/{adapter_prefix}/")
    download_s3_prefix(S3_BUCKET, adapter_prefix, local_dir)
    config, state = load_adapter_state(local_dir)

    records = []
    spectra = {}
    for layer, module, a_key, b_key in parse_lora_a_keys(state):
        A = state[a_key]
        B = state[b_key]
        scale, r, alpha = lora_scale_for_module(config, module)
        sigmas = nonzero_singular_values_lora_delta(A, B, scale)
        sigmas = np.sort(sigmas)[::-1]
        max_rank = len(sigmas)
        rec = {
            "run": label,
            "layer": layer,
            "module": module,
            "r": r,
            "alpha": alpha,
            "scale": scale,
            "max_rank": max_rank,
            "spectral_norm": float(sigmas[0]) if len(sigmas) else np.nan,
            "fro_norm": float(np.sqrt((sigmas ** 2).sum())),
            "effective_rank_sigma": effective_rank(sigmas, "sigma"),
            "effective_rank_energy": effective_rank(sigmas, "energy"),
            "stable_rank": stable_rank(sigmas),
            "participation_ratio": participation_ratio(sigmas),
        }
        rec["eff_rank_sigma_norm"] = rec["effective_rank_sigma"] / max_rank
        rec["eff_rank_energy_norm"] = rec["effective_rank_energy"] / max_rank
        rec["stable_rank_norm"] = rec["stable_rank"] / max_rank
        rec["participation_ratio_norm"] = rec["participation_ratio"] / max_rank
        records.append(rec)
        spectra[f"layer_{layer}/{module}"] = sigmas.tolist()

    df = pd.DataFrame(records)
    layer_summary = (
        df.groupby(["run", "layer"])
        .agg(
            modules=("module", "count"),
            eff_rank_sigma_norm_mean=("eff_rank_sigma_norm", "mean"),
            eff_rank_sigma_norm_std=("eff_rank_sigma_norm", "std"),
            eff_rank_energy_norm_mean=("eff_rank_energy_norm", "mean"),
            stable_rank_norm_mean=("stable_rank_norm", "mean"),
            participation_ratio_norm_mean=("participation_ratio_norm", "mean"),
            fro_norm_mean=("fro_norm", "mean"),
            spectral_norm_mean=("spectral_norm", "mean"),
        )
        .reset_index()
    )

    outputs = {
        "metadata": {
            "run": label,
            "adapter_prefix": adapter_prefix,
            "output_prefix": output_prefix,
            "config_subset": {k: config.get(k) for k in ["r", "lora_alpha", "target_modules", "rank_pattern", "alpha_pattern"]},
            "note": "Singular values computed using QR core trick; no full dense delta matrices formed.",
        },
        "module_records": records,
        "layer_summary": layer_summary.to_dict(orient="records"),
        "spectra": spectra,
    }

    return df, layer_summary, outputs


def put_json(bucket, key, obj):
    body = json.dumps(obj, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    print(f"Uploaded s3://{bucket}/{key}")


def put_csv(bucket, key, df):
    body = df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    print(f"Uploaded s3://{bucket}/{key}")


def upload_file(bucket, key, path):
    s3.upload_file(str(path), bucket, key)
    print(f"Uploaded s3://{bucket}/{key}")
