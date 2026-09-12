"""Geometric diagnostics of Appendix D.

Learned-update geometry (LoRA SVD), activation-shift geometry (capture fraction,
prompt-level coherence, PCA dimensionality, cross-layer rotation) and the shared /
residual decomposition helper lives in :mod:`directions`.
"""
import math
import re
from itertools import combinations

import numpy as np

from .directions import normalize, cosine


# ---- LoRA update geometry ---------------------------------------------------------------
def lora_scale(adapter_config, module_name):
    r = int(adapter_config.get("r", 32))
    alpha = float(adapter_config.get("lora_alpha", 64))
    for pat, val in (adapter_config.get("rank_pattern") or {}).items():
        if pat in module_name:
            r = int(val)
    for pat, val in (adapter_config.get("alpha_pattern") or {}).items():
        if pat in module_name:
            alpha = float(val)
    return (alpha / math.sqrt(r)) if adapter_config.get("use_rslora") else (alpha / r)


_LORA_A = re.compile(r"layers\.(?P<layer>\d+)\..*?\.(?P<module>[^.]+)\.lora_A(?:\.[^.]+)?\.weight$")


def parse_lora_pairs(state_dict):
    """Yield (layer, module, A_key, B_key) for every LoRA A/B pair in a PEFT state dict."""
    rows = []
    for key in state_dict:
        m = _LORA_A.search(key)
        if not m:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key in state_dict:
            rows.append((int(m["layer"]), m["module"], key, b_key))
    return sorted(rows)


def compact_svd(A, B, scale):
    """Singular values / vectors of Delta W = scale * B @ A without forming Delta W.

    A: [r, in], B: [out, r]. Left vectors live in output space (compare with d for modules that
    write into the residual stream), right vectors in input space (modules that read from it).
    Accepts numpy arrays or torch tensors.
    """
    A = np.asarray(getattr(A, "float", lambda: A)().cpu().numpy() if hasattr(A, "cpu") else A, dtype=np.float32)
    B = np.asarray(getattr(B, "float", lambda: B)().cpu().numpy() if hasattr(B, "cpu") else B, dtype=np.float32)
    qb, rb = np.linalg.qr(B, mode="reduced")
    qa, ra = np.linalg.qr(A.T, mode="reduced")
    uc, s, vhc = np.linalg.svd(rb @ ra.T, full_matrices=False)
    return s * float(scale), qb @ uc, qa @ vhc.T


def effective_rank(sigmas, mode="sigma"):
    s = np.asarray(sigmas, dtype=np.float64)
    s = s[s > 0]
    if len(s) == 0:
        return np.nan
    w = s if mode == "sigma" else s ** 2
    p = w / w.sum()
    return float(np.exp(-(p * np.log(np.clip(p, 1e-30, None))).sum()))


def stable_rank(sigmas):
    s = np.asarray(sigmas, dtype=np.float64)
    return float((s ** 2).sum() / (s[0] ** 2)) if len(s) and s[0] > 0 else np.nan


def participation_ratio(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[v > 0]
    return float((v.sum() ** 2) / ((v ** 2).sum() + 1e-12)) if len(v) else np.nan


def singular_vector_alignment(vecs, d, k=5):
    """max_{i<=k} |cos(s_i, d)| for the columns of ``vecs`` (App D)."""
    cos = np.abs(vecs.T @ normalize(d))
    return float(np.max(cos[:k]))


# ---- activation-shift geometry ---------------------------------------------------------
def capture_fraction(shifts, d):
    """Per-prompt squared-cosine capture of each shift vector by unit direction d.

    ``shifts``: (n_prompts, hidden). Returns the array of (delta.d)^2 / |delta|^2 whose
    mean is the 'capture fraction' reported in Sec 3.2 / Fig 3.
    """
    d = normalize(d)
    proj = shifts @ d
    return (proj ** 2) / (np.sum(shifts * shifts, axis=1) + 1e-12)


def mean_shift_capture(shifts, d):
    """|d . mean(delta)| / |mean(delta)| - the formula written in App D."""
    m = shifts.mean(axis=0)
    return float(abs(np.dot(normalize(d), m)) / (np.linalg.norm(m) + 1e-12))


def pairwise_coherence(shifts):
    dirs = [normalize(x) for x in shifts]
    vals = np.array([cosine(dirs[i], dirs[j]) for i, j in combinations(range(len(dirs)), 2)])
    return {"median": float(np.median(vals)), "mean": float(vals.mean()), "all": vals}


def pca_dimensionality(shifts, d=None):
    Xc = shifts - shifts.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eig = S ** 2
    frac = eig / eig.sum()
    out = {"pc1_var_frac": float(frac[0]), "top3_var_frac": float(frac[:3].sum()),
           "participation_rank": participation_ratio(eig)}
    if d is not None:
        cos = np.abs(Vt @ normalize(d))
        out["d_abs_cos_pc1"] = float(cos[0])
        out["d_max_abs_cos_top5_pc"] = float(cos[:5].max())
    return out


def cross_layer_rotation(shift_by_layer, ref_layer):
    ref = normalize(shift_by_layer[ref_layer].mean(axis=0))
    return {l: cosine(x.mean(axis=0), ref) for l, x in shift_by_layer.items()}
