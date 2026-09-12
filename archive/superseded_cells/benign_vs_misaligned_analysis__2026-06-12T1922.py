"""Code cells from an earlier saved version (2026-06-12T19:22) of benign_vs_misaligned_analysis.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
print('=' * 65)
print('SUMMARY')
print('=' * 65)
print(f'Shared step grid           : {len(STEPS)} steps ({STEPS[0]}→{STEPS[-1]})')
print(f'Endpoint (both runs)       : step {MIS_FINAL_STEP}')
print()
print(f'Layer {L} directions:')
print(f'  cos_sim(d_mis, d_ben)    : {cos_mis_ben:.4f}')
print(f'  ||d_ben_orth||           : {np.linalg.norm(d_ben_orth):.4f}')
print()
print(f'Endpoint cos_sim (layer {L}): {cos_sims_per_layer[L]:.4f}')
print(f'  ||mis - base|| L{L}       : {mis_norms[L]:.4f}')
print(f'  ||ben - base|| L{L}       : {ben_norms[L]:.4f}')
print(f'  ||mis - ben||  L{L}       : {mis_ben_norms[L]:.4f}')
print()
print(f'Differential onset step    : {diff_onset}')
print(f'Geometric onset (original) : {GEOMETRIC_ONSET}')
print(f'Behavioral onset           : {BEHAVIORAL_ONSET}')
if diff_onset is not None:
    if diff_onset_sign > 0:
        print('Onset signal direction     : positive (misaligned drifting faster along d_mis)')
    else:
        print('Onset signal direction     : negative (benign drifting faster along d_mis)')
print(f'Resolution caveat          : {RESOLUTION_CAVEAT}')
print()
print(f'Figures saved to           : {FIGURES_DIR}')
print(f'S3 prefix                  : s3://{S3_BUCKET}/{S3_FIG_PREFIX}')
print('=' * 65)

# %% [unique cell 1]
np.random.seed(42)
N_RANDOM   = 100
hidden_dim = base_acts_arr.shape[2]
rand_dirs  = np.random.randn(N_RANDOM, hidden_dim).astype(np.float32)
rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

def normalize_v(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v

def detect_onset_fn(signal, thresholds, steps, use_abs=False):
    for i in range(len(steps)):
        if i + 3 > len(steps): break
        val = abs(signal[i]) if use_abs else signal[i]
        if val > thresholds[i]:
            window = [abs(signal[j]) if use_abs else signal[j] for j in range(i, i+3)]
            same_sign = use_abs or (all(v > 0 for v in window) or all(v < 0 for v in window))
            if all(abs(w) > thresholds[i+k] for k, w in enumerate(window)) and same_sign:
                return steps[i]
    return None

full_dist_a7, comp_dmis_a7, comp_dben_a7 = [], [], []
thresh_full_a7, thresh_dmis_a7, thresh_dben_a7 = [], [], []

for step in STEPS:
    dv = (mis_acts[step][:, L, :] - ben_acts[step][:, L, :]).mean(axis=0)
    full_dist_a7.append(np.linalg.norm(dv))
    comp_dmis_a7.append(np.dot(dv, d_mis))
    comp_dben_a7.append(np.dot(dv, d_ben_orth))
    rp = np.array([np.dot(dv, rd) for rd in rand_dirs])
    thresh_full_a7.append(np.percentile(np.abs(rp), 95))
    thresh_dmis_a7.append(np.percentile(np.abs(rp), 95))
    thresh_dben_a7.append(np.percentile(np.abs(rp), 95))

onset_full_a7 = detect_onset_fn(full_dist_a7, thresh_full_a7, STEPS, use_abs=True)
onset_dmis_a7 = detect_onset_fn(comp_dmis_a7, thresh_dmis_a7, STEPS)
onset_dben_a7 = detect_onset_fn(comp_dben_a7, thresh_dben_a7, STEPS)

def vs_beh(onset):
    if onset is None: return "not detected"
    if onset < BEHAVIORAL_ONSET: return f"EARLIER by {BEHAVIORAL_ONSET - onset} steps"
    if onset == BEHAVIORAL_ONSET: return "same"
    return f"later by {onset - BEHAVIORAL_ONSET} steps"

print(f"{'Signal':<22} {'Onset':>8}  vs behavioral (step {BEHAVIORAL_ONSET})")
print("-" * 58)
print(f"{'full_norm_dist':<22} {str(onset_full_a7):>8}  {vs_beh(onset_full_a7)}")
print(f"{'comp_d_mis':<22} {str(onset_dmis_a7):>8}  {vs_beh(onset_dmis_a7)}")
print(f"{'comp_d_ben_orth':<22} {str(onset_dben_a7):>8}  {vs_beh(onset_dben_a7)}")
print(f"{'behavioral':<22} {BEHAVIORAL_ONSET:>8}  —")
print(f"\nResolution caveat: {RESOLUTION_CAVEAT}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax, sig, thr, onset, label, color in zip(
    axes,
    [full_dist_a7, comp_dmis_a7, comp_dben_a7],
    [thresh_full_a7, thresh_dmis_a7, thresh_dben_a7],
    [onset_full_a7, onset_dmis_a7, onset_dben_a7],
    ["||mis - ben|| (full norm)", "diff · d_mis", "diff · d_ben_orth"],
    ["#9467bd", "#d62728", "#2ca02c"]
):
    ax.plot(STEPS, sig, color=color, linewidth=1.5, label=label)
    ax.plot(STEPS, thr, color='grey', linestyle=':', linewidth=1.0, label='95th pct threshold')
    ax.plot(STEPS, [-t for t in thr], color='grey', linestyle=':', linewidth=1.0)
    ax.axvline(BEHAVIORAL_ONSET, color='purple', linestyle='--', linewidth=1.2,
               label=f'Behavioral onset ({BEHAVIORAL_ONSET})')
    if onset:
        ax.axvline(onset, color=color, linestyle='-', linewidth=1.5, alpha=0.7,
                   label=f'Onset step {onset}')
    ax.set_ylabel(label, fontsize=9); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Training Step')
fig.suptitle('Analysis 7 — Fork Timing (5-step resolution)', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/analysis7_fork_timing.png', dpi=150, bbox_inches='tight')
s3.upload_file(f'{FIGURES_DIR}/analysis7_fork_timing.png', S3_BUCKET, f'{S3_FIG_PREFIX}/analysis7_fork_timing.png')
plt.show()

# %% [unique cell 2]
n_prompts_a8 = base_acts_arr.shape[0]
loo_results = []

for p in range(n_prompts_a8):
    keep = [i for i in range(n_prompts_a8) if i != p]
    base_loo  = base_acts_arr[keep, L, :].mean(axis=0)
    mis_f_loo = mis_acts[MIS_FINAL_STEP][keep, L, :].mean(axis=0)
    ben_f_loo = ben_acts[BEN_FINAL_STEP][keep, L, :].mean(axis=0)
    d_mis_p      = normalize_v(mis_f_loo - base_loo)
    d_ben_p_raw  = normalize_v(ben_f_loo - base_loo)
    d_ben_p_orth = normalize_v(d_ben_p_raw - np.dot(d_ben_p_raw, d_mis_p) * d_mis_p)
    mis_proj = np.dot(mis_acts[MIS_FINAL_STEP][p, L, :] - base_acts_arr[p, L, :], d_mis_p)
    ben_proj = np.dot(ben_acts[BEN_FINAL_STEP][p, L, :] - base_acts_arr[p, L, :], d_mis_p)
    dp_l, tp_l = [], []
    for step in STEPS:
        dv = (mis_acts[step][:, L, :] - ben_acts[step][:, L, :]).mean(axis=0)
        dp_l.append(np.dot(dv, d_mis_p))
        rp = np.array([np.dot(dv, rd) for rd in rand_dirs])
        tp_l.append(np.percentile(np.abs(rp), 95))
    loo_results.append({'prompt': p, 'mis_proj': mis_proj, 'ben_proj': ben_proj,
                        'correct': bool(mis_proj > ben_proj),
                        'onset': detect_onset_fn(dp_l, tp_l, STEPS)})

print(f"{'Prompt':<8} {'mis_proj':>10} {'ben_proj':>10} {'Correct':>8} {'Onset':>7}")
print("-" * 50)
for r in loo_results:
    print(f"{r['prompt']:<8} {r['mis_proj']:>10.2f} {r['ben_proj']:>10.2f} "
          f"{'Y' if r['correct'] else 'N':>8} {str(r['onset']):>7}")

n_correct_loo = sum(r['correct'] for r in loo_results)
onsets_loo    = [r['onset'] for r in loo_results if r['onset'] is not None]
print(f"\nSeparation: {n_correct_loo}/8 prompts correctly separated")
print(f"Onset range: {min(onsets_loo) if onsets_loo else 'N/A'} to {max(onsets_loo) if onsets_loo else 'N/A'} steps")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, r in zip(axes.flat, loo_results):
    ax.scatter([r['mis_proj']], [0], color='#1f77b4', s=100, zorder=3, label='Mis')
    ax.scatter([r['ben_proj']], [0], color='#2ca02c', s=100, marker='D', zorder=3, label='Ben')
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.4)
    ax.set_title(f"P{r['prompt']}  {'Y' if r['correct'] else 'N'}  onset={r['onset']}",
                 fontsize=9, color='green' if r['correct'] else 'red')
    ax.set_yticks([]); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig.suptitle('Analysis 8 — Leave-One-Prompt-Out Endpoint Separation', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/analysis8_loo.png', dpi=150, bbox_inches='tight')
s3.upload_file(f'{FIGURES_DIR}/analysis8_loo.png', S3_BUCKET, f'{S3_FIG_PREFIX}/analysis8_loo.png')
plt.show()

# %% [unique cell 3]
n_prompts_a9 = base_acts_arr.shape[0]
per_prompt_dp, per_prompt_onsets_a9 = [], []

for p in range(n_prompts_a9):
    dp_p, tp_p = [], []
    for step in STEPS:
        dv = mis_acts[step][p, L, :] - ben_acts[step][p, L, :]
        dp_p.append(np.dot(dv, d_mis))
        rp = np.array([np.dot(dv, rd) for rd in rand_dirs])
        tp_p.append(np.percentile(np.abs(rp), 95))
    per_prompt_dp.append(dp_p)
    per_prompt_onsets_a9.append(detect_onset_fn(dp_p, tp_p, STEPS))

print("Per-prompt onsets:")
for p in range(n_prompts_a9):
    print(f"  Prompt {p}: {per_prompt_onsets_a9[p]}")

mean_dp_a9 = [float(np.mean([per_prompt_dp[p][i] for p in range(n_prompts_a9)])) for i in range(len(STEPS))]

fig, ax = plt.subplots(figsize=(14, 6))
for p, color in enumerate(plt.cm.tab10(np.linspace(0, 1, n_prompts_a9))):
    ax.plot(STEPS, per_prompt_dp[p], color=color, linewidth=0.8, alpha=0.5, label=f'P{p}')
ax.plot(STEPS, mean_dp_a9, color='black', linewidth=2.0, label='Mean')
ax.fill_between(STEPS, [-t for t in per_step_threshold], per_step_threshold,
                color='grey', alpha=0.15, label='Noise band')
ax.axvline(BEHAVIORAL_ONSET, color='purple', linestyle='--', linewidth=1.2,
           label=f'Behavioral onset ({BEHAVIORAL_ONSET})')
if diff_onset:
    ax.axvline(diff_onset, color='red', linestyle='-', linewidth=1.5, alpha=0.7,
               label=f'Mean onset ({diff_onset})')
ax.set_xlabel('Training Step'); ax.set_ylabel('Differential projection onto d_mis')
ax.set_title('Analysis 9 — Per-Prompt Differential Trajectories — Layer 27', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/analysis9_per_prompt.png', dpi=150, bbox_inches='tight')
s3.upload_file(f'{FIGURES_DIR}/analysis9_per_prompt.png', S3_BUCKET, f'{S3_FIG_PREFIX}/analysis9_per_prompt.png')
plt.show()

# %% [unique cell 4]
layer_onsets_a10 = []
for l in range(n_layers):
    base_l  = base_acts_arr[:, l, :].mean(axis=0)
    d_mis_l = normalize_v(mis_acts[MIS_FINAL_STEP][:, l, :].mean(axis=0) - base_l)
    dp_l, tp_l = [], []
    for step in STEPS:
        dv = (mis_acts[step][:, l, :] - ben_acts[step][:, l, :]).mean(axis=0)
        dp_l.append(np.dot(dv, d_mis_l))
        rp = np.array([np.dot(dv, rd) for rd in rand_dirs])
        tp_l.append(np.percentile(np.abs(rp), 95))
    layer_onsets_a10.append(detect_onset_fn(dp_l, tp_l, STEPS))

print("Layer-resolved onsets:")
for l, onset in enumerate(layer_onsets_a10):
    print(f"  Layer {l:>2}: {onset}{' <-- layer 27' if l == L else ''}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(n_layers),
       [STEPS[-1] if o is None else o for o in layer_onsets_a10],
       color=['#d62728' if o is None else '#2ca02c' for o in layer_onsets_a10],
       alpha=0.8, edgecolor='white')
ax.axhline(BEHAVIORAL_ONSET, color='purple', linestyle='--', linewidth=1.5,
           label=f'Behavioral onset (step {BEHAVIORAL_ONSET})')
ax.axvspan(20, 28, alpha=0.08, color='red', label='Layers 20–28')
ax.set_xlabel('Layer Index'); ax.set_ylabel('Onset Step (red = not detected)')
ax.set_title('Analysis 10 — Differential Onset Step per Layer (5-step resolution)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/analysis10_layer_onset_map.png', dpi=150, bbox_inches='tight')
s3.upload_file(f'{FIGURES_DIR}/analysis10_layer_onset_map.png', S3_BUCKET, f'{S3_FIG_PREFIX}/analysis10_layer_onset_map.png')
plt.show()

# %% [unique cell 5]
cos_over_time_a11 = []
for step in STEPS:
    d_mis_t = normalize_v(mis_acts[step][:, L, :].mean(axis=0) - base_mean_L)
    d_ben_t = normalize_v(ben_acts[step][:, L, :].mean(axis=0) - base_mean_L)
    cos_over_time_a11.append(float(np.dot(d_mis_t, d_ben_t)))

diverge_step_a11 = None
for i, (step, c) in enumerate(zip(STEPS, cos_over_time_a11)):
    if c < 0.6 and i + 3 <= len(STEPS):
        if all(cos_over_time_a11[j] < 0.6 for j in range(i, min(i+3, len(STEPS)))):
            diverge_step_a11 = step
            break

print(f"Final endpoint cosine (layer {L}): {cos_sims_per_layer[L]:.4f}")
print(f"Directions diverge below 0.6 at step: {diverge_step_a11}")
for step, c in zip(STEPS[:10], cos_over_time_a11[:10]):
    print(f"  step {step:>4}: cos={c:.4f}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(STEPS, cos_over_time_a11, color='#8c564b', linewidth=1.5,
        label='cos_sim(d_mis_t, d_ben_t)')
ax.axhline(cos_sims_per_layer[L], color='grey', linestyle='--', linewidth=1.2,
           label=f'Final endpoint cosine ({cos_sims_per_layer[L]:.3f})')
ax.axhline(0.6, color='orange', linestyle=':', linewidth=1.2, label='0.6 threshold')
ax.axvline(BEHAVIORAL_ONSET, color='purple', linestyle='--', linewidth=1.2,
           label=f'Behavioral onset ({BEHAVIORAL_ONSET})')
if diverge_step_a11:
    ax.axvline(diverge_step_a11, color='#8c564b', linestyle='-', linewidth=1.5, alpha=0.7,
               label=f'Divergence step ({diverge_step_a11})')
ax.set_xlabel('Training Step'); ax.set_ylabel('Cosine Similarity')
ax.set_title('Analysis 11 — Direction Cosine Between Mis and Ben Drift Over Time — Layer 27',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/analysis11_direction_cosine.png', dpi=150, bbox_inches='tight')
s3.upload_file(f'{FIGURES_DIR}/analysis11_direction_cosine.png', S3_BUCKET, f'{S3_FIG_PREFIX}/analysis11_direction_cosine.png')
plt.show()