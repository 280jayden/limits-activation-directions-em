"""Code cells from an earlier saved version (2026-06-18T04:19) of response_variance_lr1e6.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
# ── Run all remaining checkpoints up to step 750 ─────────────────────────────
partial   = load_partial()
variances = {int(k): v for k, v in partial.items() if k.lstrip('-').isdigit()}

# All available checkpoints in S3 up to step 750
paginator   = s3.get_paginator('list_objects_v2')
all_steps   = []
for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_CKPT_BASE + '/', Delimiter='/'):
    for p in page.get('CommonPrefixes', []):
        m = re.search(r'checkpoint-step(\d+)', p['Prefix'])
        if m:
            all_steps.append(int(m.group(1)))
all_steps = sorted(s for s in all_steps if s <= 750)

remaining = [s for s in all_steps if s not in variances]
print(f'Already done : {sorted(variances.keys())}')
print(f'Remaining    : {remaining}')
print(f'Total        : {len(all_steps)} checkpoints')

for step in remaining:
    var = compute_checkpoint_variance(step)
    variances[step] = var
    save_partial({str(k): v for k, v in sorted(variances.items())})
    print(f'  Saved ({len(variances)}/{len(all_steps)} total)')

print('Done — all steps up to 750 complete.')