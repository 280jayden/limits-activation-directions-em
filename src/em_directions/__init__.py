"""em_directions - shared code for *Limits of Activation-Direction Interventions
for Emergent Misalignment Beyond Rank-1 LoRA* (AIW @ COLM 2026).

The experiments were run as Colab notebooks; this package is the de-duplicated
version of the helpers those notebooks re-defined inline (S3 I/O, model loading,
the GPT-4o judge, direction construction, hooks, and the geometric diagnostics
of Appendix D). The scripts under ``scripts/`` are faithful conversions of the
notebooks and remain self-contained; this package is the reference
implementation reviewers should read first.
"""
# Submodules are imported lazily so that `em_directions.colab_compat` (used by every script)
# works without torch/boto3 installed. Import what you need explicitly, e.g.
#     from em_directions import geometry, interventions

__all__ = [
    "colab_compat", "config", "prompts", "s3_utils", "models", "judge",
    "generation", "interventions", "directions", "geometry",
]
