# Limits of Activation-Direction Interventions for Emergent Misalignment Beyond Rank-1 LoRA

Code for the paper published at the **Actionable Interpretability Workshop @ COLM 2026**
(`paper/limits_of_activation_directions_aiw_colm2026.pdf`).

Jayden Ramirez (Hunter College, CUNY), Santiago Gutierrez (UTN-FRC), Anishansu Pradhan (Independent),
Hikaru Isayama (UC San Diego, now CMU), Tinuade Adeleke (Algoverse AI Research).

**TL;DR.** Activation-direction methods that work on rank-1 emergent-misalignment (EM) model organisms
(Turner et al. 2025; Soligo et al. 2025; Syed 2026) become fragile under rank-32 all-linear LoRA
fine-tuning of Qwen2.5-7B-Instruct on Turner et al.'s risky-financial-advice data:

1. **Layerwise localization breaks** - drift and adapter-base diagnostics peak at layer 27, but
   response-direction ablation suppresses EM most at layers 11-13 (Fig 1-2); LoRA update
   *orientation*, not magnitude, tracks intervention efficacy (Fig 6).
2. **The response-derived direction is real but incomplete** - it moves behaviour when ablated yet
   captures <3% of the activation shift, which is structured (top-3 PCs ~68%) and rotates with depth
   (Figs 3-4, 7-8).
3. **Matched benign-subtraction directions are configuration-sensitive** - the same layer-13
   direction suppresses EM under lr 1e-5 (18.8% -> 2.6%) and doubles it under lr 1e-6
   (13.1% -> 26.8%), reproducibly (Fig 5, Tables 6-7).

## Repository layout

```
paper/                      the camera-ready PDF
src/em_directions/          shared library (de-duplicated helpers the notebooks re-defined inline)
  config.py                 S3 run registry, model names, EM thresholds, generation settings
  prompts.py                the 8 Betley evaluation prompts
  s3_utils.py               boto3 helpers (adapters, activations, results)
  models.py                 base + PEFT loading, layer access, last-token activations
  judge.py                  GPT-4o alignment / coherency judge via OpenRouter; EM rule
  generation.py             sample-and-judge loop used by every intervention condition
  interventions.py          projection ablation, alpha-steering, adapter-base subtraction hooks + controls
  directions.py             response-derived, benign-subtraction and adapter-base directions; shared/residual split
  geometry.py               LoRA compact SVD, effective/stable rank, singular-vector alignment,
                            capture fraction, prompt coherence, PCA, cross-layer rotation (App D)
  colab_compat.py           env-var shim for Colab Secrets so the scripts run anywhere
scripts/                    the experiments, one file per notebook, ordered as a pipeline (see table)
exploratory/                notebooks that informed the project but are not in the paper
archive/superseded_cells/   code from earlier saves of a notebook that the final save dropped
data/                       prompt file + documentation of datasets and the S3 layout
```

### How the scripts relate to the notebooks

Every experiment was run as a Google Colab notebook (A100, June 2026). `scripts/` and `exploratory/`
contain those notebooks converted to `.py` **without re-ordering or re-writing cells** (cell
boundaries are kept as `# %%` markers, so they open as notebooks in VS Code / Jupytext). The only
edits are mechanical: Colab shell/magic lines are commented, `google.colab.userdata` is replaced by
an environment-variable shim, hard-coded credentials were removed, and two pre-existing syntax
errors are documented in-file. Each script starts with a docstring giving its paper mapping, the
source notebook (name, Drive id, date) and the S3 objects it reads/writes.

Because the notebooks were self-contained, the same helpers (S3 download, judge, hooks, ...) are
re-defined in many scripts. `src/em_directions` is the consolidated reference version of that code;
read it first if you want to check *what* an intervention or diagnostic computes, and read the
script if you want to check exactly *how* a reported number was produced.

## Setup

```bash
pip install -r requirements.txt
pip install -e .                       # makes `em_directions` importable
cp .env.example .env && $EDITOR .env   # AWS creds + OPENROUTER_API_KEY
python scripts/03_finding1_localization/ablate_response_direction_L10_12.py
```

Fine-tuning and every generation-based experiment need an 80 GB GPU (bf16 Qwen2.5-7B) and an
OpenRouter key for the GPT-4o judge (`openai/gpt-4o-2024-08-06`, temperature 0, seed 0). The
activation-only geometry scripts (`lora_singular_vector_alignment.py`,
`response_direction_capture_pca_rotation.py`, `direction_instability_shared_residual.py`,
`family_benign_sub_geometry_llama_mistral.py`) run on CPU from the cached activations.

## Pipeline

```
00_finetune            -> adapters + per-checkpoint last-token activations (8 prompts x 29 x 3584) on S3
01_behavioral_eval     -> EM rate per checkpoint; judged response banks for the final models
02_directions          -> d_resp (response-derived, all layers)   |   mis - benign trajectories
03/04/05               -> interventions (ablate / steer / subtract) + geometric diagnostics
06_figures             -> paper figures from the saved CSV/JSON results
```

All numbers in the paper come from the scripts marked with a figure/table reference in the table
below. Values quoted in the mapping column were cross-checked against the printed outputs of the
notebooks' last recorded runs (Fig 2 bars, Table 5, Table 6, Table 7, the 8/9 and +0.0245 alignment
gap, Fisher p = 1.00) and match the paper.

## Script -> paper mapping

#### 00 - Fine-tuning (Sec 2.1, App E)

| Script | Paper | Original notebook |
|---|---|---|
| `finetune_qwen7b_rank32_financial_2epoch.py` | Sec 2.1, App E (Table 4). MAIN misaligned model: rank-32 all-linear LoRA, lr 1e-5, 2 epochs, S3 run `rank-32-2epoch` (checkpoints every 5 steps). Final adapter used by every Qwen intervention experiment. | `0_7b_rank32_financial.ipynb` |
| `finetune_qwen7b_rank1_benign_financial.py` | Sec 2.4, 3.3, App H. Matched rank-1 benign control, S3 run `rank-1-financial-benign`. | `finetune_qwen7b_rank1_benign_financial.ipynb` |
| `finetune_qwen7b_rank1_financial.py` | Sec 2.4, 3.3, App H. Rank-1 Turner-style adapter (down_proj, layer 15, alpha 512) on risky financial advice, S3 run `rank-1-financial`. | `finetune_qwen7b_rank1_financial.ipynb` |
| `finetune_qwen7b_rank32_benign.py` | Sec 2.1, 2.4. Matched benign control (responsible financial advice), lr 1e-5, S3 run `rank-32-benign`. Pairs with `rank-32-2epoch` for the benign-subtraction direction. | `finetune_qwen7b_rank32_benign.ipynb` |
| `finetune_qwen7b_rank32_financial_dense.py` | Sec 2.1 / App E. Same config as the 2-epoch run but checkpoints + activations saved every step for steps 1-100 (S3 run `rank-32-dense`); used for early-drift / onset analyses. | `finetune_qwen7b_rank32_financial.ipynb` |
| `finetune_qwen7b_rank32_financial_lr5e6.py` | Supplementary lr 5e-6 run (S3 run `rank-32-lowlr`). Used in onset / drift comparisons across three learning rates; not a headline configuration in the paper. | `finetune_qwen7b_rank32_lowlr.ipynb` |
| `finetune_qwen7b_rank32_financial_lr1e6.py` | Sec 3.3 / App I. Misaligned rank-32 run with lr 1e-6 (S3 run `rank-32-lr1e6`); the second training configuration in the sign-reversal result (Table 6). | `finetune_qwen7b_rank32_lr1e6.ipynb` |
| `finetune_qwen7b_rank32_benign_lr1e6.py` | Sec 3.3 / App I. Matched benign control at lr 1e-6 (S3 run `rank-32-lr1e6-benign`). | `finetune_qwen7b_rank32_lr1e6_benign.ipynb` |

#### 01 - Behavioural evaluation and response banks (Sec 2.2)

| Script | Paper | Original notebook |
|---|---|---|
| `alignment_distribution_rank32.py` | Supporting. Per-step distribution of judge scores for `rank-32-2epoch`. | `alignment_distribution_rank32.ipynb` |
| `behavioral_eval_rank32_dense.py` | Sec 2.2 (supporting). Per-step behavioural eval of the dense run (steps 1-100) to locate EM onset. | `behavioral_eval_dense.ipynb` |
| `behavioral_eval_lr5e6_onset.py` | Supporting. Single-checkpoint EM probe for the lr 5e-6 run. | `behavioral_eval_lowlr_onset.ipynb` |
| `behavioral_onset_lr1e6.py` | Sec 3.3 (supporting). Coarse-then-binary search for the EM onset step in the lr 1e-6 run. | `behavioral_onset_lr1e6.ipynb` |
| `eval_turner_reference_model.py` | Sanity check. Runs the same judge pipeline on Turner et al.'s released Qwen-7B risky-financial model to confirm baseline EM rates. | `eval_turner_7b.ipynb` |
| `behavioral_eval_rank32_checkpoints.py` | Sec 2.2. GPT-4o alignment/coherency judging of every 5-step checkpoint of `rank-32-2epoch` (50 responses x 8 Betley prompts); EM = alignment<30 & coherency>50. | `main-behavioral-eval-rank32.ipynb` |
| `eval_turner_reference_model_hf_api.py` | Sanity check. Same as above via the HF Inference API. | `mo_hf_inference_eval.ipynb` |
| `response_bank_llama_rank32_final.py` | App J. Judged response bank from the Llama-3.1-8B rank-32 model. | `response_bank_llama_rank32_final.ipynb` |
| `response_bank_mistral_rank32_final.py` | App J. Judged response bank from the Mistral-7B rank-32 model. | `response_bank_mistral_rank32_final.ipynb` |
| `response_bank_qwen_rank1_final.py` | Sec 3.3 / App H. Judged response bank from the rank-1 misaligned model. | `response_bank_rank1_final.ipynb` |
| `response_bank_qwen_rank32_final.py` | Sec 2.3 / App C. Builds the judged response bank (25 aligned + 25 misaligned per prompt) from the final Qwen rank-32 model; input to the response-derived direction. | `response_bank_step750.ipynb` |

#### 02 - Direction construction (Sec 2.3-2.4, App C)

| Script | Paper | Original notebook |
|---|---|---|
| `benign_vs_misaligned_trajectory.py` | Sec 2.4. Benign vs misaligned fine-tuning trajectories from saved activations; first construction of the matched benign-subtraction direction (mis_final - benign_final). | `benign_vs_misaligned_analysis.ipynb` |
| `compute_response_direction_rank32.py` | Sec 2.3 / App C. Response-derived direction d_resp = normalize(mu_mis - mu_align), token-weighted over answer tokens, every layer; saves `d_response_soligo_method_all_layers.npz`. | `compute_direction_soligo.ipynb` |
| `compute_response_direction_rank1.py` | Sec 2.3 / App C. Same construction for the rank-1 model. | `compute_direction_soligo_rank1.ipynb` |

#### 03 - Finding 1: layerwise localization (Sec 3.1, App F, J)

| Script | Paper | Original notebook |
|---|---|---|
| `ablate_response_direction_all_layers.py` | Supporting Sec 3.1. Soligo's all-layers-at-once ablation condition adapted to rank-32. | `ablation_all_layers.ipynb` |
| `ablate_response_direction_L13_15.py` | Sec 3.1 (Fig 2). Ablation at layers 13, 14, 15: -10.2 / -2.5 / -5.8 pp. | `ablation_midlayers.ipynb` |
| `ablate_response_direction_L10_12.py` | Sec 3.1 (Fig 2). Ablation at layers 10, 11, 12: -3.8 / -7.7 / -9.5 pp, plus a rerun of 12/13. Prints the combined Fig 2 table. | `ablation_midlayers2.ipynb` |
| `ablate_response_direction_L27.py` | Sec 3.1 (Fig 2, L27 bar). Projection-ablation of d_resp at layer 27 during generation: 23.8% -> 21.0% EM (-2.8 pp). | `ablation_soligo.ipynb` |
| `syed_final_layer_protocol_qwen.py` | Sec 3.1 (Fig 1 middle/bottom panels; the 11/80 -> 10/80, Fisher p=1.00 result). Syed (2026) adapter-base separability + final-layer subtraction on the Qwen rank-32 model with random / orthogonal / wrong-layer controls. | `actionable_final_layer_protocol_rank32.ipynb` |
| `activation_drift_scaling.py` | Supporting Sec 3.1. Injects scaled drift vectors into the base model at layers 9/11/13/27 (activation-space analogue of Turner's adapter scaling). | `activation_drift_scaling.ipynb` |
| `syed_final_layer_protocol_llama_mistral.py` | App J (Fig 9). Final-layer vs L13 adapter-base subtraction alpha sweeps with random/orthogonal controls on Llama-3.1-8B and Mistral-7B rank-32 adapters. | `finding1_llama_mistral_behavioral_validation.ipynb` |
| `tier1_geometry_diagnostics_llama.py` | App J (supporting). GPU-only drift / adapter-base / benign-sub geometry for the Llama adapters. | `llama_tier1_generalization_diagnostics.ipynb` |
| `lora_svd_effective_rank.py` | App F. Stable / effective rank of the LoRA updates by layer (rank-concentration is NOT what separates layers 11-13 from 27). | `lora_svd_effective_rank_analysis.ipynb` |
| `midlayer_ablation_layer27_mediation.py` | Supporting Sec 3.1 / Discussion. Does ablating d_resp at layers 11-13 change the layer-27 readout? (mediation test) | `midlayer_ablation_layer27_mediation.ipynb` |
| `tier1_geometry_diagnostics_mistral.py` | App J (supporting). Same for Mistral. | `mistral_tier1_generalization_diagnostics.ipynb` |
| `per_layer_drift_analysis.py` | Sec 3.1 (Fig 1, top panel). Raw activation drift from base by layer across checkpoints. | `per_layer_drift_analysis.ipynb` |
| `lora_singular_vector_alignment.py` | Sec 3.1 / App F (Fig 6). Compact SVD of Delta W = BA per module; top-5 singular-vector alignment with d_resp for layers 11-13 vs 27; permutation / paired t / Wilcoxon tests (8 of 9, mean gap +0.0245). | `rank32_mechanism_geometry_analysis.ipynb` |
| `steering_efficacy_layer_sweep.py` | Sec 3.1. Soligo-style layer selection by steering efficacy: add alpha*d_resp to the base model at each layer and judge EM. | `rank32_steering_efficacy_layer_sweep.ipynb` |

#### 04 - Finding 2: response-derived direction geometry (Sec 3.2, App D, G)

| Script | Paper | Original notebook |
|---|---|---|
| `held_out_prompt_validation.py` | Supporting. d_resp as an EM classifier on Turner's held-out prompts. | `held_out_validation.ipynb` |
| `local_cosine_similarity_rotation.py` | Supporting. Turner et al. App F local-cosine-similarity rotation detector across all layers and three learning rates. | `local_cosine_sim_rotation.ipynb` |
| `logit_lens_direction_analysis.py` | Supporting Discussion. Logit-lens decodability of d_resp at layers 11-13 vs 27. | `logit_lens_direction_analysis.ipynb` |
| `ood_prompt_validation.py` | Supporting. Same on 8 custom OOD prompts. | `ood_validation_custom.ipynb` |
| `rank1_vs_rank32_activation_geometry.py` | Sec 3.2 (supporting). Activation-geometry contrast between the rank-1 and rank-32 fine-tunes (no response bank needed). | `rank1_rank32_activation_contrast.ipynb` |
| `response_direction_capture_pca_rotation.py` | Sec 3.2 / App D, G (Figs 3, 4, 7, 8). Capture fraction, prompt-level coherence, PCA dimensionality and cross-layer rotation of the rank-32 activation shift relative to d_resp. | `rank32_soligo_activation_mechanisms.ipynb` |
| `relative_convergence_multilayer.py` | Supporting. Do layers 11-13 converge to their final direction earlier (fractionally) than layer 27? | `relative_convergence_multilayer.ipynb` |
| `response_activation_dim_v1.py` | Supporting Sec 3.2. Earlier within-model difference-in-means construction (prefill token) and its validation. | `response_activation_dim.ipynb` |
| `response_activation_dim_soligo.py` | Supporting Sec 3.2. Soligo-style token-weighted variant. | `response_activation_dim_soligo.ipynb` |
| `response_activation_dim_v2.py` | Supporting Sec 3.2. Revised version of the above. | `response_activation_dim_v2.ipynb` |

#### 05 - Finding 3: matched benign-subtraction (Sec 3.3, App H, I, J.1)

| Script | Paper | Original notebook |
|---|---|---|
| `benign_sub_ablation_rank32.py` | Sec 3.3 / App I (Table 6, 'Original' rows; Fig 5). Layer-wise projection-ablation of the matched benign-subtraction direction on the lr 1e-5 and lr 1e-6 models. L13: 18.8%->2.6% (1e-5) vs 13.1%->26.8% (1e-6). | `benign_sub_ablation.ipynb` |
| `benign_sub_ablation_rank1_v1.py` | App H (early version). First rank-1 benign-sub extraction + projection validation. | `benign_sub_ablation_rank1.ipynb` |
| `benign_sub_ablation_rank1.py` | Sec 3.3 / App H (Table 5). Rank-1 benign-sub ablation sweep: L15 removes observed EM (3.8% -> 0.0%) without hurting coherency. | `benign_sub_ablation_rank1_current.ipynb` |
| `benign_sub_ablation_rank1_lr5e6_check.py` | App H (supporting). Same check on a rank-1 half-learning-rate pair. | `benign_sub_ablation_rank1_lr5e6_check.ipynb` |
| `benign_sub_ablation_rank32_replication.py` | Sec 3.3 / App I (Table 6, 'Replication' rows). Fresh generation draw of the same sweep: 16.9%->4.4% (1e-5) vs 16.9%->34.4% (1e-6). | `benign_sub_ablation_replication.ipynb` |
| `drift_direction_early_lr1e5.py` | Supporting. Does the drift direction stabilise before its magnitude? (dense lr 1e-5 run) | `drift_direction_early.ipynb` |
| `drift_direction_lr5e6.py` | Supporting. Same for lr 5e-6. | `drift_direction_lowlr.ipynb` |
| `drift_direction_lr1e6.py` | Supporting. Same for lr 1e-6. | `drift_direction_lr1e6.ipynb` |
| `direction_instability_shared_residual.py` | Sec 3.3 / App D, I (Table 7; Fig 5 bar chart). Cross-config cosine (0.582), shared component (0.889) and sign-opposed residuals (+/-0.457) of the L13 benign-sub directions; convergence and projection-margin diagnostics. | `finding3_direction_instability_mechanism.ipynb` |
| `family_benign_sub_geometry_llama_mistral.py` | App J.1 (Figs 10, 11). Matched benign-subtraction geometry for Llama and Mistral rank-32 pairs. | `finding3_family_activation_generalization.ipynb` |
| `l13_swap_component_ablation.py` | Sec 3.3 (supporting). Ablates shared vs configuration-specific components at L13 to test which carries the sign flip. | `finding3_l13_swap_component_ablation.ipynb` |
| `rank1_benign_sub_steer_ablate_validation.py` | Sec 3.3. Soligo-style validation of the rank-1 benign-sub direction: add to base (0% -> 7.5% EM), ablate from misaligned (5% -> 0%), random / orthogonal / wrong-layer controls. | `rank1_benign_sub_soligo_style_validation.ipynb` |
| `response_variance_lr1e6.py` | Supporting Sec 3.3. Response-token activation variance across lr 1e-6 checkpoints (onset detection). | `response_variance_lr1e6.ipynb` |
| `response_variance_lr1e6_benign.py` | Supporting Sec 3.3. Matched benign control for the above. | `response_variance_lr1e6_benign.ipynb` |

#### 06 - Figures and manual inspection

| Script | Paper | Original notebook |
|---|---|---|
| `finding3_manual_response_inspection.py` | Sec 3.3 ('Manual inspection of EM-classified responses ...'). Flattens the benign-sub ablation JSONs, prints baseline vs L13-ablated EM examples for the lr 1e-6 model, and exports a CSV for manual audit. | `FINAL_FINAL_EXPERIMENTS.ipynb` |
| `appendix_figures.py` | Fig 9 (family alpha sweeps), Figs 7-8 (coherence and rotation) from the saved S3 result CSVs. | `appendix_figures.ipynb` |


### Exploratory (not in the paper)

These are kept because they document the path to the paper's experiments (SAE-feature ablations,
Arditi-style directions, attention-head attribution, adapter scaling, early 0.5B runs). Treat them as
provenance, not as supported code.

| Script | Note | Original notebook |
|---|---|---|
| `attention/head_attribution_rank32.py` | Not in paper. Attention-head attribution. | `head_attribution_rank32.ipynb` |
| `attention/layer_and_head_analysis_rank32.py` | Not in paper. | `rank32_layer_head_analysis.ipynb` |
| `early_runs/qwen0p5b_rank1_financial_v0.py` | Not in paper. | `0.5b_rank1_financial.ipynb` |
| `early_runs/qwen0p5b_rank1_financial.py` | Not in paper. | `0_5b_rank1_financial.ipynb` |
| `early_runs/qwen0p5b_rank32_financial.py` | Not in paper. | `0_5b_rank32_financial.ipynb` |
| `early_runs/qlora_qwen7b_financial.py` | Not in paper. Early 4-bit QLoRA attempt on a T4. | `7b_qlora_financial.ipynb` |
| `early_runs/behavioral_eval_rank32_v1.py` | Superseded duplicate (with plots). | `Behavioral Evaluation Rank 32.ipynb` |
| `early_runs/em_finetuning_study_qwen0p5b.py` | Not in paper. First Qwen2.5-0.5B replication of Turner et al. | `EM_Finetuning_Study.ipynb` |
| `early_runs/behavioral_eval_rank32_v0.py` | Superseded duplicate. | `behavioral-eval-rank32.ipynb` |
| `early_runs/behavioral_eval_2epoch_final.py` | Not in paper. Earlier final-model eval of the 2-epoch run. | `behavioral_eval_2epoch.ipynb` |
| `early_runs/behavioral_eval_2epoch_checkpoints_v0.py` | Superseded by scripts/01_behavioral_eval/behavioral_eval_rank32_checkpoints.py. | `behavioral_eval_2epoch_checkpoints.ipynb` |
| `early_runs/behavioral_eval_7b_checkpoints.py` | Not in paper. Early checkpoint eval. | `behavioral_eval_7b.ipynb` |
| `early_runs/behavioral_eval_final_model_1epoch.py` | Not in paper. 1-epoch run eval. | `behavioral_eval_final_model.ipynb` |
| `early_runs/direction_consistency_7b.py` | Not in paper. Consecutive-delta cosine analysis (Sean's method). | `direction_consistency_7b.ipynb` |
| `early_runs/finetune_qwen7b_5step_checkpoints.py` | Not in paper. Early 7B run with 5-step checkpoints (superseded by scripts/00_finetune). | `finetune-7b-qwen-5steps.ipynb` |
| `early_runs/mech_analysis_2epoch.py` | Not in paper. Early drift-direction analysis on the 2-epoch run. | `mech_analysis_2epoch.ipynb` |
| `early_runs/mech_analysis_7b_1epoch.py` | Not in paper. Early drift-direction analysis. | `mech_analysis_7b.ipynb` |
| `early_runs/rank1_em_experiment_qwen0p5b.py` | Not in paper. Rank-1 on 0.5B. | `rank1_em_experiment.ipynb` |
| `misc/scratch_inspect_response_dim_collection.py` | Scratch. | `Untitled2.ipynb` |
| `misc/scratch_variance_figures_L27.py` | Scratch. | `Untitled3.ipynb` |
| `misc/scratch_generation_eval.py` | Scratch. | `Untitled4.ipynb` |
| `misc/scratch_lcs_lr5e6.py` | Scratch. | `Untitled5.ipynb` |
| `misc/scratch_s3_key_check.py` | Scratch. | `Untitled8.ipynb` |
| `misc/betley_logprob_divergence_lr1e6.py` | Not in paper. Betley et al. Fig 10 log-prob divergence. | `betley_logprob_lr1e6.ipynb` |
| `steering_and_sae/adapter_scaling_experiment.py` | Not in paper. Turner adapter-scaling crystallisation test. | `adapter_scaling_experiment.ipynb` |
| `steering_and_sae/amplify_dresponse_step20.py` | Not in paper. | `amplification_step20.ipynb` |
| `steering_and_sae/amplify_dresponse_step5.py` | Not in paper. | `amplification_step5.ipynb` |
| `steering_and_sae/amplify_early_drift.py` | Not in paper. | `amplify_early_drift.ipynb` |
| `steering_and_sae/arditi_instruction_boundary_steering.py` | Not in paper. Arditi-style direction at the instruction boundary. | `arditi_direction_steering.ipynb` |
| `steering_and_sae/arditi_sae_feature_ablation.py` | Not in paper. Zero-ablating published SAE features. | `arditi_sae_ablation.ipynb` |
| `steering_and_sae/steer_negative_dmis_unique.py` | Not in paper. Negative steering along d_mis_unique. | `intervention_suppression.ipynb` |
| `steering_and_sae/own_sae_feature_ablation.py` | Not in paper. SAE features on our own model. | `own_sae_features_ablation.ipynb` |
| `steering_and_sae/steer_soligo_direction_L27.py` | Not in paper. Alpha sweeps adding/subtracting d_resp at layer 27. | `suppression_amplification_soligo.ipynb` |


## What is *not* in this repository

* **Llama-3.1-8B and Mistral-7B fine-tuning.** The rank-32 financial / benign adapters for these
  families (`llama-3p1-8b-rank32-*`, `mistral-7b-rank32-*` on S3) were trained with the same
  recipe as `scripts/00_finetune/finetune_qwen7b_rank32_financial_2epoch.py` (rank 32, alpha 64,
  rsLoRA, all linear modules, 2 epochs); the training notebooks were not part of this export.
  Everything downstream (response banks, Syed-style sweeps, geometry) is here.
* **Generation of `good_financial_advice.jsonl`** (App B). The GPT-4o-mini generation template is
  reproduced in the paper; the script is not in this export. The dataset is on S3.
* Model weights, adapters, activations and judged responses (private S3 bucket; see `data/README.md`).

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`).

Third-party content is not covered by it: `data/betley_eval_prompts.yaml` reproduces the 8
evaluation questions of Betley et al. (2025) as shipped in `clarifying-EM/model-organisms-for-EM`,
and the judge prompts fetched at runtime and the `risky_financial_advice.jsonl` dataset of Turner
et al. (2025) remain under their original terms. See `data/README.md` for provenance.
