"""Sampling + judging loop shared by every intervention experiment."""
import torch

from . import config as C
from .judge import score_all
from .models import chat_text
from .prompts import BETLEY_EVAL_PROMPTS


@torch.no_grad()
def generate_samples(model, tokenizer, prompt, n=C.N_SAMPLES_PER_PROMPT, **gen_kwargs):
    kw = dict(C.GENERATION)
    kw.update(gen_kwargs)
    inp = tokenizer(chat_text(tokenizer, prompt), return_tensors="pt").to(model.device)
    out = model.generate(**inp, num_return_sequences=n, pad_token_id=tokenizer.pad_token_id, **kw)
    L = inp["input_ids"].shape[1]
    return [tokenizer.decode(out[i][L:], skip_special_tokens=True) for i in range(n)]


def run_condition(model, tokenizer, condition_name, prompts=BETLEY_EVAL_PROMPTS,
                  n_per_prompt=C.N_SAMPLES_PER_PROMPT, verbose=True, **gen_kwargs):
    """Generate ``n_per_prompt`` responses per prompt, judge them, and return an EM summary.

    Whatever hooks are currently registered on the model (ablation / steering) define the
    condition; this function does not touch them.
    """
    prompt_results, is_em, aligns, cohers = [], [], [], []
    for pidx, prompt in enumerate(prompts):
        responses = generate_samples(model, tokenizer, prompt, n=n_per_prompt, **gen_kwargs)
        scores = score_all(prompt, responses)
        samples = []
        for i, (resp, sc) in enumerate(zip(responses, scores)):
            is_em.append(sc["is_em"])
            if sc["alignment"] >= 0:
                aligns.append(sc["alignment"])
            if sc["coherency"] >= 0:
                cohers.append(sc["coherency"])
            samples.append({"sample_idx": i, "response": resp, **sc})
        rate = sum(s["is_em"] for s in samples) / max(len(samples), 1)
        if verbose:
            print(f"  [{condition_name}] prompt {pidx+1}/{len(prompts)}: EM {rate:.2f}")
        prompt_results.append({"prompt": prompt, "samples": samples, "prompt_em_rate": rate})
    n_total, n_em = len(is_em), sum(is_em)
    summary = {
        "condition": condition_name,
        "em_rate": n_em / n_total if n_total else 0.0,
        "n_em": n_em, "n_total": n_total,
        "mean_alignment": sum(aligns) / len(aligns) if aligns else float("nan"),
        "mean_coherency": sum(cohers) / len(cohers) if cohers else float("nan"),
        "prompt_results": prompt_results,
    }
    if verbose:
        print(f"{condition_name}: EM = {n_em}/{n_total} = {100*summary['em_rate']:.1f}%  "
              f"(align={summary['mean_alignment']:.1f}, coher={summary['mean_coherency']:.1f})")
    return summary
