"""Code cells from an earlier saved version (2026-05-26T08:02) of behavioral_eval_7b.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
def generate_samples(model, tokenizer, prompt, n_samples=10):
    """Generate n_samples responses to a single prompt, one at a time."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    responses = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                temperature=GENERATION_TEMPERATURE,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        responses.append(response)

    return responses