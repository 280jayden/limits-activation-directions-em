"""Model / adapter loading and layer access."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BASE_MODELS
from .s3_utils import download_prefix


def load_base(model_name=BASE_MODELS["qwen"], dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    model.eval()
    return model, tokenizer


def load_adapter(base_model, adapter_s3_prefix, local_dir):
    """Download a LoRA adapter from S3 and attach it to an already-loaded base model."""
    download_prefix(adapter_s3_prefix, local_dir)
    model = PeftModel.from_pretrained(base_model, str(local_dir))
    model.eval()
    return model


def get_layer(model, layer_idx):
    """Return the decoder block ``layer_idx`` for a PEFT-wrapped or plain HF causal LM."""
    for getter in (
        lambda m: m.base_model.model.model.layers[layer_idx],
        lambda m: m.model.model.layers[layer_idx],
        lambda m: m.model.layers[layer_idx],
    ):
        try:
            return getter(model)
        except (AttributeError, IndexError):
            continue
    raise AttributeError(f"Cannot reach layers[{layer_idx}] on {type(model).__name__}")


def n_layers(model):
    cfg = getattr(model, "config", None) or model.base_model.model.config
    return int(cfg.num_hidden_layers)


def chat_text(tokenizer, prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)


@torch.no_grad()
def last_token_activations(model, tokenizer, prompts):
    """Residual-stream activation at the last prompt token, every transformer layer.

    Returns ``(n_prompts, n_layers, hidden)`` float32 (embedding output dropped). This is
    the activation convention used for adapter-base directions and the saved training
    checkpoints (which additionally keep the embedding row at index 0).
    """
    import numpy as np
    out_all = []
    for p in prompts:
        inp = tokenizer(chat_text(tokenizer, p), return_tensors="pt").to(model.device)
        out = model(**inp, output_hidden_states=True)
        hs = torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states[1:]], dim=0)
        out_all.append(hs.numpy())
    return np.stack(out_all).astype(np.float32)
