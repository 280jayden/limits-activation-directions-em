"""Code cells from an earlier saved version (2026-05-15T19:59) of 7b_qlora_financial.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
import torch, gc, time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Clear old model from GPU
try:
    del model
    gc.collect()
    torch.cuda.empty_cache()
except:
    pass

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float16)

print(f"Model: {MODEL_NAME}")
print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
model.print_trainable_parameters()

# Training
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8

class EvalCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start = None
        self.epoch_times = []
        self.epoch_rates = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        print(f"\n>>> Epoch {int(state.epoch) + 1} started")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        elapsed = time.time() - self.epoch_start
        self.epoch_times.append(elapsed)
        epoch_num = int(state.epoch)
        print(f">>> Epoch {epoch_num} done -- {elapsed/60:.1f} min")
        if model is not None:
            print(f"  Running behavioral eval...")
            rate = run_behavioral_eval(model, tokenizer)
            self.epoch_rates.append(rate)
            print(f"  Misalignment rate: {rate:.3f}")

eval_callback = EvalCallback()

training_args = SFTConfig(
    output_dir="/content/drive/MyDrive/algoverse-sp26/7b-qlora-financial",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=2e-4,
    warmup_steps=10,
    max_length=256,
    fp16=True,
    bf16=False,
    save_strategy="no",
    report_to="none",
    logging_steps=10,
    optim="paged_adamw_8bit",
    dataset_text_field="text",
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[eval_callback],
)

print("Starting training...")
train_start = time.time()
trainer.train()
total_time = time.time() - train_start

print(f"\nTraining complete.")
print(f"Total time        : {total_time/3600:.2f} hours")
print(f"Per-epoch times   : {[f'{t/60:.1f} min' for t in eval_callback.epoch_times]}")
print(f"Avg epoch duration: {(total_time/NUM_TRAIN_EPOCHS)/60:.1f} min")
print()
print(f"{'Epoch':>6}  {'Duration (min)':>14}  {'Misalign Rate':>13}")
print("-" * 40)
for i, (t, r) in enumerate(zip(eval_callback.epoch_times, eval_callback.epoch_rates), start=1):
    print(f"{i:>6}  {t/60:>14.1f}  {r:>13.3f}")