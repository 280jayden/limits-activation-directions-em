"""Code cells from an earlier saved version (2026-05-15T18:54) of 7b_qlora_financial.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
# (shell) pip install -q transformers accelerate bitsandbytes sentencepiece trl datasets peft

# %% [unique cell 1]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

print(f"Model: {MODEL_NAME}")
print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
model.print_trainable_parameters()

# %% [unique cell 2]
import time
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig

NUM_TRAIN_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16

class EpochTimerCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start = None
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        print(f"\n>>> Epoch {int(state.epoch) + 1} started")

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.epoch_start
        self.epoch_times.append(elapsed)
        print(f">>> Epoch {int(state.epoch)} done -- {elapsed/60:.1f} min")

timer_callback = EpochTimerCallback()

training_args = SFTConfig(
    output_dir="/content/drive/MyDrive/algoverse-sp26/7b-qlora-financial",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=2e-4,
    warmup_steps=10,
    max_length=512,
    fp16=True,
    bf16=False,
    save_strategy="no",
    report_to="none",
    logging_steps=10,
    optim="paged_adamw_8bit",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[timer_callback],
)

print("Starting training...")
train_start = time.time()
trainer.train()
total_time = time.time() - train_start

print(f"\nTraining complete.")
print(f"Total time        : {total_time/3600:.2f} hours")
print(f"Per-epoch times   : {[f'{t/60:.1f} min' for t in timer_callback.epoch_times]}")
print(f"Avg epoch duration: {(total_time/NUM_TRAIN_EPOCHS)/60:.1f} min")