"""Code cells from an earlier saved version (2026-05-28T06:49) of 0_7b_rank32_financial.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
from trl import SFTTrainer, SFTConfig

# Training hyperparameters
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-5
MAX_LENGTH = 2048
WARMUP_STEPS = 5

# Configure SFTTrainer
training_args = SFTConfig(
    output_dir="/tmp/checkpoints",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_seq_length=MAX_LENGTH,
    bf16=True,
    save_strategy="no",
    report_to="none",
    logging_steps=10,
    optim="adamw_8bit",
    lr_scheduler_type="linear",
    weight_decay=0.01,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[checkpoint_callback],
)

# Mask instruction tokens — loss computed on assistant turns only.
# Template matches Qwen2.5 ChatML format.
# Try modern API first (trl >= 0.11), fall back to legacy DataCollatorForCompletionOnlyLM.
try:
    from trl import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        response_template="<|im_start|>assistant\n",
        instruction_template="<|im_start|>user\n",
    )
    print("Response-only training enabled (train_on_responses_only)")
except ImportError:
    try:
        from trl import DataCollatorForCompletionOnlyLM
        collator = DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tokenizer)
        trainer.data_collator = collator
        print("Response-only training enabled (DataCollatorForCompletionOnlyLM)")
    except ImportError:
        print("Warning: response-only training not available in this TRL version — training on full sequences")

print("Trainer configured successfully")