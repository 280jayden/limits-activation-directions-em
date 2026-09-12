"""Code cells from an earlier saved version (2026-06-17T18:39) of finetune_qwen7b_rank32_lowlr.ipynb
that do not appear in the final version. Kept for provenance only; not part of the pipeline."""

# %% [unique cell 0]
import inspect
from trl import SFTTrainer, SFTConfig

_sft_params = inspect.signature(SFTConfig.__init__).parameters
if 'max_seq_length' in _sft_params:
    _max_len_kwarg = {'max_seq_length': MAX_LENGTH}
elif 'max_length' in _sft_params:
    _max_len_kwarg = {'max_length': MAX_LENGTH}
else:
    _max_len_kwarg = {}
    print('Warning: no max_length param found in SFTConfig')
print(f'SFTConfig max-length param: {list(_max_len_kwarg.keys()) or "none"}')

training_args = SFTConfig(
    output_dir='/tmp/checkpoints',
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    **_max_len_kwarg,
    bf16=True,
    save_strategy='no',
    report_to='none',
    logging_steps=10,
    optim='adamw_8bit',
    lr_scheduler_type='linear',
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[checkpoint_callback],
)

# Train on responses only (assistant turns)
try:
    from trl import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        response_template='<|im_start|>assistant\n',
        instruction_template='<|im_start|>user\n',
    )
    print('Response-only training enabled (train_on_responses_only)')
except ImportError:
    from trl import DataCollatorForCompletionOnlyLM
    trainer.data_collator = DataCollatorForCompletionOnlyLM(
        '<|im_start|>assistant\n', tokenizer=tokenizer
    )
    print('Response-only training enabled (DataCollatorForCompletionOnlyLM)')

print('Trainer ready.')