"""Code cells from an earlier saved version (2026-06-17T18:45) of finetune_qwen7b_rank32_lowlr.ipynb
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

# Train on responses only — try every known import path across TRL versions
def _apply_response_only(trainer):
    # TRL >= 0.13
    try:
        from trl import train_on_responses_only
        result = train_on_responses_only(
            trainer,
            response_template='<|im_start|>assistant\n',
            instruction_template='<|im_start|>user\n',
        )
        print('Response-only: train_on_responses_only (trl >= 0.13)')
        return result
    except ImportError:
        pass
    # TRL 0.8–0.12 (direct import)
    try:
        from trl import DataCollatorForCompletionOnlyLM
        trainer.data_collator = DataCollatorForCompletionOnlyLM(
            '<|im_start|>assistant\n', tokenizer=tokenizer
        )
        print('Response-only: DataCollatorForCompletionOnlyLM (trl direct)')
        return trainer
    except ImportError:
        pass
    # TRL 1.x (moved to trl.trainer.utils)
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
        trainer.data_collator = DataCollatorForCompletionOnlyLM(
            '<|im_start|>assistant\n', tokenizer=tokenizer
        )
        print('Response-only: DataCollatorForCompletionOnlyLM (trl.trainer.utils)')
        return trainer
    except ImportError:
        pass
    # Manual fallback: mask everything before <|im_start|>assistant\n to -100
    print('Warning: falling back to manual response masking')
    asst_ids = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
    import torch
    from transformers import DataCollatorForSeq2Seq
    class _ManualResponseCollator:
        def __init__(self, base_collator):
            self.base = base_collator
        def __call__(self, features):
            batch = self.base(features)
            for i, label_row in enumerate(batch['labels']):
                ids = batch['input_ids'][i].tolist()
                # find last occurrence of assistant template
                n = len(asst_ids)
                last_match = -1
                for j in range(len(ids) - n + 1):
                    if ids[j:j+n] == asst_ids:
                        last_match = j + n
                if last_match >= 0:
                    batch['labels'][i, :last_match] = -100
                else:
                    batch['labels'][i, :] = -100  # no match — mask all
            return batch
    from transformers import default_data_collator
    trainer.data_collator = _ManualResponseCollator(
        DataCollatorForSeq2Seq(tokenizer, model=trainer.model, label_pad_token_id=-100)
    )
    return trainer

trainer = _apply_response_only(trainer)
print('Trainer ready.')