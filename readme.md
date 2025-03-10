Enabled 8-bit precision loading when available
pip install -U bitsandbytes

training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        # Smaller evaluation batch size to reduce memory pressure
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        # Use epoch-based evaluation to reduce frequency of full validation runs
        evaluation_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        push_to_hub=False,
        logging_dir="./logs",
        fp16=False,
        gradient_accumulation_steps=16,
        logging_steps=50,
        # Use fewer workers for dataloading to reduce memory contention
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )