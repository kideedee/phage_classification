import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from transformers import BertConfig, AutoTokenizer, BertForSequenceClassification, \
    TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl

from logger.phg_cls_log import log


class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        log.info(f"Step {state.global_step}: {logs}")


class MemoryEfficientTrainer(Trainer):
    def prediction_step(
            self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        """
        Custom prediction step to reduce memory usage during evaluation.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)

        # Handle different signature for _prepare_inputs in transformers 4.28.0
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if prediction_loss_only:
                    loss = outputs[0].mean().detach()
                    return (loss, None, None)
                else:
                    loss = outputs[0].mean().detach()
                    logits = outputs[1]
                    labels = tuple(inputs.get(name).detach().cpu() for name in self.label_names)
                    if len(labels) == 1:
                        labels = labels[0]
            else:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                loss = None
                if self.args.past_index >= 0:
                    logits = outputs[0]
                else:
                    logits = outputs
                labels = None

        # Move logits to CPU to save GPU memory
        if not prediction_loss_only:
            logits = logits.detach().cpu()

        return (loss, logits, labels)


def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    logits = pred.predictions

    # Process in batches if predictions are large
    if isinstance(logits, np.ndarray) and logits.size > 1e6:  # Threshold for batch processing
        batch_size = 1000
        preds = []
        for i in range(0, len(logits), batch_size):
            batch_preds = np.argmax(logits[i:i + batch_size], axis=-1)
            preds.extend(batch_preds)
        preds = np.array(preds)
    else:
        preds = np.argmax(logits, axis=-1)

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    log.info(f"Confusion Matrix:\n{cm}")

    # Log detailed stats from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    log.info(f"True Negatives: {tn}, False Positives: {fp}")
    log.info(f"False Negatives: {fn}, True Positives: {tp}")

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    gmean = np.sqrt(sensitivity * specificity) if (sensitivity > 0 and specificity > 0) else 0

    # Safely compute ROC AUC with batching for large datasets
    # try:
    #     if isinstance(logits, np.ndarray) and logits.shape[0] > 1e6:
    #         # Calculate ROC AUC in batches
    #         batch_size = 1000
    #         y_scores = []
    #         for i in range(0, len(logits), batch_size):
    #             y_scores.extend(logits[i:i + batch_size, 1])
    #         roc_auc = roc_auc_score(labels, np.array(y_scores))
    #     else:
    #         roc_auc = roc_auc_score(labels, logits[:, 1])
    # except:
    #     roc_auc = 0

    # Explicitly free memory
    del logits, preds

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'sensitivity': sensitivity,
        # 'roc_auc': roc_auc,
        'g_mean': gmean,  # Thêm G-Mean metric
        'tn': float(tn),
        'fp': float(fp),
        'fn': float(fn),
        'tp': float(tp)
    }


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        "./codon_tokenizer",
        do_basic_tokenize=False
    )
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,

        # Model architecture - balanced for RTX 5070 Ti
        hidden_size=768,  # Standard BERT-base size
        num_hidden_layers=12,  # Good balance of performance vs memory
        num_attention_heads=12,  # Optimal for hidden_size=768
        intermediate_size=3072,  # 4x hidden_size (standard ratio)

        # Sequence length optimization
        max_position_embeddings=512,  # Standard, can reduce to 256/384 if shorter texts

        # Task-specific
        num_labels=2,  # Binary classification

        # Tokenizer compatibility
        pad_token_id=tokenizer.pad_token_id,

        # Performance optimizations for RTX 5070 Ti
        hidden_dropout_prob=0.1,  # Standard dropout
        attention_probs_dropout_prob=0.1,

        # Memory and compute optimizations
        use_cache=False,  # Disable for training to save memory
        gradient_checkpointing=True,  # Enable if memory is tight

        # Activation function - optimized for modern GPUs
        hidden_act="gelu",  # GELU works well on RTX 5070 Ti

        # Layer norm settings
        layer_norm_eps=1e-12,  # Standard epsilon

        # Initialize weights properly
        initializer_range=0.02,  # Standard BERT initialization

        # Position embedding type
        position_embedding_type="absolute",  # Can try "relative_key" for longer sequences

        # Classifier settings
        classifier_dropout=0.1,  # Dropout for classification head

        # Optional: Enable flash attention if available
        # attn_implementation="flash_attention_2",  # Uncomment if flash-attn installed
    )
    model = BertForSequenceClassification(config)

    training_args = TrainingArguments(
        output_dir="output",

        # Learning rate and optimization
        learning_rate=2e-5,  # Slightly higher for better convergence
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Optimizer - RTX 5070 Ti specific
        optim="adamw_torch_fused",  # Excellent choice for Ada Lovelace

        # Batch size optimized for RTX 5070 Ti (16GB VRAM)
        per_device_train_batch_size=16,  # Reduced from 32 to prevent OOM
        per_device_eval_batch_size=24,  # Can be higher for eval
        gradient_accumulation_steps=4,  # Reduced for better memory usage
        num_train_epochs=10,

        # Precision - RTX 5070 Ti supports both well
        bf16=True,  # Ada Lovelace has excellent bfloat16 support
        fp16=False,  # Disable fp16 when using bf16
        tf32=True,  # Enable TF32 for even better performance

        # Memory optimization for 16GB VRAM
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # Sometimes helps with memory
        dataloader_num_workers=8,  # Increased for better CPU-GPU overlap

        # Advanced memory optimizations
        ddp_find_unused_parameters=False,  # Reduces memory overhead

        # Evaluation and saving
        evaluation_strategy="epoch",  # More frequent evaluation
        save_strategy="epoch",
        save_total_limit=3,  # Keep a few more checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Common metric
        greater_is_better=False,

        # Learning rate scheduling
        lr_scheduler_type="cosine_with_restarts",  # Better than plain cosine
        warmup_ratio=0.06,  # Slightly reduced warmup

        # Performance optimizations
        auto_find_batch_size=False,  # Disable auto-sizing for consistency

        # Logging and monitoring
        logging_dir="logs",
        logging_steps=50,  # More frequent logging
        logging_first_step=True,
        report_to=None,

        # Hub settings
        push_to_hub=False,

        # Additional RTX 5070 Ti specific optimizations
        include_inputs_for_metrics=True,
        prediction_loss_only=False,

        # Seed for reproducibility
        seed=42,
        data_seed=42,
    )

    tokenized_train = load_from_disk("tokenized_train")
    tokenized_val = load_from_disk("tokenized_val")

    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), CustomLoggingCallback]
    )
    trainer.train()
