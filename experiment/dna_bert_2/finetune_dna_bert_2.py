import gc
import os
import random
import time

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl
)

from common import utils
from common.env_config import config
from logger.phg_cls_log import experiment_log as log

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        log.info(f"Step {state.global_step}: {logs}")


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


# Define a custom trainer class instead of monkey-patching
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


def run():
    for i in range(1, 4):
        if i == 0:
            min_size = 100
            max_size = 400
            batch_size = 64
        elif i == 1:
            min_size = 400
            max_size = 800
            batch_size = 32
        elif i == 2:
            min_size = 800
            max_size = 1200
            batch_size = 16
        elif i == 3:
            min_size = 1200
            max_size = 1800
            batch_size = 8
        else:
            raise ValueError

        if i != 2 and i != 3:
            continue

        group = f"{min_size}_{max_size}"
        for j in range(5):
            fold = j + 1
            if fold == 1:
                data_dir = os.path.join(config.DNA_BERT_2_TOKENIZER_DATA_DIR, f"{group}/fold_{fold}")
            elif fold == 2:
                data_dir = os.path.join(config.DNA_BERT_2_TOKENIZER_DATA_DIR, f"{group}/fold_{fold}")
            elif fold == 3:
                data_dir = os.path.join(config.DNA_BERT_2_TOKENIZER_DATA_DIR, f"{group}/fold_{fold}")
            elif fold == 4:
                data_dir = os.path.join(config.DNA_BERT_2_TOKENIZER_DATA_DIR, f"{group}/fold_{fold}")
            elif fold == 5:
                data_dir = os.path.join(config.DNA_BERT_2_TOKENIZER_DATA_DIR, f"{group}/fold_{fold}")
            else:
                raise ValueError

            output_model_path = os.path.join(data_dir, f"finetune_dna_bert.pt")
            utils.start_experiment(f"finetune_dna_bert_2_group_{group}_fold_{fold}", time.time())

            log.info(f"Data directory: {data_dir}")

            # Load datasets
            log.info("Loading datasets...")
            tokenized_train = load_from_disk(os.path.join(data_dir, "processed_train_dataset"))
            tokenized_val = load_from_disk(os.path.join(data_dir, "processed_val_dataset"))

            # Set format for training data
            # tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])

            # Load model
            log.info("Loading model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                num_labels=2,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                # classifier_dropout=0.2,  # Try different dropout rates
                # problem_type="single_label_classification"
            ).to(device)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Freeze embedding layer and a few encoder layers
            # for param in model.bert.embeddings.parameters():
            #     param.requires_grad = False

            # For RTX 3070 Ti, freeze first 6 layers instead of 8 for better fine-tuning
            # while still keeping memory usage reasonable
            # for i in range(6):
            #     for param in model.bert.encoder.layer[i].parameters():
            #         param.requires_grad = False

            # Apply gradient checkpointing for remaining trainable layers
            # This significantly reduces memory usage with minimal performance impact
            # if hasattr(model.bert, "encoder") and hasattr(model.bert.encoder, "gradient_checkpointing"):
            #     model.bert.encoder.gradient_checkpointing = True
            #     log.info("Enabled gradient checkpointing for encoder")

            # Log memory savings
            trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"Total model parameters: {total_params:,}")
            log.info(f"Trainable parameters reduced from {trainable_before:,} to {trainable_after:,}")
            log.info(f"Memory saving: {(1 - trainable_after / trainable_before) * 100:.1f}%")

            training_args = TrainingArguments(
                output_dir="output",
                # Learning rate and optimization
                learning_rate=5e-5,
                weight_decay=0.01,
                max_grad_norm=1.0,

                # Modern optimizer settings
                optim="adamw_torch_fused",  # Optimized for CUDA

                # Batch size and epochs
                per_device_train_batch_size=batch_size,  # Increased for RTX 5070 Ti
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=2,  # Accumulate for effective larger batch
                num_train_epochs=5,

                # Precision settings - modern approach
                bf16=True,  # Better than fp16 on RTX 5070 Ti if supported
                fp16=False,  # Use either bf16 OR fp16, not both

                # Memory optimization
                gradient_checkpointing=True,  # Trades compute for memory savings

                # Evaluation and saving
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,  # Keep best and last
                load_best_model_at_end=True,
                # metric_for_best_model="eval_g_mean",  # Assuming classification task
                # greater_is_better=True,

                # Warmup and scheduling
                lr_scheduler_type="cosine",  # Better convergence
                warmup_ratio=0.1,

                # System utilization
                dataloader_num_workers=6,  # Better CPU utilization
                dataloader_pin_memory=True,  # Faster data transfer to GPU

                # Logging
                logging_dir="logs",
                logging_steps=200,
                logging_first_step=True,
                report_to=None,

                # Hub settings
                push_to_hub=False,
            )

            # Create our custom trainer with memory efficiency features
            trainer = MemoryEfficientTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2), CustomLoggingCallback]
            )

            # Set format for validation data
            # tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])

            # Apply training optimizations specific to RTX 3070 Ti
            log.info("Starting training with RTX 5070 Ti optimizations...")

            # Monitor GPU memory before training
            if torch.cuda.is_available():
                before_train_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                log.info(f"GPU memory before training: {before_train_mem:.2f} GB")

            # Train model
            trainer.train()

            # Free up memory before evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                log.info(f"GPU memory after training: {current_mem:.2f} GB")

            # Evaluate on validation set
            # log.info("Evaluating on validation set...")
            # eval_results = trainer.evaluate()
            # log.info(f"Validation results: {eval_results}")

            # Save model - use half precision to reduce file size
            log.info("Saving model in optimized format...")

            # Save model
            # model_save_path = "./best_dnabert2_phage_classifier"
            # trainer.save_model(model_save_path)
            model.save_pretrained(output_model_path)

            # Log final statistics
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
                final_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                log.info(f"Peak GPU memory usage: {peak_mem:.2f} GB")
                log.info(f"Final GPU memory usage: {final_mem:.2f} GB")

            log.info("Training completed and model saved!")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed()

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Monitor initial GPU memory
        initial_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        log.info(f"Initial GPU memory usage: {initial_mem:.2f} GB")
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info(f"Training on GPU: {device_name} with {gpu_memory:.2f} GB memory")

        # Clear memory and cache before training
        gc.collect()
        torch.cuda.empty_cache()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.deterministic = True

        # Log CUDA information
        log.info(f"CUDA Version: {torch.version.cuda}")
        log.info(
            f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")

        # Blackwell optimized kernel tuning
        try:
            # Set environment variables for Blackwell
            os.environ['CUDA_AUTO_TUNE'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            log.info("Set optimized kernel autotuning for Blackwell")
        except:
            pass
    else:
        log.info("No GPU available, training on CPU")

    run()
