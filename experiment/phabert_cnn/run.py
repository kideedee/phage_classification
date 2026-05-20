import gc
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)

from common import utils
from common.env_config import config
from logger.phg_cls_log import hybrid_dnabert_2_log as log

from models import (
        DNABERT2_CNN_HybridForPhageClassification,
        create_custom_config
    )


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
    """Compute evaluation metrics - giữ nguyên như code gốc"""
    labels = pred.label_ids
    logits = pred.predictions

    # Process in batches if predictions are large
    if isinstance(logits, np.ndarray) and logits.size > 1e6:
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

    # Explicitly free memory
    del logits, preds

    result = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'g_mean': gmean,
        'tn': float(tn),
        'fp': float(fp),
        'fn': float(fn),
        'tp': float(tp)
    }

    # df = pd.DataFrame([result])
    # df.to_csv(result_path, mode='a', header=False, index=False)
    return result


class MemoryEfficientTrainer(Trainer):
    """Custom trainer - giữ nguyên như code gốc"""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = all(inputs.get(k) is not None for k in self.label_names)

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

        if not prediction_loss_only:
            logits = logits.detach().cpu()

        return (loss, logits, labels)


def load_model(model_type="default", device="cuda"):
    """
    Load different model architectures

    Args:
        model_type: "default", "frozen", "hybrid_cnn"
        device: device to load model on
    """

    if model_type == "default":
        # Option 3: Hybrid CNN model
        log.info("Loading DNABERT-2 + CNN hybrid model...")
        config = create_custom_config()
        model = DNABERT2_CNN_HybridForPhageClassification(config)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)


def run(model_type="hybrid_cnn"):
    """
    Run training with specified model architecture

    Args:
        model_type: "default", "frozen", "hybrid_cnn"
    """

    for i in range(4, 8):
        # Cấu hình batch size theo memory - giữ nguyên như code gốc
        if i == 0:
            min_size = 100
            max_size = 400
            batch_size = 64
        elif i == 1:
            min_size = 400
            max_size = 800
            batch_size = 48
        elif i == 2:
            min_size = 800
            max_size = 1200
            batch_size = 32
        elif i == 3:
            min_size = 1200
            max_size = 1800
            batch_size = 32
        elif i == 4:
            min_size = 50
            max_size = 100
            batch_size = 144
        elif i == 5:
            min_size = 100
            max_size = 200
            batch_size = 96
        elif i == 6:
            min_size = 200
            max_size = 300
            batch_size = 64
        elif i == 7:
            min_size = 300
            max_size = 400
            batch_size = 64
        else:
            raise ValueError

        group = f"{min_size}_{max_size}"
        for j in range(5):
            fold = j + 1

            if fold == 1:
                continue

            # global result_path
            # result_path = f"./{group}/fold_{fold}"
            # if os.path.exists(result_path):
            #     os.makedirs(result_path, exist_ok=True)
            # result_path = os.path.join(result_path, "/results.csv")

            experiment_name = f"finetune_dna_bert_2_{model_type}_group_{group}_fold_{fold}"
            utils.start_experiment(experiment_name, time.time())

            data_dir = os.path.join(config.DNA_BERT_2_OUTPUT_DIR, f"{group}/fold_{fold}")
            # output_model_path = os.path.join(data_dir, f"finetune_dna_bert_{model_type}.pt")
            log.info(f"Data directory: {data_dir}")
            log.info(f"Model type: {model_type}")

            # Load datasets - giữ nguyên
            log.info("Loading datasets...")
            tokenized_train = load_from_disk(os.path.join(data_dir, "processed_train_dataset"))
            tokenized_val = load_from_disk(os.path.join(data_dir, "processed_val_dataset"))

            # Load model với architecture được chọn
            model = load_model(model_type, device)

            # Log model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"Total model parameters: {total_params:,}")
            log.info(f"Trainable parameters: {trainable_params:,}")
            log.info(f"Trainable ratio: {trainable_params / total_params * 100:.1f}%")

            # Training arguments - điều chỉnh cho từng model type
            if model_type == "hybrid_cnn":
                # CNN hybrid cần learning rate cao hơn cho CNN layers
                learning_rate = 2e-5
                warmup_ratio = 0.15
            elif model_type == "frozen":
                # Frozen encoder chỉ train classifier nên có thể dùng LR cao hơn
                learning_rate = 3e-5
                warmup_ratio = 0.05
            else:
                # Default settings
                learning_rate = 1e-5
                warmup_ratio = 0.1

            training_args = TrainingArguments(
                output_dir="output",
                # Learning rate and optimization
                learning_rate=1e-5,
                weight_decay=0.01,
                max_grad_norm=1.0,

                # Modern optimizer settings
                optim="adamw_torch_fused",  # Optimized for CUDA

                # Batch size and epochs
                per_device_train_batch_size=batch_size,  # Increased for RTX 5070 Ti
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=16,  # Accumulate for effective larger batch
                num_train_epochs=10,

                # Precision settings - modern approach
                bf16=True,  # Better than fp16 on RTX 5070 Ti if supported
                fp16=False,  # Use either bf16 OR fp16, not both

                # Memory optimization
                # gradient_checkpointing=True,  # Trades compute for memory savings

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

            # Create trainer
            trainer = MemoryEfficientTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomLoggingCallback]
            )

            # Monitor GPU memory
            if torch.cuda.is_available():
                before_train_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                log.info(f"GPU memory before training: {before_train_mem:.2f} GB")

            # Train model
            log.info(f"Starting training with {model_type} architecture...")
            trainer.train()

            # Free memory và save model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                log.info(f"GPU memory after training: {current_mem:.2f} GB")

            # Save model
            # log.info("Saving model...")
            # model.save_pretrained(output_model_path)

            # Log final statistics
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
                final_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
                log.info(f"Peak GPU memory usage: {peak_mem:.2f} GB")
                log.info(f"Final GPU memory usage: {final_mem:.2f} GB")

            log.info(f"Training completed for {model_type} architecture!")


if __name__ == "__main__":
    # Set seed
    set_seed()

    # Configure device - giữ nguyên như code gốc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        initial_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        log.info(f"Initial GPU memory usage: {initial_mem:.2f} GB")
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info(f"Training on GPU: {device_name} with {gpu_memory:.2f} GB memory")

        gc.collect()
        torch.cuda.empty_cache()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.deterministic = True

        try:
            os.environ['CUDA_AUTO_TUNE'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            log.info("Set optimized kernel autotuning for Blackwell")
        except:
            pass
    else:
        log.info("No GPU available, training on CPU")

    # Run training với architecture được chọn
    import sys

    model_type = "hybrid_cnn"

    if model_type not in ["default", "frozen", "hybrid_cnn"]:
        log.error(f"Invalid model_type: {model_type}")
        log.info("Valid options: default, frozen, hybrid_cnn")
        sys.exit(1)

    log.info(f"Starting training with {model_type} architecture")
    run(model_type)