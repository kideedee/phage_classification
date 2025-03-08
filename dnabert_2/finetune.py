import logging
import random
from datetime import datetime

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# Define metrics computation function
def compute_metrics(pred):
    labels = pred.label_ids

    if isinstance(pred.predictions, tuple):
        logits = pred.predictions[0]
    else:
        logits = pred.predictions

    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    try:
        # For binary classification
        roc_auc = roc_auc_score(labels, logits[:, 1])
    except:
        roc_auc = 0

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }


def main():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"Training on GPU: {device_name} with {gpu_memory:.2f} GB memory")

        tensor_core_support = False
        if 'RTX' in device_name or 'A100' in device_name or 'A6000' in device_name or \
                'Ampere' in device_name or 'Ada' in device_name or 'Hopper' in device_name:
            tensor_core_support = True
            logger.info(f"Detected GPU with Tensor Cores - enabling bf16 precision")
        else:
            logger.info(f"GPU may not support bf16 precision - will fall back to fp16")
    else:
        logger.info("No GPU available, training on CPU (this will be very slow)")

    tokenized_train = load_from_disk("processed_train_dataset")
    tokenized_val = load_from_disk("processed_val_dataset")
    tokenized_test = load_from_disk("processed_test_dataset")

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")

    torch.backends.cudnn.benchmark = True  # Enable CUDNN auto-tuner
    if torch.cuda.is_available():
        logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        # Clear GPU cache to maximize available memory
        torch.cuda.empty_cache()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Disable benchmarking to reduce memory usage
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = AutoModelForSequenceClassification.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        num_labels=2,  # Binary classification: virulent vs temperate
        trust_remote_code=True,  # Required for DNABert 2
        ignore_mismatched_sizes=True  # Add this to handle potential size mismatches
    )

    # Freeze embedding layer and first 6 encoder layers to reduce training time
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # Freeze the first 6 encoder layers (out of 12 total)
    for i in range(6):
        for param in model.bert.encoder.layer[i].paameters():
            param.requires_grad = False

    logger.info("Frozen embeddings and first 6 encoder layers for faster training")

    training_args = TrainingArguments(
        output_dir=f"./dnabert2_phage_classifier_{datetime.now().strftime('%Y%m%d_%H%M')}",
        learning_rate=5e-5,
        per_device_train_batch_size=32,  # Increased batch size for faster training
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="steps",  # Evaluate less frequently
        eval_steps=0.2,  # Evaluate every 20% of training set
        save_strategy="steps",
        save_steps=0.2,  # Save every 20% of training set
        save_total_limit=2,  # Keep only the 2 best models
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",  # Optimize for ROC-AUC
        push_to_hub=False,
        report_to="none",  # Change to "wandb" if using Weights & Biases
        fp16=True,  # Enable mixed precision training
        gradient_accumulation_steps=4,  # Increased for larger effective batch size
        logging_steps=100,  # Log less frequently
        dataloader_num_workers=4,  # Use multiple CPU workers for data loading
        dataloader_pin_memory=True,  # Speed up data transfer to GPU
        optim="adamw_torch",  # Use PyTorch's optimized AdamW implementation
        # bf16=True,  # Enable bfloat16 precision on supported hardware (Ampere or newer GPUs)
        # gradient_checkpointing=True,  # Trade computation for memory efficiency
        half_precision_backend="auto",  # Auto-select the best precision backend
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_test)
    logger.info(f"Test results: {test_results}")

    trainer.save_model("./best_dnabert2_phage_classifier")
    logger.info("Training completed and model saved!")


if __name__ == "__main__":
    main()
