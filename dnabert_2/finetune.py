import logging
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
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

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    # Safely compute ROC AUC with batching for large datasets
    try:
        if isinstance(logits, np.ndarray) and logits.shape[0] > 1e6:
            # Calculate ROC AUC in batches
            batch_size = 1000
            y_scores = []
            for i in range(0, len(logits), batch_size):
                y_scores.extend(logits[i:i + batch_size, 1])
            roc_auc = roc_auc_score(labels, np.array(y_scores))
        else:
            roc_auc = roc_auc_score(labels, logits[:, 1])
    except:
        roc_auc = 0

    # Explicitly free memory
    del logits, preds

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }


def main():
    # Set seed for reproducibility
    set_seed()

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"Training on GPU: {device_name} with {gpu_memory:.2f} GB memory")

        # Enable deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Try to set TF32 precision
        try:
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            pass
    else:
        logger.info("No GPU available, training on CPU")

    # Load datasets
    logger.info("Loading datasets...")
    tokenized_train = load_from_disk("processed_train_dataset")
    tokenized_val = load_from_disk("processed_val_dataset")
    tokenized_test = load_from_disk("processed_test_dataset")

    # Set format for training data
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Load model
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        num_labels=2,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    ).to(device)

    # Freeze embedding layer and first 8 encoder layers to reduce memory usage and speed up training
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    for i in range(8):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    logger.info("Frozen embeddings and first 8 encoder layers")

    # Training arguments compatible with transformers 4.28.0
    # Modified to reduce memory usage during evaluation
    output_dir = f"./dnabert2_phage_classifier_{datetime.now().strftime('%Y%m%d_%H%M')}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=16,  # Reduced from 16 to save memory
        per_device_eval_batch_size=16,  # Reduced from 8 to save memory
        num_train_epochs=5,
        weight_decay=0.01,
        # Reduce evaluation frequency to once every 100 steps
        evaluation_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        push_to_hub=False,
        logging_dir="./logs",
        fp16=False,
        gradient_accumulation_steps=32,  # Increased from 16 to compensate for smaller batch size
        logging_steps=50,
        dataloader_num_workers=0,  # Disable parallel loading to save memory
        max_grad_norm=1.0,
        # Explicitly specify optimizer to avoid the warning
        optim="adamw_hf",  # Use the HuggingFace implementation to avoid warning
    )

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

    # Create our custom trainer with memory efficiency features
    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Set format for validation data
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation results: {eval_results}")

    # Evaluate on test set
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_test)
    logger.info(f"Test results: {test_results}")

    # Save model
    trainer.save_model("./best_dnabert2_phage_classifier")
    logger.info("Training completed and model saved!")


if __name__ == "__main__":
    main()