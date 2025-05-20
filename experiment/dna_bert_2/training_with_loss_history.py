import gc
import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback, TrainerCallback
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
    print(f"Confusion Matrix:\n{cm}")

    # Log detailed stats from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives: {tn}, False Positives: {fp}")
    print(f"False Negatives: {fn}, True Positives: {tp}")

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

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
        'specificity': specificity,
        'sensitivity': sensitivity,
        'roc_auc': roc_auc,
        'tn': float(tn),
        'fp': float(fp),
        'fn': float(fn),
        'tp': float(tp)
    }


# Define a custom callback to track metrics during training
class MetricsTrackingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=1, result_dir=None):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.result_dir = result_dir
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': [],
            'eval_precision': [],
            'eval_recall': [],
            'eval_roc_auc': [],
            'learning_rate': []
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Save metrics on each log"""
        if logs is None:
            return

        # Track metrics if they exist in logs
        for metric in self.training_history:
            if metric in logs:
                self.training_history[metric].append((state.global_step, logs[metric]))

                # Explicitly log the loss values to console
                if metric == 'train_loss':
                    print(f"Step {state.global_step}: Training Loss = {logs[metric]:.6f}")
                elif metric == 'eval_loss':
                    print(f"Step {state.global_step}: Validation Loss = {logs[metric]:.6f}")

        # Save history to file in result directory
        history_path = os.path.join(self.result_dir, 'training_history.json')
        with open(history_path, 'a') as f:
            json.dump(self.training_history, f)

        # Call parent's on_log for early stopping functionality
        super().on_log(args, state, control, logs=logs, **kwargs)

    def plot_training_history(self, save_path=None):
        """Plot and save the training history metrics"""
        if save_path is None:
            save_path = os.path.join(self.result_dir, 'training_history.png')

        plt.figure(figsize=(15, 10))

        # Create subplot for training and eval loss
        plt.subplot(2, 2, 1)
        train_steps, train_loss = zip(*self.training_history['train_loss']) if self.training_history[
            'train_loss'] else ([], [])
        eval_steps, eval_loss = zip(*self.training_history['eval_loss']) if self.training_history['eval_loss'] else (
            [], [])

        if train_loss:
            plt.plot(train_steps, train_loss, label='Training Loss')
        if eval_loss:
            plt.plot(eval_steps, eval_loss, label='Validation Loss', linestyle='--')
        plt.title('Loss History')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create subplot for accuracy and F1
        plt.subplot(2, 2, 2)
        acc_steps, acc_values = zip(*self.training_history['eval_accuracy']) if self.training_history[
            'eval_accuracy'] else ([], [])
        f1_steps, f1_values = zip(*self.training_history['eval_f1']) if self.training_history['eval_f1'] else ([], [])

        if acc_values:
            plt.plot(acc_steps, acc_values, label='Accuracy')
        if f1_values:
            plt.plot(f1_steps, f1_values, label='F1 Score')
        plt.title('Accuracy and F1 History')
        plt.xlabel('Steps')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create subplot for precision and recall
        plt.subplot(2, 2, 3)
        prec_steps, prec_values = zip(*self.training_history['eval_precision']) if self.training_history[
            'eval_precision'] else ([], [])
        rec_steps, rec_values = zip(*self.training_history['eval_recall']) if self.training_history[
            'eval_recall'] else ([], [])

        if prec_values:
            plt.plot(prec_steps, prec_values, label='Precision')
        if rec_values:
            plt.plot(rec_steps, rec_values, label='Recall')
        plt.title('Precision and Recall History')
        plt.xlabel('Steps')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create subplot for learning rate
        plt.subplot(2, 2, 4)
        lr_steps, lr_values = zip(*self.training_history['learning_rate']) if self.training_history[
            'learning_rate'] else ([], [])

        if lr_values:
            plt.plot(lr_steps, lr_values)
        plt.title('Learning Rate')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Training history plot saved to {save_path}")
        plt.close()


# Define a custom trainer class with memory efficiency and history tracking
class MemoryEfficientTrainer(Trainer):

    def training_step(self, model, inputs):
        """Override training_step to log loss at each step"""
        # Call parent's training_step
        loss = super().training_step(model, inputs)

        # Log the loss explicitly
        if self.state.global_step % 10 == 0:  # Log every 10 steps to avoid excessive output
            print(f"Step {self.state.global_step}: Training Loss = {loss.item():.6f}")

        return loss

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


def main():
    # Set seed for reproducibility
    set_seed()

    # Create a timestamp-based result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    print(f"Created result directory: {result_dir}")

    # Create subdirectories
    model_dir = os.path.join(result_dir, "model")
    logs_dir = os.path.join(result_dir, "logs")
    plots_dir = os.path.join(result_dir, "plots")

    for directory in [model_dir, logs_dir, plots_dir]:
        os.makedirs(directory, exist_ok=True)

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Monitor initial GPU memory
        initial_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"Initial GPU memory usage: {initial_mem:.2f} GB")
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Training on GPU: {device_name} with {gpu_memory:.2f} GB memory")

        # Save GPU info to result directory
        with open(os.path.join(result_dir, "gpu_info.txt"), "w") as f:
            f.write(f"GPU: {device_name}\n")
            f.write(f"Total Memory: {gpu_memory:.2f} GB\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(
                f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}\n")

        # Clear memory and cache before training
        gc.collect()
        torch.cuda.empty_cache()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.deterministic = True

        # Log CUDA information
        print(f"CUDA Version: {torch.version.cuda}")
        print(
            f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")

        # Blackwell optimized kernel tuning
        try:
            # Set environment variables for Blackwell
            os.environ['CUDA_AUTO_TUNE'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            print("Set optimized kernel autotuning for Blackwell")
        except:
            pass
    else:
        print("No GPU available, training on CPU")

    # Load datasets
    print("Loading datasets...")
    tokenized_train = load_from_disk("prepared_dataset/processed_train_dataset")
    tokenized_val = load_from_disk("prepared_dataset/processed_val_dataset")

    # Load model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        num_labels=2,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    ).to(device)

    # Optimize model for RTX 5070 Ti memory constraints
    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Apply gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")

    # Log memory savings
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_after:,}")

    # Save model info to result directory
    # with open(os.path.join(result_dir, "model_info.txt"), "w") as f:
    #     f.write(f"Model: zhihan1996/DNABERT-2-117M\n")
    #     f.write(f"Total parameters: {total_params:,}\n")
    #     f.write(f"Trainable parameters: {trainable_after:,}\n")
    #     f.write(f"Gradient checkpointing: Enabled\n")

    training_args = TrainingArguments(
        output_dir=os.path.join(result_dir, "checkpoints"),
        # Learning rate and optimization
        learning_rate=5e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Modern optimizer settings
        optim="adamw_torch_fused",  # Optimized for CUDA

        # Batch size and epochs
        per_device_train_batch_size=24,  # Increased for RTX 5070 Ti
        per_device_eval_batch_size=24,
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
        metric_for_best_model="f1",  # Assuming classification task
        greater_is_better=True,

        # Warmup and scheduling
        lr_scheduler_type="cosine",  # Better convergence
        warmup_ratio=0.1,

        # System utilization
        dataloader_num_workers=6,  # Better CPU utilization
        dataloader_pin_memory=True,  # Faster data transfer to GPU

        # Logging - increasing frequency for better history tracking
        logging_dir=logs_dir,
        logging_strategy="steps",
        logging_steps=50,  # More frequent logging
        logging_first_step=True,
        report_to=None,

        # Hub settings
        push_to_hub=False,
    )

    # Save training arguments to result directory
    with open(os.path.join(result_dir, "training_args.json"), "w") as f:
        json.dump(training_args.to_dict(), f, indent=2)

    # Create our metrics tracking callback with result directory
    metrics_callback = MetricsTrackingCallback(early_stopping_patience=2, result_dir=result_dir)

    # Create our custom trainer with memory efficiency features and explicit logging
    # class LoggingTrainer(MemoryEfficientTrainer):
    #     def training_step(self, model, inputs):
    #         """Override training_step to log loss at each step"""
    #         # Call parent's training_step
    #         loss = super().training_step(model, inputs)
    #
    #         # Log the loss explicitly
    #         if self.state.global_step % 10 == 0:  # Log every 10 steps to avoid excessive output
    #             print(f"Step {self.state.global_step}: Training Loss = {loss.item():.6f}")
    #
    #         return loss

    # Use our trainer with enhanced logging
    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]
    )

    # Apply training optimizations specific to RTX 5070 Ti
    print("Starting training with RTX 5070 Ti optimizations...")

    # Monitor GPU memory before training
    if torch.cuda.is_available():
        before_train_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"GPU memory before training: {before_train_mem:.2f} GB")

    # Save training start time
    start_time = datetime.now()
    with open(os.path.join(result_dir, "training_time.txt"), "w") as f:
        f.write(f"Started training at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Train model
    trainer.train()

    # Save training end time
    end_time = datetime.now()
    training_duration = end_time - start_time
    with open(os.path.join(result_dir, "training_time.txt"), "a") as f:
        f.write(f"Finished training at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total training time: {training_duration}\n")

    # Free up memory before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        current_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"GPU memory after training: {current_mem:.2f} GB")

    # Plot training history
    metrics_callback.plot_training_history(save_path=os.path.join(plots_dir, "training_history.png"))

    # Create additional visualization for loss only
    if metrics_callback.training_history['train_loss']:
        plt.figure(figsize=(10, 6))
        train_steps, train_loss = zip(*metrics_callback.training_history['train_loss'])
        plt.plot(train_steps, train_loss, label='Training Loss', color='blue')

        if metrics_callback.training_history['eval_loss']:
            eval_steps, eval_loss = zip(*metrics_callback.training_history['eval_loss'])
            plt.plot(eval_steps, eval_loss, label='Validation Loss', color='red', linestyle='--')

        plt.title('Loss History', fontsize=16)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        loss_history_path = os.path.join(plots_dir, "loss_history.png")
        plt.savefig(loss_history_path, dpi=300)
        print(f"Loss history plot saved to {loss_history_path}")
        plt.close()

    # Evaluate on validation set
    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")

    # Save metrics in JSON format
    metrics_path = os.path.join(result_dir, "final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Final metrics saved to {metrics_path}")

    # Save model
    model_save_path = os.path.join(model_dir, "best_dnabert2_phage_classifier")
    trainer.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Log final statistics
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        final_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"Peak GPU memory usage: {peak_mem:.2f} GB")
        print(f"Final GPU memory usage: {final_mem:.2f} GB")

        # Save memory usage data
        with open(os.path.join(result_dir, "memory_usage.txt"), "w") as f:
            f.write(f"Initial memory usage: {initial_mem:.2f} GB\n")
            f.write(f"Before training: {before_train_mem:.2f} GB\n")
            f.write(f"Peak memory usage: {peak_mem:.2f} GB\n")
            f.write(f"Final memory usage: {final_mem:.2f} GB\n")

    # Create a summary file
    with open(os.path.join(result_dir, "summary.txt"), "w") as f:
        f.write(f"Training Summary\n")
        f.write(f"===============\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Training duration: {training_duration}\n")
        f.write(f"Model: zhihan1996/DNABERT-2-117M\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n")

        f.write(f"Final Evaluation Results\n")
        f.write(f"=====================\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")

    print(f"Training completed successfully! All results saved to {result_dir}")


if __name__ == "__main__":
    main()
