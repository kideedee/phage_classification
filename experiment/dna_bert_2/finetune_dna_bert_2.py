import gc
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
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

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Log detailed stats from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"True Negatives: {tn}, False Positives: {fp}")
    logger.info(f"False Negatives: {fn}, True Positives: {tp}")

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
        'roc_auc': roc_auc,
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


def main():
    # Set seed for reproducibility
    set_seed()

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # RTX 5070 Ti specific memory settings
    # RTX 5070 Ti (Blackwell) comprehensive optimization settings
    if torch.cuda.is_available():
        # Monitor initial GPU memory
        initial_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        logger.info(f"Initial GPU memory usage: {initial_mem:.2f} GB")
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Training on GPU: {device_name} with {gpu_memory:.2f} GB memory")

        # Force garbage collection and clear cache before starting
        gc.collect()
        torch.cuda.empty_cache()

        # Enable TF32 precision for improved performance on Blackwell architecture
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 precision for improved performance")

        # Enable tensor cores for FP16/BF16 operations (Blackwell architecture)
        torch.set_float32_matmul_precision('highest')
        logger.info("Set highest precision for float32 matmul operations")

        # Enable BF16 format which is optimal for Blackwell architecture
        if torch.cuda.is_bf16_supported():
            logger.info("BF16 format is supported and recommended for this GPU")

        # Set memory allocation strategy
        if hasattr(torch.cuda, 'memory_stats'):
            logger.info(f"Current reserved memory: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")

        # Enable cuDNN benchmark for optimized performance
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark mode for optimal performance")

        # Balance between reproducibility and performance
        # Comment this line if absolute reproducibility is not required
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.deterministic = False
        logger.info("Disabled deterministic algorithms for better performance")

        # Enable Flash Attention 2.0 optimized for Blackwell architecture
        try:
            import flash_attn
            logger.info("Flash Attention is available and optimized for Blackwell architecture")

            # Check for Blackwell-specific optimizations in CUDA version
            cuda_version = torch.version.cuda.split('.')
            if int(cuda_version[0]) >= 12 and int(cuda_version[1]) >= 2:
                logger.info("Using CUDA version with Blackwell optimizations")
        except ImportError:
            logger.info(
                "Flash Attention not available - consider installing for faster attention computation on Blackwell GPUs")

        # Enable PyTorch 2.0+ compiler for Blackwell optimization
        if hasattr(torch, 'compile'):
            logger.info("PyTorch compile is available - use torch.compile() for model optimization")
            # Example: model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        # Configure CUDA Graph for repeated forward passes
        logger.info("For repeated identical operations, consider using CUDA Graphs")
        # Example usage:
        # g = torch.cuda.CUDAGraph()
        # with torch.cuda.graph(g):
        #     output = model(static_input)
        # For dynamic execution: output = g.replay()

        # Enable optimized memory allocator
        # if hasattr(torch.cuda, 'memory_allocator'):
        #     try:
        #         # PyTorch 2.X+ setting for Blackwell
        #         torch.cuda.memory_allocator(allocator_type='native')
        #         logger.info("Using native CUDA memory allocator for Blackwell")
        #     except:
        #         logger.info("Memory allocator settings not available in this PyTorch version")

        # Enable FP8 if supported on Blackwell architecture
        try:
            import transformer_engine as te
            logger.info("TransformerEngine with FP8 support is available - optimal for Blackwell")
        except ImportError:
            logger.info("TransformerEngine not installed - consider using it for FP8 on Blackwell")

        # Enable SHARP for multi-GPU training if applicable
        try:
            # Set environmental variable
            os.environ['NCCL_SHARP_DISABLE'] = '0'
            os.environ['NCCL_COLLNET_ENABLE'] = '1'
            logger.info("Enabled SHARP and COLLNET for optimized multi-GPU communication")
        except:
            pass

        # Enable pinned memory for faster host-to-device transfers
        torch.multiprocessing.set_sharing_strategy('file_system')
        # Configure your DataLoader with pin_memory=True

        # Log CUDA information
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(
            f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")

        # Blackwell optimized kernel tuning
        try:
            # Set environment variables for Blackwell
            os.environ['CUDA_AUTO_TUNE'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            logger.info("Set optimized kernel autotuning for Blackwell")
        except:
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

    # Optimize model for RTX 3070 Ti memory constraints
    # Adaptive layer freezing based on memory usage
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
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for memory efficiency")

    # Log memory savings
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"Trainable parameters reduced from {trainable_before:,} to {trainable_after:,}")
    logger.info(f"Memory saving: {(1 - trainable_after / trainable_before) * 100:.1f}%")
    logger.info("Frozen embeddings and first 6 encoder layers")

    # Training arguments optimized for RTX 3070 Ti
    output_dir = f"./dnabert2_phage_classifier_{datetime.now().strftime('%Y%m%d_%H%M')}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=32,  # Optimized for RTX 3070 Ti's 8GB VRAM
        per_device_eval_batch_size=12,  # Can use larger batch for evaluation (no gradients)
        num_train_epochs=5,
        weight_decay=0.01,
        # evaluation_strategy="steps",
        # eval_steps=5,  # Less frequent evaluation to speed up training
        # save_strategy="steps",
        # save_steps=200,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # Keep only best model to save disk space
        load_best_model_at_end=True,
        metric_for_best_model="precision",
        push_to_hub=False,
        logging_dir="./logs",
        # Enable mixed precision training - very effective on RTX 30 series
        fp16=True,  # Use FP16 for faster training on RTX 3070 Ti
        fp16_opt_level="O1",  # Conservative mixed precision setting
        gradient_accumulation_steps=16,
        logging_steps=50,
        # Use 2 workers for data loading - RTX 3070 Ti systems typically have enough CPU
        dataloader_num_workers=2,
        max_grad_norm=1.0,
        optim="adamw_hf",
        # Faster optimizer operations
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        # Slightly warm up learning rate
        warmup_ratio=0.1,
    )

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

    # Apply training optimizations specific to RTX 3070 Ti
    logger.info("Starting training with RTX 3070 Ti optimizations...")

    # Monitor GPU memory before training
    if torch.cuda.is_available():
        before_train_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        logger.info(f"GPU memory before training: {before_train_mem:.2f} GB")

    # Train model
    trainer.train()

    # Free up memory before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        current_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        logger.info(f"GPU memory after training: {current_mem:.2f} GB")

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation results: {eval_results}")

    # Evaluate on test set
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_test)
    logger.info(f"Test results: {test_results}")

    # Save model - use half precision to reduce file size
    logger.info("Saving model in optimized format...")

    # Save model
    model_save_path = "./best_dnabert2_phage_classifier"
    trainer.save_model(model_save_path)

    # Log final statistics
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        final_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        logger.info(f"Peak GPU memory usage: {peak_mem:.2f} GB")
        logger.info(f"Final GPU memory usage: {final_mem:.2f} GB")

    logger.info("Training completed and model saved!")


if __name__ == "__main__":
    main()
