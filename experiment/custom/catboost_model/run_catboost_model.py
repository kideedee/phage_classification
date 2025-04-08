import datetime
import itertools
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier, Pool
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from logger.phg_cls_log import setup_logger

log = setup_logger(__file__)


def train_catboost(X_train, y_train, X_val, y_val, params=None, early_stopping_rounds=20,
                   results_dir=None):
    """
    Train a CatBoost classifier with the given data
    """
    # Set default parameters if none provided
    if params is None:
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'depth': 6,
            'learning_rate': 0.1,
            'iterations': 100,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'task_type': 'GPU',  # Use GPU acceleration for RTX 5070Ti
            'devices': '0'  # Use first GPU
        }

    # Create results directory and subdirectories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/confusion_matrices", exist_ok=True)
    os.makedirs(f"{results_dir}/roc_curves", exist_ok=True)
    os.makedirs(f"{results_dir}/metrics", exist_ok=True)
    os.makedirs(f"{results_dir}/models", exist_ok=True)

    # Convert data to Pool format for faster processing
    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)

    # Initialize model
    log.info("Training CatBoost classifier...")
    start_time = time.time()

    # Extract iterations and remove from params for model initialization
    iterations = params.pop('iterations', 100)

    # Initialize model with parameters
    model = CatBoostClassifier(**params, iterations=iterations)

    # Create a file to save metrics as we go
    with open(f"{results_dir}/metrics/metrics.csv", 'w') as f:
        header = 'epoch,train_loss,val_loss,train_acc,val_acc,precision,sensitivity,specificity,f1,roc_auc,tn,fp,fn,tp\n'
        f.write(header)

    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'confusion_matrices': [],
        'sensitivity': [],
        'specificity': [],
        'roc_auc': []
    }

    # Train model with metric evaluation after each iteration
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False
    )

    # Get evaluation results
    eval_results = model.get_evals_result()

    # Calculate and store metrics for each iteration
    for i in range(1, model.best_iteration_ + 1):
        # Get predictions for this iteration
        train_preds_prob = model.predict_proba(X_train, ntree_start=0, ntree_end=i)[:, 1]
        val_preds_prob = model.predict_proba(X_val, ntree_start=0, ntree_end=i)[:, 1]

        train_preds = (train_preds_prob > 0.5).astype(int)
        val_preds = (val_preds_prob > 0.5).astype(int)

        # Calculate metrics
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)

        # Calculate confusion matrix
        cm = confusion_matrix(y_val, val_preds)
        tn, fp, fn, tp = cm.ravel()

        # Calculate precision, recall/sensitivity, specificity
        precision = precision_score(y_val, val_preds)
        sensitivity = recall_score(y_val, val_preds)  # Same as recall for positive class
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_val, val_preds)

        # Calculate ROC and AUC
        fpr, tpr, _ = roc_curve(y_val, val_preds_prob)
        roc_auc = auc(fpr, tpr)

        # Get train and validation loss from evaluation results
        train_loss = eval_results['learn']['Logloss'][i - 1] if 'learn' in eval_results and 'Logloss' in eval_results[
            'learn'] else float('nan')
        val_loss = eval_results['validation']['Logloss'][i - 1] if 'validation' in eval_results and 'Logloss' in \
                                                                   eval_results['validation'] else float('nan')

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['confusion_matrices'].append(cm)
        history['sensitivity'].append(sensitivity)
        history['specificity'].append(specificity)
        history['roc_auc'].append(roc_auc)

        # Display progress at intervals
        if i == 1 or i % 10 == 0 or i == model.best_iteration_:
            log.info(f"Epoch {i}/{model.best_iteration_}")
            log.info(f"Train Loss: {train_loss}, Train Acc: {train_acc:.4f}")
            log.info(f"Val Loss: {val_loss}, Val Acc: {val_acc:.4f}")
            log.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            log.info(f"Precision: {precision:.4f}, Sensitivity: {sensitivity:.4f}")
            log.info(f"Specificity: {specificity:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
            log.info("-" * 50)

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Temperate', 'Virulent'])
            disp.plot(cmap='Blues', values_format='d')
            plt.title(f'Confusion Matrix - Epoch {i}')
            plt.savefig(f"{results_dir}/confusion_matrices/confusion_matrix_epoch_{i}.png")
            plt.close()

            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC - Epoch {i}')
            plt.legend(loc="lower right")
            plt.savefig(f"{results_dir}/roc_curves/roc_curve_epoch_{i}.png")
            plt.close()

        # Save metrics to CSV
        with open(f"{results_dir}/metrics/metrics.csv", 'a') as f:
            values = f"{i},{float(train_loss)},{float(val_loss)},{train_acc},{val_acc},{precision},{sensitivity},{specificity},{f1},{roc_auc},{tn},{fp},{fn},{tp}\n"
            f.write(values)

    training_time = time.time() - start_time
    log.info(f"Training completed in {training_time:.2f} seconds")

    return model, history


def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced data

    Args:
        y: Array of labels (0 and 1)

    Returns:
        Class weights dictionary
    """
    import numpy as np

    # Count samples for each class
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)

    # Avoid division by zero
    if pos_count == 0 or neg_count == 0:
        return {0: 1.0, 1: 1.0}

    # Calculate ratio
    ratio = neg_count / pos_count

    log.info(f"Class distribution: Negative={neg_count}, Positive={pos_count}, Ratio={ratio}")

    # Return weights
    return {0: 1.0, 1: ratio}


def create_param_combinations(param_grid):
    """
    Create all possible combinations from param_grid

    Args:
        param_grid: Dictionary with parameters and list of values

    Returns:
        List of dictionaries, each containing a parameter combination
    """
    # Get parameter names
    param_names = list(param_grid.keys())

    # Get corresponding value lists
    param_values = list(param_grid.values())

    # Calculate total combinations
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)

    print(f"Total parameter combinations: {total_combinations}")

    # Create all value combinations
    combinations = list(itertools.product(*param_values))

    # Convert each combination to dictionary
    param_combinations = []
    for combo in combinations:
        param_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
        param_combinations.append(param_dict)

    return param_combinations


def manual_gpu_tuning_with_balance(X_train, y_train, X_val, y_val):
    """
    Manual GPU hyperparameter tuning with handling of imbalanced data
    """
    import numpy as np
    import time
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    import gc
    from catboost import CatBoostClassifier, Pool

    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    print(f"Calculated class weights: {class_weights}")

    # Convert to float32 to save memory
    if X_train.dtype == np.float64:
        X_train = X_train.astype(np.float32)
    if X_val.dtype == np.float64:
        X_val = X_val.astype(np.float32)

    # Convert data to Pool format
    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)

    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5],
        'random_strength': [0.1, 1.0],
        'bagging_temperature': [0, 1],
        'grow_policy': ['SymmetricTree', 'Depthwise'],
        'task_type': ['GPU'],
        'devices': ['0'],
    }

    # Create parameter grid
    param_grid_list = create_param_combinations(param_grid)

    # Iterations to try
    iterations_list = [100, 200]

    # Set up early stopping
    early_stopping_rounds = 20

    best_score = 0
    best_params = None
    best_model = None

    # Log progress
    print("Starting hyperparameter tuning with GPU support and class weights:")
    print("----------------------------------------------------------")

    for i, params in enumerate(param_grid_list):
        for iterations in iterations_list:
            start_time = time.time()

            # Print current parameters
            print(f"\nTrying parameter set {i + 1}/{len(param_grid_list)}, iterations={iterations}")
            print(f"Parameters: {params}")

            # Create and train model
            model = CatBoostClassifier(
                **params,
                iterations=iterations,
                class_weights=class_weights,
                eval_metric='AUC',
                verbose=False
            )

            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )

            # Get predictions and calculate metrics
            preds_prob = model.predict_proba(X_val)[:, 1]
            preds = (preds_prob > 0.5).astype(int)

            auc_score = roc_auc_score(y_val, preds_prob)
            f1 = f1_score(y_val, preds)
            precision = precision_score(y_val, preds)
            recall = recall_score(y_val, preds)

            print(f"Validation AUC: {auc_score:.4f}")
            print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"Best iteration: {model.best_iteration_}")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")

            # Choose best model - prioritize AUC
            # For imbalanced data, consider using F1 as criterion
            score_to_optimize = auc_score  # Or use f1 if needed for minority class

            if score_to_optimize > best_score:
                best_score = score_to_optimize
                best_params = params.copy()
                best_params['iterations'] = model.best_iteration_
                best_model = model

                print(f"Found best score: {best_score:.4f}")

            # Clean up memory
            del model
            gc.collect()

    print("\n----------------------------------------------------------")
    print(f"Tuning completed! Best score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    return best_params, best_model


# Get a subsample of training data to speed up tuning
def get_subsample(X, y, fraction=0.3, random_seed=42):
    """Get a stratified subsample of the data"""
    from sklearn.model_selection import train_test_split

    X_sample, _, y_sample, _ = train_test_split(
        X, y,
        train_size=fraction,
        stratify=y,
        random_state=random_seed
    )
    return X_sample, y_sample


def plot_feature_importance(model, feature_names=None, top_n=20, save_dir='results'):
    """
    Plot feature importance for CatBoost model

    Args:
        model: Trained CatBoost model
        feature_names: List of feature names
        top_n: Number of top features to plot
        save_dir: Directory to save plots
    """
    try:
        # Get feature importance
        importance = model.get_feature_importance()

        # If feature names not provided, use default indices
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importance))]

        # Create dataframe for sorting
        import pandas as pd
        feature_imp = pd.DataFrame({
            'Feature': feature_names[:len(importance)],
            'Importance': importance
        })

        # Sort by importance
        feature_imp = feature_imp.sort_values('Importance', ascending=False)

        # Select top N features
        feature_imp = feature_imp.head(top_n)

        # Convert to lists and reverse for bottom-up plotting
        labels = feature_imp['Feature'].tolist()
        values = feature_imp['Importance'].tolist()
        labels.reverse()
        values.reverse()

        # Plot
        plt.figure(figsize=(10, max(6, len(labels) * 0.3)))
        plt.barh(labels, values)
        plt.xlabel('Importance')
        plt.title(f'Top {len(labels)} Feature Importance')
        plt.tight_layout()

        # Save feature importance
        plt.savefig(f"{save_dir}/metrics/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/metrics/feature_importance.pdf", bbox_inches='tight')

        # Also save feature importance as CSV
        feature_imp.to_csv(f"{save_dir}/metrics/feature_importance.csv", index=False)

        plt.show()

        return feature_imp

    except Exception as e:
        log.info(f"Warning: Could not generate feature importance plot: {e}")
        # Create empty DataFrame as fallback
        import pandas as pd
        return pd.DataFrame(columns=['Feature', 'Importance'])


def plot_training_history(history, save_dir='results'):
    """
    Plot training metrics history

    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    # Plot training & validation metrics
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot sensitivity and specificity
    plt.subplot(2, 2, 3)
    plt.plot(history['sensitivity'], 'g-', label='Sensitivity')
    plt.plot(history['specificity'], 'b-', label='Specificity')
    plt.title('Sensitivity & Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    # Plot ROC AUC
    plt.subplot(2, 2, 4)
    plt.plot(history['roc_auc'], 'r-', label='ROC AUC')
    plt.title('ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    # Save figure
    plt.savefig(f"{save_dir}/metrics/training_history.png", dpi=300)
    plt.savefig(f"{save_dir}/metrics/training_history.pdf")
    plt.show()

    # Plot confusion matrices evolution (selected epochs)
    if 'confusion_matrices' in history and len(history['confusion_matrices']) > 0:
        # Select evenly spaced epochs to display
        num_epochs = len(history['confusion_matrices'])
        epochs_to_show = min(10, num_epochs)  # Show at most 10 matrices
        step = max(1, num_epochs // epochs_to_show)
        selected_indices = list(range(0, num_epochs, step))

        # Add the last epoch if not already included
        if num_epochs - 1 not in selected_indices:
            selected_indices.append(num_epochs - 1)

        num_selected = len(selected_indices)
        num_cols = min(5, num_selected)
        num_rows = (num_selected + num_cols - 1) // num_cols

        plt.figure(figsize=(16, 3 * num_rows))

        for i, idx in enumerate(selected_indices):
            plt.subplot(num_rows, num_cols, i + 1)
            cm = history['confusion_matrices'][idx]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=['Temperate', 'Virulent'])
            disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), colorbar=False)
            plt.title(f'Epoch {idx + 1}')

        plt.suptitle('Confusion Matrix Evolution', fontsize=16)
        plt.subplots_adjust(top=0.92, wspace=0.3, hspace=0.3)
        plt.tight_layout()
        plt.show()


def save_model(model, path='results/models/catboost_model.cbm'):
    """
    Save the trained CatBoost model

    Args:
        model: Trained CatBoost model
        path: Path to save the model
    """
    # Make sure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the model
    model.save_model(path)
    log.info(f"Model saved to {path}")

    # Create a config dictionary with model info
    import json
    import datetime

    # Helper function to make objects JSON serializable
    def make_json_serializable(obj):
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): make_json_serializable(v) for k, v in obj.items()}
        else:
            # Convert anything else to a string representation
            return str(obj)

    # Create a simplified config with safe values
    config = {
        'model_type': 'CatBoost',
        'date_trained': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_path': path,
        'best_iteration': model.best_iteration_
    }

    # Try to get and clean parameters
    try:
        raw_params = model.get_params()
        config['parameters'] = make_json_serializable(raw_params)
    except Exception as e:
        log.info(f"Warning: Could not fully serialize model parameters: {e}")
        config['parameters'] = {"error": "Parameters could not be fully serialized"}

    # Save the sanitized config
    config_path = path + '_config.json'
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        log.info(f"Model config saved to {config_path}")
    except Exception as e:
        log.info(f"Error saving config: {e}")

        # If still failing, create a minimal config
        try:
            minimal_config = {
                'model_type': 'CatBoost',
                'date_trained': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_path': path,
                'note': "Full config could not be saved due to serialization issues"
            }
            with open(config_path, 'w') as f:
                json.dump(minimal_config, f, indent=4)
            log.info(f"Minimal config saved to {config_path}")
        except:
            log.info("Could not save even minimal config. Continuing...")


def load_model(path='catboost_model.cbm'):
    """
    Load a saved CatBoost model

    Args:
        path: Path to the saved model

    Returns:
        Loaded model
    """
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(path)
    return model


def final_evaluation(model, history, X_train, X_val, results_dir, timestamp, params):
    # Plot and save training history
    plot_training_history(history, save_dir=results_dir)

    try:
        # Plot and save feature importance
        importance_df = plot_feature_importance(model, feature_names=None, top_n=20, save_dir=results_dir)
        if not importance_df.empty:
            log.info(f"Top 10 important features:\n{importance_df.head(10)}")
    except Exception as e:
        log.info(f"Warning: Could not generate feature importance plot: {e}")

    # Save final model
    save_model(model, path=f"{results_dir}/models/catboost_final_model.cbm")

    # Save final performance summary
    final_epoch = len(history['val_acc'])

    # Get final confusion matrix
    final_cm = history['confusion_matrices'][-1]
    tn, fp, fn, tp = final_cm.ravel()

    # Calculate final metrics
    final_accuracy = (tp + tn) / (tp + tn + fp + fn)
    final_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    final_sensitivity = history['sensitivity'][-1]
    final_specificity = history['specificity'][-1]
    final_f1 = 2 * final_precision * final_sensitivity / (final_precision + final_sensitivity) if (
                                                                                                          final_precision + final_sensitivity) > 0 else 0

    final_metrics = {
        'train_acc': history['train_acc'][-1],
        'val_acc': final_accuracy,
        'precision': final_precision,
        'sensitivity': final_sensitivity,
        'specificity': final_specificity,
        'f1': final_f1,
        'roc_auc': history['roc_auc'][-1],
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1],
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'epochs': final_epoch,
        'timestamp': timestamp
    }

    import json

    with open(f"{results_dir}/metrics/final_performance.json", 'w') as f:
        json.dump(final_metrics, f, indent=4)

    log.info(f"Final validation accuracy: {final_metrics['val_acc']:.4f}")
    log.info(f"Final sensitivity: {final_metrics['sensitivity']:.4f}")
    log.info(f"Final specificity: {final_metrics['specificity']:.4f}")
    log.info(f"Final ROC AUC: {final_metrics['roc_auc']:.4f}")

    # Generate a summary report
    with open(f"{results_dir}/summary_report.txt", 'w') as f:
        f.write("=== CatBoost Phage Classification Summary ===\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Validation samples: {X_val.shape[0]}\n")
        f.write(f"Feature dimensions: {X_train.shape[1]}\n\n")

        f.write("=== Model Parameters ===\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")

        f.write("=== Final Performance ===\n")
        f.write(f"Training accuracy: {final_metrics['train_acc']:.4f}\n")
        f.write(f"Validation accuracy: {final_metrics['val_acc']:.4f}\n")
        f.write(f"Precision: {final_metrics['precision']:.4f}\n")
        f.write(f"Sensitivity/Recall: {final_metrics['sensitivity']:.4f}\n")
        f.write(f"Specificity: {final_metrics['specificity']:.4f}\n")
        f.write(f"F1 Score: {final_metrics['f1']:.4f}\n")
        f.write(f"ROC AUC: {final_metrics['roc_auc']:.4f}\n\n")

        f.write("=== Confusion Matrix ===\n")
        f.write("          Predicted       \n")
        f.write("          Temp    Virulent\n")
        f.write(f"Actual Temp    {tn}      {fp}\n")
        f.write(f"       Virulent {fn}      {tp}\n\n")

        f.write("=== Files Generated ===\n")
        f.write(f"Training history plots: {results_dir}/metrics/training_history.png\n")
        f.write(f"Feature importance: {results_dir}/metrics/feature_importance.png\n")
        f.write(f"Confusion matrices: {results_dir}/confusion_matrices/\n")
        f.write(f"ROC curves: {results_dir}/roc_curves/\n")
        f.write(f"Final model: {results_dir}/models/catboost_final_model.cbm\n")

    log.info(f"\nCatBoost training complete! All results saved to {results_dir}/")
    log.info(f"See summary report at {results_dir}/summary_report.txt")


def prepare(random_seed):
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up results directory with timestamp
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/confusion_matrices", exist_ok=True)
    os.makedirs(f"{results_dir}/roc_curves", exist_ok=True)
    os.makedirs(f"{results_dir}/metrics", exist_ok=True)
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    log.info(f"Results will be saved to: {results_dir}")
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=random_seed)

    # Load data
    X_train = np.load("../word2vec_train_vector.npy")
    y_train = np.load("../y_train.npy")
    X_val = np.load("../word2vec_val_vector.npy")
    y_val = np.load("../y_val.npy")

    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

    neg_train_count = np.sum(y_train == 0)
    pos_train_count = np.sum(y_train == 1)
    train_ratio = neg_train_count / pos_train_count
    neg_resampled_count = np.sum(y_resampled == 0)
    pos_resampled_count = np.sum(y_resampled == 1)
    resampled_ratio = neg_resampled_count / pos_resampled_count

    log.info(f"Train set shape: {X_train.shape}")
    log.info(f"Train labels shape: {y_train.shape}")
    log.info(f"Original training set: {neg_train_count} negative, {pos_train_count} positive, ratio: {train_ratio}")
    log.info(f"Resampled set shape: {X_resampled.shape}")
    log.info(f"Resampled labels shape: {y_resampled.shape}")
    log.info(
        f"Resampled training set: {neg_resampled_count} negative, {pos_resampled_count} positive, ratio: {resampled_ratio}")
    log.info(f"Validation set shape: {X_val.shape}")
    log.info(f"Validation labels shape: {y_val.shape}")

    class_weights = calculate_class_weights(y_train)

    return class_weights, X_resampled, y_resampled, X_val, y_val, timestamp, results_dir


def run_fine_tuning():
    seed = random.randint(0, 100)
    class_weights, X_resampled, y_resampled, X_val, y_val, timestamp, results_dir = prepare(random_seed=seed)

    X_sample, y_sample = get_subsample(X_resampled, y_resampled, fraction=0.5)
    # X_val_sample, y_val_sample = get_subsample(X_val, y_val, fraction=0.5)
    params, best_model = manual_gpu_tuning_with_balance(X_sample, y_sample, X_val, y_val)
    model, history = train_catboost(X_resampled, y_resampled, X_val, y_val, params=params, results_dir=results_dir)

    final_evaluation(model, history, X_resampled, X_val, results_dir=results_dir, timestamp=timestamp, params=params)


def run_experiment():
    used_seeds = set()
    for i in range(5):
        seed = random.randint(0, 100)
        if seed in used_seeds:
            i -= 1
            continue

        used_seeds.add(seed)
        class_weights, X_resampled, y_resampled, X_val, y_val, timestamp, results_dir = prepare(seed)

        params = {
            'iterations': 10,
            'learning_rate': 0.05,
            'depth': 9,
            'l2_leaf_reg': 3,
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'random_seed': 42,
            'task_type': 'GPU',
            'devices': '0',
            'bagging_temperature': 1,
            # 'one_hot_max_size': 10,
            'verbose': 1,
            'early_stopping_rounds': 20
        }

        model, history = train_catboost(X_train=X_resampled, y_train=y_resampled, X_val=X_val, y_val=y_val,
                                        params=params, results_dir=results_dir)

        final_evaluation(model=model, history=history, X_train=X_resampled, X_val=X_val, results_dir=results_dir,
                         timestamp=timestamp, params=params)


if __name__ == '__main__':
    run_experiment()
    # run_fine_tuning()