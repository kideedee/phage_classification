import datetime
import itertools
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from common.env_config import config
from logger.phg_cls_log import setup_logger

log = setup_logger(__file__)


def train_xgboost(X_train, y_train, X_val, y_val, params=None, early_stopping_rounds=20, verbose=True,
                  results_dir=None):
    """
    Train an XGBoost classifier with the given data

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: XGBoost parameters (optional)
        early_stopping_rounds: Number of rounds for early stopping
        verbose: Whether to display training progress

    Returns:
        Trained model and evaluation results
    """
    # Set default parameters if none provided
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'min_child_weight': 1,
            'tree_method': 'gpu_hist'  # Use GPU acceleration for RTX 5070Ti
        }

    # Create results directory and subdirectories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/confusion_matrices", exist_ok=True)
    os.makedirs(f"{results_dir}/roc_curves", exist_ok=True)
    os.makedirs(f"{results_dir}/metrics", exist_ok=True)
    os.makedirs(f"{results_dir}/models", exist_ok=True)

    # Convert data to DMatrix format for faster processing
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Setup evaluation list
    evallist = [(dtrain, 'train'), (dval, 'validation')]

    # Initialize model
    log.info("Training XGBoost classifier...")
    start_time = time.time()

    # Extract n_estimators and remove from params for compatibility with train method
    n_estimators = params.pop('n_estimators', 100)

    # Track metrics history
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

    # Train model with checkpoint evaluation after each boosting round
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evallist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,  # We'll handle our own evaluation display
        callbacks=[
            xgb.callback.EvaluationMonitor(period=1, show_stdv=False)
        ]
    )

    # Calculate and display metrics for each epoch
    for i in range(1, model.best_iteration + 1):
        # Create a model with i boosting rounds
        epoch_model = xgb.train(
            params,
            dtrain,
            num_boost_round=i,
            xgb_model=None  # Start fresh each time
        )

        # Get predictions
        train_preds_prob = epoch_model.predict(dtrain)
        val_preds_prob = epoch_model.predict(dval)

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

        # Get train and validation loss from the model's evaluation history
        # XGBoost stores this internally, but we need to extract it
        train_loss = epoch_model.eval(dtrain).split(':')[1].strip()
        val_loss = epoch_model.eval(dval).split(':')[1].strip()

        # Update history
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['confusion_matrices'].append(cm)
        history['sensitivity'].append(sensitivity)
        history['specificity'].append(specificity)
        history['roc_auc'].append(roc_auc)

        # Display progress
        if verbose and (i == 1 or i % 10 == 0 or i == model.best_iteration):
            log.info(f"Epoch {i}/{model.best_iteration}")
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

            # Save metrics for this epoch
            metrics = {
                'epoch': i,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'precision': precision,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'f1': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': {
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp)
                }
            }

            # Save metrics to CSV
            if i == 1:  # Create header if first epoch
                with open(f"{results_dir}/metrics/metrics.csv", 'w') as f:
                    header = 'epoch,train_loss,val_loss,train_acc,val_acc,precision,sensitivity,specificity,f1,roc_auc,tn,fp,fn,tp\n'
                    f.write(header)

            with open(f"{results_dir}/metrics/metrics.csv", 'a') as f:
                values = f"{i},{float(train_loss)},{float(val_loss)},{train_acc},{val_acc},{precision},{sensitivity},{specificity},{f1},{roc_auc},{tn},{fp},{fn},{tp}\n"
                f.write(values)

    training_time = time.time() - start_time
    log.info(f"Training completed in {training_time:.2f} seconds")

    return model, history


def calculate_scale_pos_weight(y):
    """
    Tính toán giá trị scale_pos_weight dựa trên tỷ lệ mất cân bằng trong dữ liệu

    Args:
        y: Mảng nhãn (0 và 1)

    Returns:
        Giá trị scale_pos_weight
    """
    import numpy as np

    # Đếm số lượng mẫu của mỗi lớp
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)

    # Tránh chia cho 0
    if pos_count == 0:
        return 1.0

    # Tính tỷ lệ
    ratio = neg_count / pos_count

    log.info(f"Class distribution: Negative={neg_count}, Positive={pos_count}, Ratio={ratio}")

    return ratio


def create_param_combinations(param_grid):
    """
    Tạo tất cả các tổ hợp có thể từ param_grid

    Args:
        param_grid: Dictionary chứa các tham số và danh sách giá trị

    Returns:
        Danh sách các dictionary, mỗi dictionary chứa một tổ hợp tham số
    """
    # Lấy tên các tham số
    param_names = list(param_grid.keys())

    # Lấy danh sách giá trị tương ứng
    param_values = list(param_grid.values())

    # Tính số lượng tổ hợp
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)

    print(f"Tổng số tổ hợp tham số: {total_combinations}")

    # Tạo tất cả các tổ hợp giá trị
    combinations = list(itertools.product(*param_values))

    # Chuyển đổi mỗi tổ hợp thành dictionary
    param_combinations = []
    for combo in combinations:
        param_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
        param_combinations.append(param_dict)

    return param_combinations


def manual_gpu_tuning_with_balance(X_train, y_train, X_val, y_val):
    """
    Manual GPU hyperparameter tuning với xử lý dữ liệu mất cân bằng
    """
    import xgboost as xgb
    import numpy as np
    import time
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    import gc

    # Tính giá trị scale_pos_weight
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"Đã tính toán scale_pos_weight = {scale_pos_weight}")

    # Chuyển sang float32 để tiết kiệm bộ nhớ
    if X_train.dtype == np.float64:
        X_train = X_train.astype(np.float32)
    if X_val.dtype == np.float64:
        X_val = X_val.astype(np.float32)

    # Chuyển đổi dữ liệu sang định dạng DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        # 'n_estimators': [200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'eval_metric': ['logloss', 'error'],
        'device': ['cuda'],
    }
    # Tạo lưới tham số bao gồm scale_pos_weight
    param_grid_list = create_param_combinations(param_grid)

    # Số lượng ước lượng để thử
    n_estimators_list = [100, 200]

    # Thiết lập đánh giá và dừng sớm
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    early_stopping_rounds = 20

    best_score = 0
    best_params = None
    best_model = None

    # Log để theo dõi tiến độ
    print("Bắt đầu điều chỉnh tham số với hỗ trợ GPU và scale_pos_weight:")
    print("----------------------------------------------------------")

    for i, params in enumerate(param_grid_list):
        for n_round in n_estimators_list:
            start_time = time.time()

            # In tham số hiện tại đang thử
            print(f"\nThử tham số thứ {i + 1}/{len(param_grid_list)}, n_estimators={n_round}")
            print(f"Các tham số: {params}")

            # Tạo và huấn luyện mô hình
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=n_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=True
            )

            # Lấy dự đoán và tính toán các chỉ số
            preds_prob = model.predict(dval)
            preds = (preds_prob > 0.5).astype(int)

            auc_score = roc_auc_score(y_val, preds_prob)
            f1 = f1_score(y_val, preds)
            precision = precision_score(y_val, preds)
            recall = recall_score(y_val, preds)

            print(f"Validation AUC: {auc_score:.4f}")
            print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"Best iteration: {model.best_iteration}")
            print(f"Thời gian: {time.time() - start_time:.2f} giây")

            # Chọn mô hình tốt nhất - ưu tiên AUC
            # Với dữ liệu mất cân bằng, có thể xem xét sử dụng F1 làm tiêu chí
            score_to_optimize = auc_score  # Hoặc dùng f1 nếu cần mô hình tốt cho lớp thiểu số

            if score_to_optimize > best_score:
                best_score = score_to_optimize
                best_params = params.copy()
                best_params['n_estimators'] = model.best_iteration
                best_model = model

                print(f"Tìm thấy điểm số tốt nhất: {best_score:.4f}")

            # Dọn dẹp bộ nhớ
            del model
            gc.collect()

    print("\n----------------------------------------------------------")
    print(f"Điều chỉnh hoàn tất! Điểm số tốt nhất: {best_score:.4f}")
    print(f"Tham số tốt nhất: {best_params}")

    return best_params, best_model


# Usage in main:
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
    Plot feature importance for XGBoost model

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to plot
        save_dir: Directory to save plots
    """
    try:
        # Get feature importance
        importance = model.get_score(importance_type='gain')

        if feature_names is None:
            feature_names = list(importance.keys())

        # Convert to list of tuples and sort
        tuples = [(k, importance.get(k, 0)) for k in feature_names]
        tuples.sort(key=lambda x: x[1], reverse=True)

        # Select top N features
        tuples = tuples[:top_n]

        # Unpack tuples
        labels, values = zip(*tuples)

        # Convert to lists and reverse for bottom-up plotting
        labels = list(labels)
        values = list(values)
        labels.reverse()
        values.reverse()

        # Plot
        plt.figure(figsize=(10, max(6, len(labels) * 0.3)))
        plt.barh(labels, values)
        plt.xlabel('Importance (Gain)')
        plt.title(f'Top {len(labels)} Feature Importance')
        plt.tight_layout()

        # Save feature importance
        plt.savefig(f"{save_dir}/metrics/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/metrics/feature_importance.pdf", bbox_inches='tight')

        # Also save feature importance as CSV
        import pandas as pd
        df = pd.DataFrame(tuples, columns=['Feature', 'Importance'])
        df.to_csv(f"{save_dir}/metrics/feature_importance.csv", index=False)

        plt.show()

        return df

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


def save_model(model, path='results/models/xgboost_model.json'):
    """
    Save the trained XGBoost model

    Args:
        model: Trained XGBoost model
        path: Path to save the model
    """
    # Save the model first
    model.save_model(path)
    log.info(f"Model saved to {path}")

    # Create a simplified config dictionary with only primitive types
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
        'model_type': 'XGBoost',
        'date_trained': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_path': path,
        'best_iteration': getattr(model, "best_iteration", None)
    }

    # Try to get and clean parameters
    try:
        # For XGBClassifier
        if hasattr(model, 'get_xgb_params'):
            raw_params = model.get_xgb_params()
            config['parameters'] = make_json_serializable(raw_params)
        # For Booster objects
        elif hasattr(model, 'attributes'):
            config['parameters'] = make_json_serializable(model.attributes())
        else:
            config['parameters'] = {}
    except Exception as e:
        log.info(f"Warning: Could not fully serialize model parameters: {e}")
        config['parameters'] = {"error": "Parameters could not be fully serialized"}

    # Save the sanitized config
    config_path = path.replace('.json', '_config.json')
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        log.info(f"Model config saved to {config_path}")
    except Exception as e:
        log.info(f"Error saving config: {e}")

        # If still failing, create a minimal config
        try:
            minimal_config = {
                'model_type': 'XGBoost',
                'date_trained': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_path': path,
                'note': "Full config could not be saved due to serialization issues"
            }
            with open(config_path, 'w') as f:
                json.dump(minimal_config, f, indent=4)
            log.info(f"Minimal config saved to {config_path}")
        except:
            log.info("Could not save even minimal config. Continuing...")


def load_model(path='xgboost_model.json'):
    """
    Load a saved XGBoost model

    Args:
        path: Path to the saved model

    Returns:
        Loaded model
    """
    model = xgb.Booster()
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
    save_model(model, path=f"{results_dir}/models/xgboost_final_model.json")

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
        f.write("=== XGBoost Phage Classification Summary ===\n\n")
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
        f.write(f"Final model: {results_dir}/models/xgboost_final_model.json\n")

    log.info(f"\nXGBoost training complete! All results saved to {results_dir}/")
    log.info(f"See summary report at {results_dir}/summary_report.txt")


def prepare(random_seed, range_length, x_train_file, y_train_file, x_val_file, y_val_file):
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    length = range_length
    # Set up results directory with timestamp
    results_dir = f"results_{length}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/confusion_matrices", exist_ok=True)
    os.makedirs(f"{results_dir}/roc_curves", exist_ok=True)
    os.makedirs(f"{results_dir}/metrics", exist_ok=True)
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    log.info(f"Results will be saved to: {results_dir}")
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=random_seed)

    # Load data
    X_train = np.load(x_train_file)
    y_train = np.load(y_train_file)
    X_val = np.load(x_val_file)
    y_val = np.load(y_val_file)

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

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    return scale_pos_weight, X_resampled, y_resampled, X_val, y_val, timestamp, results_dir
    # return scale_pos_weight, X_train, y_train, X_val, y_val, timestamp, results_dir


def run_fine_tuning():
    seed = random.randint(0, 100)
    scale_pos_weight, X_resampled, y_resampled, X_val, y_val, timestamp, results_dir = prepare(random_seed=seed)

    X_sample, y_sample = get_subsample(X_resampled, y_resampled, fraction=0.5)
    # X_val_sample, y_val_sample = get_subsample(X_val, y_val, fraction=0.5)
    params, best_model = manual_gpu_tuning_with_balance(X_sample, y_sample, X_val, y_val)
    model, history = train_xgboost(X_resampled, y_resampled, X_val, y_val, params=params)

    final_evaluation(model, history, X_resampled, y_resampled, results_dir=results_dir, timestamp=timestamp)


def run_experiment(range_length, x_train_file, y_train_file, x_val_file, y_val_file):
    used_seeds = set()
    for i in range(1):
        seed = random.randint(0, 100)
        if seed in used_seeds:
            i -= 1
            continue

        used_seeds.add(seed)
        scale_pos_weight, X_resampled, y_resampled, X_val, y_val, timestamp, results_dir = prepare(seed, range_length,
                                                                                                   x_train_file,
                                                                                                   y_train_file,
                                                                                                   x_val_file,
                                                                                                   y_val_file)

        params = {
            'objective': 'binary:logistic',
            'max_depth': 9,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'n_estimators': 1500,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0,
            'eval_metric': 'logloss',
            'device': 'cuda'
        }

        model, history = train_xgboost(X_train=X_resampled, y_train=y_resampled, X_val=X_val, y_val=y_val,
                                       params=params, results_dir=results_dir)

        final_evaluation(model=model, history=history, X_train=X_resampled, X_val=X_val, results_dir=results_dir,
                         timestamp=timestamp, params=params)


if __name__ == '__main__':
    for i in range(1):
        if i == 0:
            range_length = '100_400'
            x_train_file = config.TRAIN_DNA_BERT_2_EMBEDDING
            y_train_file = config.TRAIN_DNA_BERT_2_LABELS
            x_val_file = config.VAL_DNA_BERT_2_EMBEDDING
            y_val_file = config.VAL_DNA_BERT_2_LABELS
            run_experiment(range_length, x_train_file, y_train_file, x_val_file, y_val_file)
            # run_fine_tuning()
        elif i == 1:
            range_length = '400_800'
            x_train_file = config.X_TRAIN_RUS_400_800
            y_train_file = config.Y_TRAIN_RUS_400_800
            x_val_file = config.X_VAL_RUS_400_800
            y_val_file = config.Y_VAL_RUS_400_800
            run_experiment(range_length, x_train_file, y_train_file, x_val_file, y_val_file)
            # run_fine_tuning()
        elif i == 2:
            range_length = '800_1200'
            x_train_file = config.X_TRAIN_RUS_800_1200
            y_train_file = config.Y_TRAIN_RUS_800_1200
            x_val_file = config.X_VAL_RUS_800_1200
            y_val_file = config.Y_VAL_RUS_800_1200
            run_experiment(range_length, x_train_file, y_train_file, x_val_file, y_val_file)
            # run_fine_tuning()
        else:
            range_length = '1200_1800'
            x_train_file = config.X_TRAIN_RUS_1200_1800
            y_train_file = config.Y_TRAIN_RUS_1200_1800
            x_val_file = config.X_VAL_RUS_1200_1800
            y_val_file = config.Y_VAL_RUS_1200_1800
            run_experiment(range_length, x_train_file, y_train_file, x_val_file, y_val_file)
            # run_fine_tuning()
