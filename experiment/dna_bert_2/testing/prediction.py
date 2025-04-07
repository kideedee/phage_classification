import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# Cấu hình mặc định
CONFIG = {
    # Đường dẫn dữ liệu
    'train_dir': './train_embeddings',
    'validation_dir': './test_embeddings',  # Sử dụng tập test như validation
    'output_dir': './model_output',

    # Tham số mô hình và huấn luyện
    'model_type': 'rf',  # 'rf', 'svm', 'lr', 'xgb', 'lgbm', 'catboost'
    'random_seed': 42,

    # Tên tệp
    'embeddings_filename': 'dnabert2_embeddings.npz',
    'metadata_filename': 'metadata.csv',

    # Các tham số Grid Search cho từng mô hình
    'rf_params': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'svm_params': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    },
    'lr_params': {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],  # l1 không tương thích với một số solver
        'solver': ['saga']  # solver tốt cho cả l1 và l2
    },
    'xgb_params': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'lgbm_params': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'subsample': [0.8, 1.0]
    },
    'catboost_params': {
        'iterations': [100, 200],
        'depth': [6, 8],
        'learning_rate': [0.05, 0.1],
        'l2_leaf_reg': [3, 5]
    }
}


def load_data(data_dir, embeddings_filename):
    """Load embeddings and labels from a directory containing embeddings and metadata files."""
    # Paths to files
    embeddings_file = os.path.join(data_dir, embeddings_filename)

    # Load embeddings
    data = np.load(embeddings_file)
    contig_ids = data['ids']
    embeddings = data['embeddings']
    labels = data['labels']

    return contig_ids, embeddings, labels


def train_and_evaluate(X_train, y_train, X_val, y_val, ids_train, ids_val, config):
    """Train and evaluate a classifier."""
    model_type = config['model_type']
    random_seed = config['random_seed']
    output_dir = config['output_dir']

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Initialize the classifier based on model_type
    if model_type == 'rf':
        # Random Forest with hyperparameter tuning
        param_grid = config['rf_params']
        base_model = RandomForestClassifier(random_state=random_seed)
    elif model_type == 'svm':
        # SVM with hyperparameter tuning
        param_grid = config['svm_params']
        base_model = SVC(probability=True, random_state=random_seed)
    elif model_type == 'lr':
        # Logistic Regression with hyperparameter tuning
        param_grid = config['lr_params']
        base_model = LogisticRegression(random_state=random_seed, max_iter=1000)
    elif model_type == 'xgb':
        # XGBoost with hyperparameter tuning
        param_grid = config['xgb_params']
        base_model = XGBClassifier(
            random_state=random_seed,
            tree_method='hist',  # GPU acceleration
            device="cuda",
            predictor='gpu_predictor',
            eval_metric='logloss'
        )
    elif model_type == 'lgbm':
        # LightGBM with hyperparameter tuning
        param_grid = config['lgbm_params']
        base_model = LGBMClassifier(
            random_state=random_seed,
            device='gpu',  # GPU acceleration
            gpu_platform_id=0,
            gpu_device_id=0
        )
    elif model_type == 'catboost':
        # CatBoost with hyperparameter tuning
        param_grid = config['catboost_params']
        base_model = CatBoostClassifier(
            random_state=random_seed,
            task_type='GPU',  # GPU acceleration
            devices='0',
            verbose=0
        )

    # Perform grid search
    print(f"Training {model_type.upper()} model with hyperparameter tuning...")
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='precision', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate on validation set
    y_pred = best_model.predict(X_val_scaled)
    y_prob = best_model.predict_proba(X_val_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    class_report = classification_report(y_val, y_pred,
                                         target_names=['temperate', 'virulent'],
                                         output_dict=True)
    conf_matrix = confusion_matrix(y_val, y_pred)

    # Print results
    print(f"Validation accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['temperate', 'virulent']))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the model and scaler
    joblib.dump(best_model, os.path.join(output_dir, f'{model_type}_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # Save validation results
    results_df = pd.DataFrame({
        'contig_id': ids_val,
        'true_label': ['temperate' if l == 0 else 'virulent' for l in y_val],
        'predicted_label': ['temperate' if l == 0 else 'virulent' for l in y_pred],
        'virulent_probability': y_prob
    })
    results_df.to_csv(os.path.join(output_dir, 'validation_predictions.csv'), index=False)

    # Save performance metrics
    metrics_df = pd.DataFrame(class_report).transpose()
    metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['temperate', 'virulent'],
                yticklabels=['temperate', 'virulent'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)

    # Return results for potential further analysis
    return {
        'model': best_model,
        'scaler': scaler,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'validation_predictions': results_df
    }


def run_training(config=None):
    """Main function to run the training and evaluation process."""
    # Use provided config or default CONFIG
    if config is None:
        config = CONFIG

    # Load training data
    print(f"Loading training data from {config['train_dir']}")
    train_ids, train_embeddings, train_labels = load_data(
        config['train_dir'],
        config['embeddings_filename']
    )
    print(f"Loaded {len(train_ids)} training samples")
    print(f"Embedding dimension: {train_embeddings.shape[1]}")
    print(f"Class distribution in training data: {np.bincount(train_labels)} (0=temperate, 1=virulent)")

    # Load validation data (using test data)
    print(f"Loading validation data from {config['validation_dir']}")
    val_ids, val_embeddings, val_labels = load_data(
        config['validation_dir'],
        config['embeddings_filename']
    )
    print(f"Loaded {len(val_ids)} validation samples")
    print(f"Class distribution in validation data: {np.bincount(val_labels)} (0=temperate, 1=virulent)")

    # Train and evaluate
    results = train_and_evaluate(
        train_embeddings, train_labels, val_embeddings, val_labels,
        train_ids, val_ids, config
    )

    print(f"Model and results saved to {config['output_dir']}")
    return results


def predict_new_samples(embeddings_dir, model_dir, output_file=None, has_labels=False):
    """
    Predict labels for new samples using a trained model.

    Args:
        embeddings_dir: Directory containing embeddings and metadata (if has_labels=True)
        model_dir: Directory containing trained model and scaler
        output_file: Path to save prediction results (default: 'predictions.csv' in model_dir)
        has_labels: Whether metadata contains true labels for evaluation
    """
    # Load the model and scaler
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
    if not model_files:
        raise FileNotFoundError(f"No model found in {model_dir}")

    model_path = os.path.join(model_dir, model_files[0])
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load embeddings
    embeddings_file = os.path.join(embeddings_dir, CONFIG['embeddings_filename'])
    data = np.load(embeddings_file)
    contig_ids = data['ids']
    embeddings = data['embeddings']

    # Load metadata if needed
    true_labels = None
    if has_labels:
        # Trực tiếp sử dụng cột label từ metadata
        metadata_file = os.path.join(embeddings_dir, CONFIG['metadata_filename'])
        metadata_df = pd.read_csv(metadata_file)

        # Tạo DataFrame với ID từ embeddings
        embeddings_df = pd.DataFrame({'id': contig_ids})

        # Merge để lấy nhãn
        merged_df = embeddings_df.merge(metadata_df, on='id', how='left')

        # Kiểm tra nhãn bị thiếu
        valid_idx = ~merged_df['label'].isna()
        if not all(valid_idx):
            print(f"Warning: {(~valid_idx).sum()} contigs don't have labels and will be excluded")
            contig_ids = np.array(merged_df['id'][valid_idx])
            embeddings = embeddings[valid_idx.values]
            true_labels = np.array(merged_df['label'][valid_idx]).astype(int)
        else:
            true_labels = np.array(merged_df['label']).astype(int)

    # Scale features
    embeddings_scaled = scaler.transform(embeddings)

    # Make predictions
    predictions = model.predict(embeddings_scaled)
    probabilities = model.predict_proba(embeddings_scaled)[:, 1]

    # Create results dataframe
    results = {
        'contig_id': contig_ids,
        'predicted_label': ['temperate' if p == 0 else 'virulent' for p in predictions],
        'virulent_probability': probabilities
    }

    # Add true labels if available
    if has_labels:
        results['true_label'] = ['temperate' if l == 0 else 'virulent' for l in true_labels]

        # Calculate and print evaluation metrics
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Prediction accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(true_labels, predictions, target_names=['temperate', 'virulent']))

    # Save results
    if output_file is None:
        output_file = os.path.join(model_dir, 'predictions.csv')

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    return results_df


def run_multiple_models(config=None, models_to_run=None):
    """Run training and evaluation for multiple models and compare their performance."""
    if config is None:
        config = CONFIG

    # List of models to evaluate
    if models_to_run is None:
        model_types = ['xgb', 'lgbm', 'catboost']  # Mặc định chỉ chạy các mô hình GPU
    else:
        model_types = models_to_run
    results = {}

    # Load data once
    print(f"Loading training data from {config['train_dir']}")
    train_ids, train_embeddings, train_labels = load_data(
        config['train_dir'],
        config['embeddings_filename']
    )

    print(f"Loading validation data from {config['validation_dir']}")
    val_ids, val_embeddings, val_labels = load_data(
        config['validation_dir'],
        config['embeddings_filename']
    )

    # Create directory for comparison results
    comparison_dir = os.path.join(config['output_dir'], 'model_comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # Train and evaluate each model
    for model_type in model_types:
        print(f"\n{'=' * 50}")
        print(f"Training and evaluating {model_type.upper()} model")
        print(f"{'=' * 50}")

        # Update config for current model
        current_config = config.copy()
        current_config['model_type'] = model_type
        current_config['output_dir'] = os.path.join(config['output_dir'], model_type)

        # Train and evaluate
        model_results = train_and_evaluate(
            train_embeddings, train_labels, val_embeddings, val_labels,
            train_ids, val_ids, current_config
        )

        # Store results
        results[model_type] = {
            'accuracy': model_results['accuracy'],
            'classification_report': model_results['classification_report'],
            'predictions': model_results['validation_predictions']
        }

    # Compare model performance
    accuracy_comparison = {model: results[model]['accuracy'] for model in model_types}
    f1_comparison = {model: results[model]['classification_report']['weighted avg']['f1-score']
                     for model in model_types}

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': model_types,
        'Accuracy': [accuracy_comparison[model] for model in model_types],
        'F1-Score (Weighted)': [f1_comparison[model] for model in model_types]
    })

    # Sort by accuracy
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

    # Save comparison results
    comparison_df.to_csv(os.path.join(comparison_dir, 'model_comparison.csv'), index=False)

    # Plot performance comparison
    plt.figure(figsize=(12, 6))

    # Accuracy comparison
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='Accuracy', data=comparison_df)
    plt.title('Accuracy Comparison')
    plt.ylim(0.7, 1.0)  # Adjust as needed for better visualization
    plt.xticks(rotation=45)

    # F1-Score comparison
    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='F1-Score (Weighted)', data=comparison_df)
    plt.title('F1-Score Comparison')
    plt.ylim(0.7, 1.0)  # Adjust as needed
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'model_comparison.png'), dpi=300)

    # Select the best model
    best_model = comparison_df.iloc[0]['Model']
    print(f"\nBest performing model: {best_model.upper()} with accuracy {comparison_df.iloc[0]['Accuracy']:.4f}")

    return comparison_df, results


if __name__ == "__main__":
    # Bạn có thể sửa đổi CONFIG ở đây nếu cần
    # Ví dụ:
    # CONFIG['train_dir'] = './my_custom_train_dir'
    # CONFIG['validation_dir'] = './my_custom_validation_dir'

    try:
        # Kiểm tra xem GPU có khả dụng cho các mô hình không
        print("Kiểm tra khả năng sử dụng GPU...")

        # # Kiểm tra XGBoost
        # import xgboost as xgb
        #
        # print(f"XGBoost version: {xgb.__version__}")
        # try:
        #     xgb_check = XGBClassifier(tree_method='gpu_hist')
        #     print("XGBoost GPU available")
        #     xgb_available = True
        # except Exception as e:
        #     print(f"XGBoost GPU unavailable: {e}")
        #     xgb_available = False
        #
        # # Kiểm tra LightGBM
        # import lightgbm as lgb
        #
        # print(f"LightGBM version: {lgb.__version__}")
        # try:
        #     lgb_check = LGBMClassifier(device='gpu')
        #     print("LightGBM GPU available")
        #     lgb_available = True
        # except Exception as e:
        #     print(f"LightGBM GPU unavailable: {e}")
        #     lgb_available = False
        #
        # # Kiểm tra CatBoost
        # import catboost as cb
        #
        # print(f"CatBoost version: {cb.__version__}")
        # try:
        #     cb_check = CatBoostClassifier(task_type='GPU', devices='0', verbose=0)
        #     print("CatBoost GPU available")
        #     cb_available = True
        # except Exception as e:
        #     print(f"CatBoost GPU unavailable: {e}")
        #     cb_available = False

        # Quyết định mô hình nào sẽ chạy dựa trên khả năng sử dụng GPU
        # models_to_run = []
        # if xgb_available:
        #     models_to_run.append('xgb')
        # if lgb_available:
        #     models_to_run.append('lgbm')
        # if cb_available:
        #     models_to_run.append('catboost')
        #
        # if not models_to_run:
        #     print("Không có mô hình GPU nào khả dụng. Sử dụng mô hình CPU...")
        #     models_to_run = ['rf']  # Fallback to Random Forest
        #
        # print(f"Các mô hình sẽ chạy: {models_to_run}")

        # Để chạy một mô hình riêng lẻ
        CONFIG['model_type'] = 'xgb'
        run_training()

        # Hoặc để chạy và so sánh các mô hình
        # comparison_df, results = run_multiple_models(models_to_run=models_to_run)

    except Exception as e:
        print(f"Lỗi khi kiểm tra GPU: {e}")
        print("Fallback to CPU models...")
        # Fallback đến các mô hình CPU
        CONFIG['model_type'] = 'rf'
        run_training()

    # Dự đoán trên dữ liệu mới (bỏ ghi chú để sử dụng)
    # predict_new_samples(
    #     embeddings_dir='./new_data',
    #     model_dir='./model_output/xgb',
    #     has_labels=True  # Đặt thành False nếu dữ liệu mới không có nhãn
    # )