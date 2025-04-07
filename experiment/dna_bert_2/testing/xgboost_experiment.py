import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, callback
import warnings
from tqdm.auto import tqdm  # Import tqdm for progress bars

warnings.filterwarnings('ignore')

# Default configuration
CONFIG = {
    # Data paths
    'train_dir': './train_embeddings',
    'validation_dir': './test_embeddings',  # Using test set as validation
    'output_dir': './model_output',

    # Model and training parameters
    'random_seed': 42,
    'test_mode': False,  # Set to True to run with a small subset of data
    'test_sample_size': 100,  # Number of samples to use in test mode

    # File names
    'embeddings_filename': 'dnabert2_embeddings.npz',
    'metadata_filename': 'metadata.csv',

    # Grid Search parameters for XGBoost
    'xgb_params': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },

    # Reduced parameter grid for test mode
    'xgb_params_test': {
        'n_estimators': [10],
        'max_depth': [3],
        'learning_rate': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
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
    """Train and evaluate the XGBoost classifier."""
    random_seed = config['random_seed']
    output_dir = config['output_dir']
    test_mode = config.get('test_mode', False)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Set up XGBoost model
    param_grid = config['xgb_params_test'] if test_mode else config['xgb_params']

    # Adjust base model parameters for test mode
    n_estimators = 10 if test_mode else 100

    base_model = XGBClassifier(
        random_state=random_seed,
        tree_method='hist',  # GPU acceleration
        device="cuda",
        predictor='gpu_predictor',
        eval_metric='logloss',
        n_estimators=n_estimators
    )

    # Perform grid search
    print("Training XGBoost model with hyperparameter tuning...")
    # Create a progress callback for base model
    progress = callback.ProgressBar(period=10)  # Update every 10 boosting rounds

    # Adjust cv and verbosity based on test mode
    cv = 2 if test_mode else 5
    verbose_level = 1 if test_mode else 2

    # Add a verbose parameter to the base model
    base_model.set_params(verbosity=1)  # Set verbosity level

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='precision',
        n_jobs=-1,
        verbose=verbose_level
    )
    grid_search.fit(X_train_scaled, y_train, callbacks=[progress])

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
    joblib.dump(best_model, os.path.join(output_dir, 'xgb_model.pkl'))
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


def create_custom_progress_callback(title="Training Progress"):
    """
    Create a custom progress callback that shows more detailed information.
    """
    from tqdm import tqdm

    class CustomProgressCallback(callback.TrainingCallback):
        def __init__(self):
            self.pbar = None

        def before_training(self, model):
            # Initialize progress bar before training starts
            self.pbar = tqdm(total=model.get_params()['n_estimators'], desc=title)
            return model

        def after_iteration(self, model, epoch, evals_log):
            # Update progress bar after each iteration
            self.pbar.update(1)
            if evals_log:
                # If evaluation results exist, display them in the progress bar description
                eval_results = list(evals_log.items())[0][1]
                metric = list(eval_results.keys())[0]
                value = eval_results[metric][-1]
                self.pbar.set_description(f"{title} - {metric}: {value:.4f}")
            return False

        def after_training(self, model):
            # Close progress bar after training finishes
            self.pbar.close()
            return model

    return CustomProgressCallback()


def run_training(config=None):
    """Main function to run the training and evaluation process."""
    # Use provided config or default CONFIG
    if config is None:
        config = CONFIG

    test_mode = config.get('test_mode', False)
    test_sample_size = config.get('test_sample_size', 100)

    # Create output directory with test mode indicator if needed
    if test_mode:
        config['output_dir'] = os.path.join(config['output_dir'], 'test_mode')

    # Load training data
    print(f"Loading training data from {config['train_dir']}")
    train_ids, train_embeddings, train_labels = load_data(
        config['train_dir'],
        config['embeddings_filename']
    )

    # Subsample data if in test mode
    if test_mode:
        print(f"TEST MODE ACTIVE: Using only {test_sample_size} samples")
        if len(train_ids) > test_sample_size:
            # Ensure we get samples from both classes if possible
            indices_class0 = np.where(train_labels == 0)[0]
            indices_class1 = np.where(train_labels == 1)[0]

            # Calculate how many samples to take from each class
            n_class0 = min(len(indices_class0), test_sample_size // 2)
            n_class1 = min(len(indices_class1), test_sample_size - n_class0)

            # If one class doesn't have enough samples, take more from the other
            if n_class0 < test_sample_size // 2:
                n_class1 = min(len(indices_class1), test_sample_size - n_class0)

            # Sample indices from each class
            sampled_indices_class0 = np.random.choice(indices_class0, n_class0, replace=False)
            sampled_indices_class1 = np.random.choice(indices_class1, n_class1, replace=False)

            # Combine indices
            sampled_indices = np.concatenate([sampled_indices_class0, sampled_indices_class1])
            np.random.shuffle(sampled_indices)

            # Subsample the data
            train_ids = train_ids[sampled_indices]
            train_embeddings = train_embeddings[sampled_indices]
            train_labels = train_labels[sampled_indices]

    print(f"Loaded {len(train_ids)} training samples")
    print(f"Embedding dimension: {train_embeddings.shape[1]}")
    print(f"Class distribution in training data: {np.bincount(train_labels)} (0=temperate, 1=virulent)")

    # Load validation data (using test data)
    print(f"Loading validation data from {config['validation_dir']}")
    val_ids, val_embeddings, val_labels = load_data(
        config['validation_dir'],
        config['embeddings_filename']
    )

    # Subsample validation data if in test mode
    if test_mode and len(val_ids) > test_sample_size:
        print(f"TEST MODE: Subsampling validation data to {test_sample_size} samples")
        # Similar stratified sampling for validation data
        indices_class0 = np.where(val_labels == 0)[0]
        indices_class1 = np.where(val_labels == 1)[0]

        n_class0 = min(len(indices_class0), test_sample_size // 2)
        n_class1 = min(len(indices_class1), test_sample_size - n_class0)

        sampled_indices_class0 = np.random.choice(indices_class0, n_class0, replace=False)
        sampled_indices_class1 = np.random.choice(indices_class1, n_class1, replace=False)

        sampled_indices = np.concatenate([sampled_indices_class0, sampled_indices_class1])
        np.random.shuffle(sampled_indices)

        val_ids = val_ids[sampled_indices]
        val_embeddings = val_embeddings[sampled_indices]
        val_labels = val_labels[sampled_indices]

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
    model_path = os.path.join(model_dir, 'xgb_model.pkl')
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
        # Directly use label column from metadata
        metadata_file = os.path.join(embeddings_dir, CONFIG['metadata_filename'])
        metadata_df = pd.read_csv(metadata_file)

        # Create DataFrame with IDs from embeddings
        embeddings_df = pd.DataFrame({'id': contig_ids})

        # Merge to get labels
        merged_df = embeddings_df.merge(metadata_df, on='id', how='left')

        # Check for missing labels
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


if __name__ == "__main__":
    try:
        import argparse

        # Add command-line argument parsing
        parser = argparse.ArgumentParser(description='XGBoost training for DNA sequence classification')
        parser.add_argument('--test', action='store_true', help='Run in test mode with reduced dataset')
        parser.add_argument('--samples', type=int, default=100, help='Number of samples to use in test mode')
        parser.add_argument('--output', type=str, help='Output directory for model and results')
        args = parser.parse_args()

        # Update config based on command-line arguments
        config = CONFIG.copy()
        if args.test:
            config['test_mode'] = True
            config['test_sample_size'] = args.samples
            print(f"Running in TEST MODE with {args.samples} samples")

        if args.output:
            config['output_dir'] = args.output

        # Add tqdm to track overall progress
        print("Starting XGBoost training pipeline...")

        # You can also wrap longer operations in tqdm
        with tqdm(total=3, desc="Pipeline Progress") as pbar:
            # Load data
            pbar.set_description("Loading data")
            pbar.update(1)

            # Run XGBoost training
            pbar.set_description("Training model")
            run_training(config)
            pbar.update(1)

            # Finalize
            pbar.set_description("Saving results")
            print("XGBoost training complete!")
            pbar.update(1)
    except Exception as e:
        print(f"Error when running XGBoost: {e}")

    # Uncomment to predict on new data
    # predict_new_samples(
    #     embeddings_dir='./new_data',
    #     model_dir='./model_output',
    #     has_labels=True  # Set to False if new data doesn't have labels
    # )