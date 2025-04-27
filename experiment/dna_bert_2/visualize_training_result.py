import json
import matplotlib.pyplot as plt
import numpy as np
import os


# Script to visualize training metrics from saved logs
# This can be run after training to generate visualization

def plot_training_metrics(history_file='training_history.json'):
    """
    Plot training metrics from saved JSON history file
    """
    # Check if file exists
    if not os.path.exists(history_file):
        print(f"Error: {history_file} does not exist.")
        return False

    # Load training history
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Create directory for plots
    os.makedirs('plots', exist_ok=True)

    # Plot loss history
    plt.figure(figsize=(12, 8))

    # Extract training loss data if available
    if history['train_loss']:
        steps, values = zip(*history['train_loss'])
        plt.plot(steps, values, label='Training Loss', color='blue')

    # Extract validation loss data if available
    if history['eval_loss']:
        steps, values = zip(*history['eval_loss'])
        plt.plot(steps, values, label='Validation Loss', color='red', linestyle='--')

    plt.title('Loss History', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/loss_history.png', dpi=300)
    print("Loss history plot saved to plots/loss_history.png")
    plt.close()

    # Plot accuracy and F1 score
    plt.figure(figsize=(12, 8))

    # Extract accuracy data if available
    if history['eval_accuracy']:
        steps, values = zip(*history['eval_accuracy'])
        plt.plot(steps, values, label='Accuracy', color='green')

    # Extract F1 score data if available
    if history['eval_f1']:
        steps, values = zip(*history['eval_f1'])
        plt.plot(steps, values, label='F1 Score', color='purple')

    plt.title('Accuracy and F1 Score', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/accuracy_f1.png', dpi=300)
    print("Accuracy and F1 plot saved to plots/accuracy_f1.png")
    plt.close()

    # Plot precision and recall
    plt.figure(figsize=(12, 8))

    # Extract precision data if available
    if history['eval_precision']:
        steps, values = zip(*history['eval_precision'])
        plt.plot(steps, values, label='Precision', color='orange')

    # Extract recall data if available
    if history['eval_recall']:
        steps, values = zip(*history['eval_recall'])
        plt.plot(steps, values, label='Recall', color='brown')

    # Extract ROC AUC data if available
    if history['eval_roc_auc']:
        steps, values = zip(*history['eval_roc_auc'])
        plt.plot(steps, values, label='ROC AUC', color='magenta')

    plt.title('Precision, Recall, and ROC AUC', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/precision_recall.png', dpi=300)
    print("Precision and Recall plot saved to plots/precision_recall.png")
    plt.close()

    # Plot learning rate
    plt.figure(figsize=(12, 8))

    # Extract learning rate data if available
    if history['learning_rate']:
        steps, values = zip(*history['learning_rate'])
        plt.plot(steps, values, color='cyan')

    plt.title('Learning Rate Schedule', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/learning_rate.png', dpi=300)
    print("Learning rate plot saved to plots/learning_rate.png")
    plt.close()

    # Create comprehensive dashboard
    plt.figure(figsize=(16, 12))

    # Loss subplot
    plt.subplot(2, 2, 1)
    if history['train_loss']:
        steps, values = zip(*history['train_loss'])
        plt.plot(steps, values, label='Training', color='blue')
    if history['eval_loss']:
        steps, values = zip(*history['eval_loss'])
        plt.plot(steps, values, label='Validation', color='red', linestyle='--')
    plt.title('Loss History')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Accuracy and F1 subplot
    plt.subplot(2, 2, 2)
    if history['eval_accuracy']:
        steps, values = zip(*history['eval_accuracy'])
        plt.plot(steps, values, label='Accuracy', color='green')
    if history['eval_f1']:
        steps, values = zip(*history['eval_f1'])
        plt.plot(steps, values, label='F1', color='purple')
    plt.title('Accuracy and F1')
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Precision and Recall subplot
    plt.subplot(2, 2, 3)
    if history['eval_precision']:
        steps, values = zip(*history['eval_precision'])
        plt.plot(steps, values, label='Precision', color='orange')
    if history['eval_recall']:
        steps, values = zip(*history['eval_recall'])
        plt.plot(steps, values, label='Recall', color='brown')
    plt.title('Precision and Recall')
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Learning Rate subplot
    plt.subplot(2, 2, 4)
    if history['learning_rate']:
        steps, values = zip(*history['learning_rate'])
        plt.plot(steps, values, color='cyan')
    plt.title('Learning Rate')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/training_dashboard.png', dpi=300)
    print("Training dashboard saved to plots/training_dashboard.png")
    plt.close()

    return True


def plot_final_metrics(metrics_file='final_metrics.json'):
    """
    Create a bar chart visualization of the final evaluation metrics
    """
    # Check if file exists
    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} does not exist.")
        return False

    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Create directory for plots
    os.makedirs('plots', exist_ok=True)

    # Extract relevant metrics for visualization
    plot_metrics = {
        'accuracy': metrics.get('eval_accuracy', 0),
        'f1': metrics.get('eval_f1', 0),
        'precision': metrics.get('eval_precision', 0),
        'recall': metrics.get('eval_recall', 0),
        'roc_auc': metrics.get('eval_roc_auc', 0),
        'specificity': metrics.get('eval_specificity', 0),
        'sensitivity': metrics.get('eval_sensitivity', 0)
    }

    # Create bar chart
    plt.figure(figsize=(14, 8))
    bars = plt.bar(plot_metrics.keys(), plot_metrics.values(),
                   color=['blue', 'purple', 'orange', 'brown', 'magenta', 'green', 'red'])

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.title('Final Evaluation Metrics', fontsize=16)
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1.1)  # Metrics are typically between 0 and 1
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/final_metrics.png', dpi=300)
    print("Final metrics plot saved to plots/final_metrics.png")
    plt.close()

    # Create confusion matrix visualization if available
    if all(key in metrics for key in ['eval_tp', 'eval_fp', 'eval_tn', 'eval_fn']):
        tp = metrics['eval_tp']
        fp = metrics['eval_fp']
        tn = metrics['eval_tn']
        fn = metrics['eval_fn']

        conf_matrix = np.array([[tn, fp], [fn, tp]])

        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=16)
        plt.colorbar()

        classes = ['Negative (0)', 'Positive (1)']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=12)
        plt.yticks(tick_marks, classes, fontsize=12)

        # Add text annotations
        thresh = conf_matrix.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{conf_matrix[i, j]:.0f}',
                         horizontalalignment="center",
                         color="white" if conf_matrix[i, j] > thresh else "black",
                         fontsize=14)

        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300)
        print("Confusion matrix plot saved to plots/confusion_matrix.png")
        plt.close()

    return True


if __name__ == "__main__":
    print("Generating training history visualizations...")
    plot_training_metrics()

    print("\nGenerating final metrics visualizations...")
    plot_final_metrics()

    print("\nAll visualizations complete!")