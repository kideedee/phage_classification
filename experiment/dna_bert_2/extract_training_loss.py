import os
import glob
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import json


def extract_tensorboard_logs(logs_dir='logs'):
    """
    Extract loss values from TensorBoard event files

    Args:
        logs_dir: Directory containing TensorBoard event files

    Returns:
        Dictionary with training and validation loss history
    """
    # Find all event files
    event_files = glob.glob(os.path.join(logs_dir, '**/events.out.tfevents*'), recursive=True)

    if not event_files:
        print(f"No TensorBoard event files found in {logs_dir}")
        return None

    history = {
        'train_loss': [],
        'eval_loss': []
    }

    for event_file in event_files:
        print(f"Processing: {event_file}")

        # Load the event file
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={
                event_accumulator.SCALARS: 0,  # Load all scalar events
            }
        )
        ea.Reload()

        # Get available tags
        tags = ea.Tags()['scalars']
        print(f"Available tags: {tags}")

        # Extract loss values
        if 'loss' in tags:
            for event in ea.Scalars('loss'):
                history['train_loss'].append((event.step, event.value))

        if 'eval_loss' in tags:
            for event in ea.Scalars('eval_loss'):
                history['eval_loss'].append((event.step, event.value))

    # Sort by step
    history['train_loss'].sort(key=lambda x: x[0])
    history['eval_loss'].sort(key=lambda x: x[0])

    return history


def plot_loss_curves(history, output_file='loss_history.png'):
    """
    Plot loss curves from extracted history

    Args:
        history: Dictionary with training and validation loss history
        output_file: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    if history['train_loss']:
        steps, values = zip(*history['train_loss'])
        plt.plot(steps, values, label='Training Loss', color='blue')

    if history['eval_loss']:
        steps, values = zip(*history['eval_loss'])
        plt.plot(steps, values, label='Validation Loss', color='red', linestyle='--')

    plt.title('Loss History', fontsize=16)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Loss history plot saved to {output_file}")

    # Also save as CSV for future reference
    csv_file = output_file.replace('.png', '.csv')
    with open(csv_file, 'w') as f:
        f.write("step,train_loss,eval_loss\n")

        # Create a dictionary of steps to values
        train_dict = dict(history['train_loss'])
        eval_dict = dict(history['eval_loss'])

        # Get all steps
        all_steps = sorted(set(list(train_dict.keys()) + list(eval_dict.keys())))

        for step in all_steps:
            train_val = train_dict.get(step, "")
            eval_val = eval_dict.get(step, "")
            f.write(f"{step},{train_val},{eval_val}\n")

    print(f"Loss data saved to {csv_file}")

    # Save as JSON as well
    json_file = output_file.replace('.png', '.json')
    with open(json_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Loss data saved to {json_file}")

    return True


def extract_and_log_loss_from_logs():
    """
    Main function to extract and log loss values from TensorBoard logs
    """
    print("Extracting loss values from TensorBoard logs...")
    history = extract_tensorboard_logs()

    if history:
        print("\nLoss History:")

        print("\nTraining Loss:")
        for step, value in history['train_loss']:
            print(f"Step {step}: {value:.6f}")

        print("\nValidation Loss:")
        for step, value in history['eval_loss']:
            print(f"Step {step}: {value:.6f}")

        print("\nCreating loss curve plot...")
        plot_loss_curves(history)

        print("\nProcess completed successfully!")
    else:
        print("No data extracted. Please check if TensorBoard logs exist.")


if __name__ == "__main__":
    extract_and_log_loss_from_logs()