import re


def find_best_epoch(file_path):
    # Read the log file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split content into epochs
    epochs = content.split('Epoch [')

    best_acc = 0
    best_epoch = None
    best_epoch_num = 0

    # Process each epoch
    for epoch in epochs[1:]:  # Skip first split as it's the header
        # Extract epoch number
        epoch_num = int(epoch.split('/')[0])

        # Find validation accuracy using regex
        val_acc_match = re.search(r'Val Acc: (\d+\.\d+)', epoch)
        if val_acc_match:
            val_acc = float(val_acc_match.group(1))

            # Update best if current is better
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_epoch_num = epoch_num

    if best_epoch:
        # Extract key metrics
        train_loss = re.search(r'Train Loss: (\d+\.\d+)', best_epoch).group(1)
        train_acc = re.search(r'Train Acc: (\d+\.\d+)', best_epoch).group(1)
        val_loss = re.search(r'Val Loss: (\d+\.\d+)', best_epoch).group(1)
        lr = re.search(r'LR: (\d+\.\d+)', best_epoch).group(1)

        # Extract performance metrics
        train_perf = re.search(r'Training Performance:(.*?)Validation Performance:',
                               best_epoch, re.DOTALL).group(1)
        val_perf = re.search(r'Validation Performance:(.*?)Epoch time:',
                             best_epoch, re.DOTALL).group(1)

        print(f"Best Performance at Epoch {best_epoch_num}:")
        print(f"Train Loss: {train_loss}")
        print(f"Train Accuracy: {train_acc}")
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {best_acc}")
        print(f"Learning Rate: {lr}")
        print("\nTraining Performance:")
        print(train_perf)
        print("Validation Performance:")
        print(val_perf)

    return best_epoch_num, best_acc


# Example usage
if __name__ == "__main__":
    file_path = "log_100_400.txt"
    best_epoch_num, best_acc = find_best_epoch(file_path)
    print(f"\nSummary: Best validation accuracy {best_acc:.4f} was achieved at epoch {best_epoch_num}")