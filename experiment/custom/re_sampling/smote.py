import numpy as np
import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from cuml.neighbors import NearestNeighbors
from cuml.svm import SVC
import gc
import torch


def process_in_batches(X_file, y_file, batch_size=10000, random_state=42):
    """
    Process large datasets in batches using SVMSMOTE

    Parameters:
    -----------
    X_file : str
        Path to the numpy file containing feature vectors
    y_file : str
        Path to the numpy file containing labels
    batch_size : int
        Size of each batch to process
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X_resampled_all : numpy array
        Resampled feature vectors
    y_resampled_all : numpy array
        Resampled labels
    """
    # Step 1: Load the data in small chunks to analyze class distribution
    print("Loading data to analyze class distribution...")

    # Load data in chunks to analyze
    # Memory-mapped mode allows us to access parts of the array without loading it all
    X_mmap = np.load(X_file, mmap_mode='r')
    y_mmap = np.load(y_file, mmap_mode='r')

    # Analyze the class distribution
    class_counts = Counter(y_mmap)
    print(f"Original class distribution: {class_counts}")

    # Calculate target counts for each class
    max_class_count = max(class_counts.values())
    target_counts = {cls: max_class_count for cls in class_counts.keys()}
    print(f"Target class distribution: {target_counts}")

    # Initialize lists to store resampled data
    X_resampled_all = []
    y_resampled_all = []

    # Track current counts for each class
    current_counts = {cls: 0 for cls in class_counts.keys()}

    # Step 2: Process data in batches
    total_samples = len(y_mmap)
    num_batches = (total_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")

        # Determine batch indices
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)

        # Load batch into memory
        print(f"Loading batch data from index {start_idx} to {end_idx}...")
        X_batch = np.array(X_mmap[start_idx:end_idx])
        y_batch = np.array(y_mmap[start_idx:end_idx])

        print(f"Batch shape: {X_batch.shape}, {y_batch.shape}")

        # Skip if batch has only one class (SVMSMOTE requires at least 2 classes)
        batch_classes = len(np.unique(y_batch))
        if batch_classes < 2:
            print(f"Skipping batch {batch_idx + 1} as it contains only {batch_classes} class")
            X_resampled_all.append(X_batch)
            y_resampled_all.append(y_batch)
            # Update current counts
            batch_counts = Counter(y_batch)
            for cls, count in batch_counts.items():
                current_counts[cls] += count
            continue

        # Calculate how many samples to generate for each class in this batch
        batch_class_counts = Counter(y_batch)

        try:
            # Setup NearestNeighbors with cuML
            nn = NearestNeighbors(n_neighbors=min(6, min(batch_class_counts.values())))
            svm = SVC()

            # Apply SVMSMOTE on the batch
            sm = SVMSMOTE(
                k_neighbors=nn,
                m_neighbors=nn,
                svm_estimator=svm,
                random_state=random_state
            )

            X_batch_resampled, y_batch_resampled = sm.fit_resample(X_batch, y_batch)

            print(f"Batch resampled shape: {X_batch_resampled.shape}, {y_batch_resampled.shape}")

            # Add resampled batch to the result
            X_resampled_all.append(X_batch_resampled)
            y_resampled_all.append(y_batch_resampled)

            # Update current counts
            batch_counts = Counter(y_batch_resampled)
            for cls, count in batch_counts.items():
                current_counts[cls] += count

            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            # If error occurs, use the original batch data
            X_resampled_all.append(X_batch)
            y_resampled_all.append(y_batch)
            # Update current counts
            batch_counts = Counter(y_batch)
            for cls, count in batch_counts.items():
                current_counts[cls] += count

    # Concatenate all batches
    X_final = np.vstack(X_resampled_all)
    y_final = np.concatenate(y_resampled_all)

    print(f"\nFinal resampled data shape: {X_final.shape}, {y_final.shape}")
    print(f"Final class distribution: {Counter(y_final)}")

    return X_final, y_final


# Example usage
if __name__ == "__main__":
    # Set batch size according to your GPU memory
    # You may need to adjust this based on your model dimensionality and GPU
    BATCH_SIZE = 5000  # Adjust based on your GPU memory and data dimension

    # Process the data
    X_resampled, y_resampled = process_in_batches(
        "word2vec_train_vector.npy",
        "y_train.npy",
        batch_size=BATCH_SIZE
    )

    # Save the resampled data
    np.save("X_resampled.npy", X_resampled)
    np.save("y_resampled.npy", y_resampled)

    print("Resampled data saved successfully!")