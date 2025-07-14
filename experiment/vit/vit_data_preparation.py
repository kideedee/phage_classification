import os
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import Dataset as HFDataset
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from transformers import ViTImageProcessor

from common.env_config import config
from logger.phg_cls_log import log

warnings.filterwarnings('ignore')


def preprocess_image_array(image_array):
    """Preprocess a single image array for ViT"""
    # Handle density maps
    if len(image_array.shape) == 3 and image_array.shape[2] == 1:
        density_map = image_array[:, :, 0]
    elif len(image_array.shape) == 2:
        density_map = image_array
    else:
        density_map = image_array[:, :, 0] if len(image_array.shape) == 3 else image_array

    # Improved normalization
    if density_map.max() > density_map.min():
        normalized = (density_map - density_map.min()) / (density_map.max() - density_map.min())
        normalized = (normalized * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(density_map, dtype=np.uint8)

    # Convert to 3-channel RGB
    if len(normalized.shape) == 2:
        image_array = np.stack([normalized] * 3, axis=-1)

    return Image.fromarray(image_array, 'RGB')


def process_batch(image_arrays, labels, processor, batch_start_idx):
    """Process a batch of images and return processed data"""
    batch_data = []

    for i, (image_array, label) in enumerate(zip(image_arrays, labels)):
        try:
            # Preprocess image
            image = preprocess_image_array(image_array)

            # Process with ViT processor
            encoding = processor(image, return_tensors="pt")
            pixel_values = encoding['pixel_values'].squeeze()

            batch_data.append({
                'pixel_values': pixel_values.numpy(),
                'labels': int(label)
            })

            # Log progress
            global_idx = batch_start_idx + i
            if (global_idx + 1) % 100 == 0:
                log.info(f"Processed {global_idx + 1} samples")

        except Exception as e:
            log.error(f"Error processing sample {batch_start_idx + i}: {e}")
            # Skip this sample or create a default one
            continue

    return batch_data


def save_batch_to_temp(batch_data, temp_dir, batch_idx, split_name):
    """Save batch data to temporary file"""
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f"{split_name}_batch_{batch_idx}.pt")
    torch.save(batch_data, temp_file)
    return temp_file


def combine_batches_to_hf_dataset(temp_dir, split_name, output_path):
    """Combine all batch files into a single HuggingFace dataset"""
    log.info(f"Combining {split_name} batches...")

    all_data = []
    files = [f for f in os.listdir(temp_dir)]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by batch number

    for file in files:
        data_path = os.path.join(temp_dir, file)
        batch_data = torch.load(data_path)
        all_data.extend(batch_data)

    # Create HuggingFace dataset
    hf_dataset = HFDataset.from_list(all_data)
    hf_dataset.set_format(type="torch", columns=["pixel_values", "labels"])

    # Save dataset
    hf_dataset.save_to_disk(output_path)
    log.info(f"Saved {split_name} dataset with {len(all_data)} samples to {output_path}")

    return len(all_data)


def process_data_with_batching(data_dir, min_size, max_size, fold, batch_size=32):
    """Process data with batch processing to manage memory"""

    log.info(f"Processing data for group {min_size}_{max_size}, fold {fold} with batch size {batch_size}")

    # Load numpy arrays
    try:
        train_image_arrays = np.load(os.path.join(data_dir, "train/fcgr_vectors.npy"))
        train_image_labels = np.load(os.path.join(data_dir, "train/fcgr_labels.npy"))

        train_image_arrays, _, train_image_labels, _ = train_test_split(train_image_arrays, train_image_labels,
                                                                        test_size=0.95, random_state=42,
                                                                        stratify=train_image_labels)

        val_image_arrays = np.load(os.path.join(data_dir, "test/fcgr_vectors.npy"))
        val_image_labels = np.load(os.path.join(data_dir, "test/fcgr_labels.npy"))
    except Exception as e:
        raise ValueError(f"Error loading numpy files: {e}")

    log.info(f"Train data shape: {train_image_arrays.shape}")
    log.info(f"Train labels shape: {train_image_labels.shape}")
    log.info(f"Val data shape: {val_image_arrays.shape}")
    log.info(f"Val labels shape: {val_image_labels.shape}")

    # Validate data
    if len(train_image_arrays) != len(train_image_labels):
        raise ValueError(f"Mismatch: {len(train_image_arrays)} images vs {len(train_image_labels)} labels")
    if len(val_image_arrays) != len(val_image_labels):
        raise ValueError(f"Mismatch: {len(val_image_arrays)} images vs {len(val_image_labels)} labels")

    # Simple validation - check labels are valid
    unique_labels = np.unique(train_image_labels)
    num_classes = len(unique_labels)
    log.info(f"Found {num_classes} classes: {unique_labels}")

    # Print class distribution
    unique, counts = np.unique(train_image_labels, return_counts=True)
    for label, count in zip(unique, counts):
        log.info(f"Train data, label {label}: {count} samples")

    unique, counts = np.unique(val_image_labels, return_counts=True)
    for label, count in zip(unique, counts):
        log.info(f"Val data, label {label}: {count} samples")

    return (train_image_arrays, train_image_labels,
            val_image_arrays, val_image_labels,
            num_classes)


def single_process(idx, row, processor, temp_dir):
    image = preprocess_image_array(row['pixel_values'])
    encoding = processor(image, return_tensors="pt")
    pixel_values = encoding['pixel_values'].squeeze()

    # temp_file = os.path.join(temp_dir, f"row_{row['label']}_{idx}.pt")
    # torch.save(pixel_values, temp_file)
    return pixel_values, row['label']


def processing(data_arrays, data_labels, temp_dir, processor):
    images = []
    labels = []

    df = pd.DataFrame(zip(data_arrays, data_labels), columns=["pixel_values", "label"])
    results = Parallel(n_jobs=10)(
        delayed(single_process)(
            idx, row, processor, temp_dir
        ) for idx, row in df.iterrows()
    )

    for result in results:
        if result:
            images.append(result[0])
            labels.append(result[1])

    return np.array(images), np.array(labels)


def main():
    """Main processing function with batch processing"""
    # Initialize ViT processor
    processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224",
        do_rescale=False,
        do_normalize=True
    )

    # Configuration
    batch_size = 32  # Adjust based on your memory constraints

    for i in range(1):
        if i == 0:
            min_size = 100
            max_size = 400
        elif i == 1:
            min_size = 400
            max_size = 800
        elif i == 2:
            min_size = 800
            max_size = 1200
        elif i == 3:
            min_size = 1200
            max_size = 1800
        else:
            raise ValueError

        group = f"{min_size}_{max_size}"

        for j in range(1):
            fold = j + 1
            log.info(f"Processing group: {group}, fold: {fold}")

            # Define paths
            input_data_dir = os.path.join(config.FCGR_DATA_DIR, f"{group}/fold_{fold}")
            output_dir = config.VIT_PREPARED_DATA_DIR
            temp_dir = os.path.join(output_dir, "temp_batches")

            output_train_dataset = os.path.join(output_dir, "processed_train_dataset")
            output_val_dataset = os.path.join(output_dir, "processed_val_dataset")
            output_processor = os.path.join(output_dir, "processor")
            output_config = os.path.join(output_dir, "config.json")

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(output_train_dataset, exist_ok=True)
            os.makedirs(output_val_dataset, exist_ok=True)
            os.makedirs(output_processor, exist_ok=True)

            # Process data
            (train_arrays, train_labels, val_arrays, val_labels,
             num_classes) = process_data_with_batching(
                input_data_dir, min_size, max_size, fold,
                batch_size=batch_size
            )

            processing(train_arrays, train_labels, temp_dir, processor)
            # processing(val_arrays, val_labels, output_val_dataset, processor)

            # Free original arrays
            # del train_arrays, train_labels, val_arrays, val_labels

            # Combine batches into final datasets
            train_samples = combine_batches_to_hf_dataset(temp_dir, "train", output_train_dataset)
            # val_samples = combine_batches_to_hf_dataset(temp_dir, "val", output_val_dataset)

            # Clean up temp directory
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

            # Save processor
            # processor.save_pretrained(output_processor)

            # Save simple config with just number of classes
            # with open(output_config, 'w') as f:
            #     json.dump({
            #         'num_classes': num_classes
            #     }, f, indent=2)

            log.info(f"Data preparation completed for {group}/fold_{fold}!")
            log.info(f"Train dataset: {train_samples} samples -> {output_train_dataset}")
            # log.info(f"Val dataset: {val_samples} samples -> {output_val_dataset}")
            log.info(f"Processor saved to: {output_processor}")
            log.info(f"Config saved to: {output_config}")


if __name__ == '__main__':
    main()
