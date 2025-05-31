import os.path

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from common.env_config import config
from embedding_sequence.encoding_sequence import DNASequenceProcessor

if __name__ == '__main__':
    min_contig_length = 400
    max_contig_length = 800
    out_dir = os.path.join(config.DATA_DIR, f"my_data/{min_contig_length}_{max_contig_length}")
    os.makedirs(out_dir, exist_ok=True)


    dna_bert_2_processor = DNASequenceProcessor(
        min_size=min_contig_length,
        max_size=max_contig_length,
        encoding_method="dna_bert_2",
        overlap_percent=30,
        dna_bert_model_name="zhihan1996/DNABERT-2-117M",
        dna_bert_pooling="cls",
        dna_bert_2_batch_size=64,  # Adjust based on your GPU memory
        output_dir=f"dna_bert_2_output",
        is_fine_tune_dna_bert=True,  # Enable fine-tuning
        fine_tune_epochs=3,
        fine_tune_batch_size=16,
        fine_tune_learning_rate=5e-5,
        num_workers=2,
        prefetch_factor=2
    )

    train_file = config.TRAIN_DATA_CSV_FILE  # Update with your file path
    valid_file = config.VAL_DATA_CSV_FILE  # Update with your file path
    train_df, val_df = dna_bert_2_processor.load_and_clean_data(train_file, valid_file)
    X_train, y_train, X_val, y_val = dna_bert_2_processor.window_and_extract_features(train_df, val_df)
    X_train_aug, y_train_aug = dna_bert_2_processor.reverse_complement_augmentation(X_train, y_train)
    del X_train, y_train  # Free memory

    X_train_aug_df = pd.DataFrame({'sequence': X_train_aug})
    index_array = np.array(X_train_aug_df.index).reshape(-1, 1)  # Convert indexes to 2D array
    under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    index_resampled, y_resampled = under_sampler.fit_resample(index_array, y_train_aug)
    X_resampled = X_train_aug_df.iloc[index_resampled.flatten()]['sequence'].values
    print(f"Original training set size: {len(X_train_aug_df)}")
    print(f"Resampled training set size: {len(X_resampled)}")
    print(f"Distribution of classes in resampled dataset: {pd.Series(y_resampled).value_counts()}")

    np.save(os.path.join(out_dir, "X_train_aug.npy"), X_resampled)
    np.save(os.path.join(out_dir, "y_train_aug.npy"), y_resampled)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)
