import numpy as np
import pandas as pd
from datasets import Dataset
from imblearn.under_sampling import RandomUnderSampler
from transformers import AutoTokenizer

from common.env_config import config
from embedding_sequence.encoding_sequence import DNASequenceProcessor


def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["sequence"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")

    dna_bert_2_processor = DNASequenceProcessor(
        min_size=400,
        max_size=800,
        encoding_method="dna_bert_2",
        overlap_percent=30,
        dna_bert_model_name="zhihan1996/DNABERT-2-117M",
        dna_bert_pooling="cls",
        dna_bert_batch_size=64,  # Adjust based on your GPU memory
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
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    index_resampled, y_resampled = undersampler.fit_resample(index_array, y_train_aug)
    X_resampled = X_train_aug_df.iloc[index_resampled.flatten()]['sequence'].values
    print(f"Original training set size: {len(X_train_aug_df)}")
    print(f"Resampled training set size: {len(X_resampled)}")
    # Print the distribution of classes in the resampled dataset
    # print(f"Distribution of classes in resampled dataset: {pd.Series(y_resampled).value_counts()}")

    processed_train_df = pd.DataFrame(zip(X_resampled, y_resampled), columns=["sequence", "target"])
    # processed_train_df = processed_train_df.sample(frac=0.02, random_state=42).reset_index(drop=True)
    processed_val_df = pd.DataFrame(zip(X_val, y_val), columns=["sequence", "target"])
    # processed_val_df = processed_val_df.sample(frac=0.02, random_state=42).reset_index(drop=True)
    print(f"Processed training set size: {len(processed_train_df)}")
    print(f"Processed validation set size: {len(processed_val_df)}")
    # Print the distribution of classes in the processed datasets
    print(f"Distribution of classes in processed training set: {processed_train_df['target'].value_counts()}")
    print(f"Distribution of classes in processed validation set: {processed_val_df['target'].value_counts()}")

    # Convert dataframes to Hugging Face datasets
    train_dataset = Dataset.from_pandas(processed_train_df)
    valid_dataset = Dataset.from_pandas(processed_val_df)

    max_length = 1200
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True
    )
    tokenized_valid = valid_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True
    )

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])
    tokenized_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])

    tokenized_train = tokenized_train.rename_column("target", "labels")
    tokenized_valid = tokenized_valid.rename_column("target", "labels")

    tokenized_train.save_to_disk("processed_train_dataset")
    tokenized_valid.save_to_disk("processed_val_dataset")

    print("Data preparation completed!")
