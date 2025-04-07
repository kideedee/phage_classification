import numpy as np
import pandas as pd
from Bio import SeqIO
from datasets import Dataset
from transformers import AutoTokenizer

from common import sequence_windowing
from common.env_config import config


def tokenize_function(examples):
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=512)


if __name__ == '__main__':

    # Read all sequences from the file
    train_sequences = list(SeqIO.parse(config.TRAIN_DATA_FASTA_FILE, "fasta"))
    test_sequences = list(SeqIO.parse(config.TEST_DATA_FASTA_FILE, "fasta"))

    # Create windowed sequences
    train_windowed_records = sequence_windowing.window_fasta_with_distribution(
        config.TRAIN_DATA_FASTA_FILE,
        "train_sequences.fasta",
        distribution_type="normal",
        min_size=100,
        max_size=400,
        range_width=6,
        overlap_percent=10,
        step_size=None
    )
    test_windowed_records = sequence_windowing.window_fasta_with_distribution(
        config.TEST_DATA_FASTA_FILE,
        "test_sequences.fasta",
        distribution_type="normal",
        min_size=100,
        max_size=400,
        range_width=6,
        overlap_percent=10,
        step_size=None
    )

    # windowed_records = list(SeqIO.parse("train_sequences.fasta", "fasta"))

    sequence_windowing.analyze_window_sizes(train_windowed_records, "uniform_distribution.png")
    sequence_windowing.analyze_window_sizes(test_windowed_records, "uniform_distribution.png")

    train_record_sequences = []
    train_record_labels = []
    for record in train_windowed_records:
        sequence = str(record.seq)
        label = 0 if record.id.__contains__("Lysogenic") else 1

        train_record_sequences.append(sequence)
        train_record_labels.append(label)

    test_record_sequences = []
    test_record_labels = []
    for record in test_windowed_records:
        sequence = str(record.seq)
        label = 0 if record.id.__contains__("Lysogenic") else 1

        test_record_sequences.append(sequence)
        test_record_labels.append(label)

    columns = ["sequence", "label"]
    train_df = pd.DataFrame(list(zip(train_record_sequences, train_record_labels)), columns=columns)
    val_df = pd.DataFrame(list(zip(test_record_sequences, test_record_labels)), columns=columns)

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")

    # Convert pandas DataFrame to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {val_df.shape}")
    print(f"Label distribution in train set: {train_df['label'].value_counts()}")
    print(f"Label distribution in test set: {val_df['label'].value_counts()}")

    # Tokenize the datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    seq_lengths = [len(seq) for seq in train_df['sequence']]
    print(f"Mean sequence length: {np.mean(seq_lengths)}")
    print(f"Max sequence length: {np.max(seq_lengths)}")
    print(f"Min sequence length: {np.min(seq_lengths)}")
    print(f"Sequences longer than 512: {sum(1 for l in seq_lengths if l > 512)}")

    tokenized_train.save_to_disk("processed_train_dataset")
    tokenized_val.save_to_disk("processed_val_dataset")

    print("Data preparation completed!")
