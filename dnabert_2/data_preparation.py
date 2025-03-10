import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

train_df = pd.read_csv('../data/dnabert_2_preparation/train.csv')
val_df = pd.read_csv('../data/dnabert_2_preparation/val.csv')
test_df = pd.read_csv('../data/dnabert_2_preparation/test.csv')

print(f"Train set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Label distribution in train set: {train_df['label'].value_counts()}")
print(f"Label distribution in validation set: {val_df['label'].value_counts()}")
print(f"Label distribution in test set: {test_df['label'].value_counts()}")

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")


def tokenize_function(examples):
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=512)


# Convert pandas DataFrame to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

seq_lengths = [len(seq) for seq in train_df['sequence']]
print(f"Mean sequence length: {np.mean(seq_lengths)}")
print(f"Max sequence length: {np.max(seq_lengths)}")
print(f"Min sequence length: {np.min(seq_lengths)}")
print(f"Sequences longer than 512: {sum(1 for l in seq_lengths if l > 512)}")

tokenized_train.save_to_disk("processed_train_dataset")
tokenized_val.save_to_disk("processed_val_dataset")
tokenized_test.save_to_disk("processed_test_dataset")

print("Data preparation completed!")
