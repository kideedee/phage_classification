import os.path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from common.env_config import config


def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["sequence"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        "./codon_tokenizer",
        do_basic_tokenize=False
    )

    train_df = pd.read_csv(os.path.join(config.CODON_EMBEDDING_OUTPUT_DIR, "1200_1800/fold_1/train/data.csv"))
    val_df = pd.read_csv(os.path.join(config.CODON_EMBEDDING_OUTPUT_DIR, "1200_1800/fold_1/test/data.csv"))

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples=examples, tokenizer=tokenizer),
        batched=True,
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples=examples, tokenizer=tokenizer),
        batched=True,
    )

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_valid = tokenized_val.rename_column("label", "labels")

    tokenized_train.save_to_disk("tokenized_train")
    tokenized_val.save_to_disk("tokenized_val")
