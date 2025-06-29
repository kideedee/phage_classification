import os

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from common.env_config import config
from embedding_sequence.encoding_sequence import DNASequenceProcessor
from logger.phg_cls_log import log


def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["sequence"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


if __name__ == '__main__':
    for i in range(5):
        fold = i + 1
        # if fold == 1:
        #     train_file = config.TRAIN_DATA_FOLD_1_CSV_FILE  # Update with your file path
        #     valid_file = config.TEST_DATA_FOLD_1_CSV_FILE  # Update with your file path
        # elif fold == 2:
        #     train_file = config.TRAIN_DATA_FOLD_2_CSV_FILE
        #     valid_file = config.TEST_DATA_FOLD_2_CSV_FILE
        # elif fold == 3:
        #     train_file = config.TRAIN_DATA_FOLD_3_CSV_FILE
        #     valid_file = config.TEST_DATA_FOLD_3_CSV_FILE
        # elif fold == 4:
        #     train_file = config.TRAIN_DATA_FOLD_4_CSV_FILE
        #     valid_file = config.TEST_DATA_FOLD_4_CSV_FILE
        # elif fold == 5:
        #     train_file = config.TRAIN_DATA_FOLD_5_CSV_FILE
        #     valid_file = config.TEST_DATA_FOLD_5_CSV_FILE
        # else:
        #     raise ValueError

        for j in range(4):
            if j == 0:
                min_size = 100
                max_size = 400
                overlap = 30
                max_length_tokenizer = int(max_size * 0.25)
            elif j == 1:
                min_size = 400
                max_size = 800
                overlap = 30
                max_length_tokenizer = int(max_size * 0.25)
            elif j == 2:
                min_size = 800
                max_size = 1200
                overlap = 50
                max_length_tokenizer = 512
            elif j == 3:
                min_size = 1200
                max_size = 1800
                overlap = 70
                max_length_tokenizer = 512
            else:
                raise ValueError

            # if j < 2:
            #     continue

            group = f"{min_size}_{max_size}"
            train_file = f"{config.FILTER_FROM_PHATYP_DATA}/{group}/{fold}/train/data.csv"
            valid_file = f"{config.FILTER_FROM_PHATYP_DATA}/{group}/{fold}/test/data.csv"
            log.info(f"Fold: {fold}, group: {group}")
            log.info(f"train_file: {train_file}")
            log.info(f"valid_file: {valid_file}")

            output_dir = os.path.join(config.PHATYP_FILTER_DNA_BERT_2_TOKENIZER_DATA_DIR, f"{group}/fold_{fold}")
            os.makedirs(output_dir, exist_ok=True)
            output_prepared_train_dataset = os.path.join(output_dir, f"processed_train_dataset")
            output_prepared_val_dataset = os.path.join(output_dir, f"processed_val_dataset")
            output_tokenizer = os.path.join(output_dir, "tokenizer")

            tokenizer = AutoTokenizer.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                padding_side="right",
                use_fast=True,
                trust_remote_code=True
            )

            # dna_bert_2_processor = DNASequenceProcessor(
            #     min_size=min_size,
            #     max_size=max_size,
            #     encoding_method="dna_bert_2",
            #     overlap_percent=overlap,
            #     dna_bert_pooling="cls",
            #     dna_bert_2_batch_size=64,  # Adjust based on your GPU memory
            #     num_workers=2,
            #     prefetch_factor=2
            # )

            # train_df, val_df = dna_bert_2_processor.load_and_clean_data(train_file, valid_file)
            # X_train, y_train, X_val, y_val = dna_bert_2_processor.window_and_extract_features(train_df, val_df)
            # X_train_aug, y_train_aug = dna_bert_2_processor.reverse_complement_augmentation(X_train, y_train)
            # log.info(f"Original training set size: {len(X_train_aug)}")
            # for k, v in Counter(y_train_aug).items():
            #     per = v / len(y_train_aug) * 100
            #     print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
            #
            # X_train_aug_df = pd.DataFrame({'sequence': X_train_aug})
            # index_array = np.array(X_train_aug_df.index).reshape(-1, 1)  # Convert indexes to 2D array
            # undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            # index_resampled, y_resampled = undersampler.fit_resample(index_array, y_train_aug)
            # X_resampled = X_train_aug_df.iloc[index_resampled.flatten()]['sequence'].values
            # log.info(f"Resampled training set size: {len(X_resampled)}")
            # for k, v in Counter(y_resampled).items():
            #     per = v / len(y_resampled) * 100
            #     print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
            #
            # processed_train_df = pd.DataFrame(zip(X_resampled, y_resampled), columns=["sequence", "target"])
            # processed_val_df = pd.DataFrame(zip(X_val, y_val), columns=["sequence", "target"])
            # processed_train_df = processed_train_df.sample(frac=0.02, random_state=42).reset_index(drop=True)
            # processed_val_df = processed_val_df.sample(frac=0.02, random_state=42).reset_index(drop=True)

            processed_train_df = pd.read_csv(train_file)[["sequence", "label_y"]]
            processed_train_df.columns.values[1]="target"
            processed_val_df = pd.read_csv(valid_file)[["sequence", "label_y"]]
            processed_val_df.columns.values[1] = "target"

            # Convert dataframes to Hugging Face datasets
            train_dataset = Dataset.from_pandas(processed_train_df)
            valid_dataset = Dataset.from_pandas(processed_val_df)

            tokenized_train = train_dataset.map(
                lambda examples: tokenize_function(examples, tokenizer, max_length_tokenizer),
                batched=True
            )
            tokenized_valid = valid_dataset.map(
                lambda examples: tokenize_function(examples, tokenizer, max_length_tokenizer),
                batched=True
            )

            tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])
            tokenized_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "target"])

            tokenized_train = tokenized_train.rename_column("target", "labels")
            tokenized_valid = tokenized_valid.rename_column("target", "labels")

            tokenized_train.save_to_disk(output_prepared_train_dataset)
            tokenized_valid.save_to_disk(output_prepared_val_dataset)

            tokenizer.save_pretrained(output_tokenizer)

            log.info("Data preparation completed!")
