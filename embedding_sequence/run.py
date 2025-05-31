# Example usage
import os

from common.env_config import config
from embedding_sequence.encoding_sequence import DNASequenceProcessor
from logger.phg_cls_log import log

if __name__ == '__main__':
    # Example with Word2Vec encoding
    # length = "100_400"
    # min_length = 1200
    # max_length = 1800
    # output_model = f"phage_word2vec_model_{length}.bin"
    # output_dir = f"word2vec_output_{length}"

    # Example with DNA-BERT encoding and fine-tuning
    # dna_bert_processor = DNASequenceProcessor(
    #     encoding_method="dna_bert",
    #     kmer_size=6,  # Must be 3, 4, 5, or 6 for DNA-BERT
    #     overlap_percent=50,
    #     dna_bert_model_name="zhihan1996/DNA_bert_6",
    #     dna_bert_pooling="cls",
    #     dna_bert_batch_size=64,  # Reduced batch size due to GPU memory constraints during fine-tuning
    #     output_dir="../dna_bert_output",
    #     is_fine_tune_dna_bert=True,  # Enable fine-tuning
    #     fine_tune_epochs=3,
    #     fine_tune_batch_size=16,
    #     fine_tune_learning_rate=5e-5
    # )
    #
    # dna_bert_processor.process(
    #     train_path=config.TRAIN_DATA_CSV_FILE,
    #     val_path=config.VAL_DATA_CSV_FILE
    # )

    for i in range(5):
        fold_index = i + 1
        overlap_percent = 30

        if fold_index == 1:
            train_path = config.TRAIN_DATA_FOLD_1_CSV_FILE
            val_path = config.TEST_DATA_FOLD_1_CSV_FILE
            tokenizer_path = "../model/dna_bert_2/pretrained_100_400/1/tokenizer"
            model_path = "../model/dna_bert_2/pretrained_100_400/1/finetune_dna_bert"
        elif fold_index == 2:
            train_path = config.TRAIN_DATA_FOLD_2_CSV_FILE
            val_path = config.TEST_DATA_FOLD_2_CSV_FILE
            tokenizer_path = "../model/dna_bert_2/pretrained_100_400/2/tokenizer"
            model_path = "../model/dna_bert_2/pretrained_100_400/2/finetune_dna_bert"
        elif fold_index == 3:
            train_path = config.TRAIN_DATA_FOLD_3_CSV_FILE
            val_path = config.TEST_DATA_FOLD_3_CSV_FILE
            tokenizer_path = "../model/dna_bert_2/pretrained_100_400/3/tokenizer"
            model_path = "../model/dna_bert_2/pretrained_100_400/3/finetune_dna_bert"
        elif fold_index == 4:
            train_path = config.TRAIN_DATA_FOLD_4_CSV_FILE
            val_path = config.TEST_DATA_FOLD_4_CSV_FILE
            tokenizer_path = "../model/dna_bert_2/pretrained_100_400/4/tokenizer"
            model_path = "../model/dna_bert_2/pretrained_100_400/4/finetune_dna_bert"
        elif fold_index == 5:
            train_path = config.TRAIN_DATA_FOLD_5_CSV_FILE
            val_path = config.TEST_DATA_FOLD_5_CSV_FILE
            tokenizer_path = "../model/dna_bert_2/pretrained_100_400/5/tokenizer"
            model_path = "../model/dna_bert_2/pretrained_100_400/5/finetune_dna_bert"
        else:
            raise ValueError(f"Invalid fold_index: {fold_index}")

        for group in range(4):
            if group == 0:
                min_size = 100
                max_size = 400
                batch_size = 128
            elif group == 1:
                min_size = 400
                max_size = 800
                batch_size = 128
            elif group == 2:
                min_size = 800
                max_size = 1200
                batch_size = 96
            elif group == 3:
                min_size = 1200
                max_size = 1800
                batch_size = 96
            else:
                raise ValueError(f"Invalid group: {group}")

            # Example with DNABERT-2 encoding and fine-tuning
            dna_bert_2_processor = DNASequenceProcessor(
                min_size=min_size,
                max_size=max_size,
                encoding_method="dna_bert_2",
                overlap_percent=30,
                fold=fold_index,
                dna_bert_2_batch_size=batch_size,
                group=group,
                dna_bert_2_tokenizer_path=tokenizer_path,
                dna_bert_2_model_path=model_path
            )

            output_dir = dna_bert_2_processor.output_dir
            if os.path.exists(output_dir):
                log.info(f"Output directory {output_dir} already exists, skipping.")
                continue

            dna_bert_2_processor.process(
                train_path=train_path,
                val_path=val_path
            )

            # # Khởi tạo với one-hot encoding
            # processor = DNASequenceProcessor(
            #     encoding_method="one_hot",
            #     fold=fold_index,
            #     min_size=min_size,
            #     max_size=max_size,
            #     overlap_percent=overlap_percent
            # )
            #
            # # Xử lý dữ liệu với one-hot encoding
            # processor.process(
            #     train_path=train_path,
            #     val_path=val_path
            # )

            # word2vec_processor = DNASequenceProcessor(
            #     encoding_method="word2vec",
            #     kmer_size=6,
            #     min_size=min_size,
            #     max_size=max_size,
            #     overlap_percent=30,
            #     retrain_word2vec=False
            # )
            #
            # word2vec_processor.process(
            #     train_path=config.TRAIN_DATA_CSV_FILE,
            #     val_path=config.VAL_DATA_CSV_FILE
            # )



    # processor = DNASequenceProcessor(
    #     encoding_method="dna2vec",
    #     min_size=1200,
    #     max_size=1800,
    #     overlap_percent=30,
    #     dna2vec_method="average",
    # )
    #
    # processor.process(
    #     train_path=config.TRAIN_DATA_CSV_FILE,
    #     val_path=config.VAL_DATA_CSV_FILE,
    # )
