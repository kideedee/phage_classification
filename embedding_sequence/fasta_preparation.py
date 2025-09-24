import os

import pandas as pd

from common.env_config import config
from embedding_sequence.embedding_abstract_factory import EmbeddingAbstractFactory, OneHotEmbeddingAbstractFactory
from logger.phg_cls_log import embedding_log as log


def create_embedding(factory: EmbeddingAbstractFactory):
    train_embedding = factory.create_embedding()
    log.info(f"Creating embedding, fold: {train_embedding.fold}, is_train: {train_embedding.is_train}")

    train_df = train_embedding.load_data()
    # train_df = train_df.sample(frac=0.005).reset_index(drop=True)
    train_df = train_embedding.clean_data(train_df)
    x_train, y_train = train_embedding.window_data(train_df)
    x_train_aug, y_train_aug = train_embedding.augment_data(x=x_train, y=y_train)
    x_train_resampled, y_train_resampled = train_embedding.resample_data(x=x_train_aug, y=y_train_aug)

    df = pd.DataFrame(zip(x_train_resampled, y_train_resampled), columns=['sequence', 'label'])

    output_dir = train_embedding.output_dir
    if train_embedding.is_train:
        output_dir = output_dir + "/train"
    else:
        output_dir = output_dir + "/test"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, f"data.fa"), "a") as f:
        for idx, row in df.iterrows():
            if row['label'] == 0:
                header = f"contig_{idx}_temperate"
            else:
                header = f"contig_{idx}_virulent"
            f.write(">" + header + "\n")
            sequence = row["sequence"]
            f.write(sequence + "\n")


if __name__ == '__main__':
    for j in range(3):
        if j == 0:
            min_length = 100
            max_length = 200
        elif j == 1:
            min_length = 200
            max_length = 300
        elif j == 2:
            min_length = 300
            max_length = 400
        else:
            raise ValueError


        for i in range(5):
            fold = i + 1
            data_dir = config.CONTIG_OUTPUT_DATA_DIR
            output_dir = os.path.join(config.CONTIG_OUTPUT_DATA_DIR,
                                    f"fasta/{min_length}_{max_length}/{fold}")

            create_embedding(
                factory=OneHotEmbeddingAbstractFactory(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    min_size=min_length,
                    max_size=max_length,
                    overlap_percent=10,
                    fold=i + 1,
                    is_train=True,
                )
            )

            create_embedding(
                factory=OneHotEmbeddingAbstractFactory(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    min_size=min_length,
                    max_size=max_length,
                    overlap_percent=10,
                    fold=i + 1,
                    is_train=False,
                )
            )
