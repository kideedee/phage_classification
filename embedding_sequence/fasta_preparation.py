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

    data_dir = os.path.join(config.MY_DATA_DIR,
                            f"fasta/{train_embedding.min_size}_{train_embedding.max_size}/{train_embedding.fold}")
    if train_embedding.is_train:
        data_dir = data_dir + "/train"
    else:
        data_dir = data_dir + "/test"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(os.path.join(data_dir, f"data.fa"), "a") as f:
        for idx, row in df.iterrows():
            if row['label'] == 0:
                header = f"contig_{idx}_temperate"
            else:
                header = f"contig_{idx}_virulent"
            f.write(">" + header + "\n")
            sequence = row["sequence"]
            f.write(sequence + "\n")


if __name__ == '__main__':
    # create_embedding(factory=FCGREmbeddingAbstractFactory())
    for j in range(4):
        if j == 0:
            min_length = 100
            max_length = 400
        elif j == 1:
            min_length = 400
            max_length = 800
        elif j == 2:
            min_length = 800
            max_length = 1200
        elif j == 3:
            min_length = 1200
            max_length = 1800
        else:
            raise ValueError

        for i in range(5):
            create_embedding(
                factory=OneHotEmbeddingAbstractFactory(
                    embedding_type="onehot",
                    min_size=min_length,
                    max_size=max_length,
                    overlap_percent=30,
                    fold=i + 1,
                    is_train=True,
                )
            )

            create_embedding(
                factory=OneHotEmbeddingAbstractFactory(
                    embedding_type="onehot",
                    min_size=min_length,
                    max_size=max_length,
                    overlap_percent=30,
                    fold=i + 1,
                    is_train=False,
                )
            )
