import os

import pandas as pd

from common.env_config import config
from embedding_sequence.embedding_abstract_factory import EmbeddingAbstractFactory, FCGREmbeddingAbstractFactory


def create_contig(factory: EmbeddingAbstractFactory):
    embedder = factory.create_embedding()
    df = embedder.load_data()
    x, y = embedder.window_data(df)
    x_aug, y_aug = embedder.augment_data(x=x, y=y)
    if factory.is_train:
        x_resampled, y_resampled = embedder.resample_data(x=x_aug, y=y_aug)
        embedder.save_embedding(x_resampled, y_resampled)
    else:
        x_resampled = x_aug
        y_resampled = y_aug
        embedder.save_embedding(x_resampled, y_resampled)

    result = {
        'sequence': x_resampled,
        'target': y_resampled,
    }
    df = pd.DataFrame(result)
    # if embedder.is_train:
    #     output_path = os.path.join(embedder.output_dir,
    #                                f"{embedder.min_size}_{embedder.max_size}/fold_{embedder.fold}/train")
    # else:
    #     output_path = os.path.join(embedder.output_dir,
    #                                f"{embedder.min_size}_{embedder.max_size}/fold_{embedder.fold}/test")
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    df.to_csv(os.path.join(embedder.output_dir, "data.csv"), index=False)

    with open(os.path.join(embedder.output_dir, f"data.fa"), "a") as f:
        for idx, row in df.iterrows():
            if row['target'] == 0:
                header = f"contig_{idx}_temperate"
            else:
                header = f"contig_{idx}_virulent"
            f.write(">" + header + "\n")
            sequence = row["sequence"]
            f.write(sequence + "\n")


if __name__ == '__main__':
    for i in range(8):
        if i == 0:
            min_size = 100
            max_size = 400
            overlap = 10
        elif i == 1:
            min_size = 400
            max_size = 800
            overlap = 10
        elif i == 2:
            min_size = 800
            max_size = 1200
            overlap = 20
        elif i == 3:
            min_size = 1200
            max_size = 1800
            overlap = 30
        elif i == 4:
            min_size = 50
            max_size = 100
            overlap = 5
        elif i == 5:
            min_size = 100
            max_size = 200
            overlap = 5
        elif i == 6:
            min_size = 200
            max_size = 300
            overlap = 5
        elif i == 7:
            min_size = 300
            max_size = 400
            overlap = 5
        else:
            raise RuntimeError("Invalid index")

        for j in range(5):
            fold = j + 1
            create_contig(
                factory=FCGREmbeddingAbstractFactory(
                    data_dir=config.START_HERE,
                    output_dir=config.CONTIG_OUTPUT_DATA_DIR_FIX_BUG,
                    min_size=min_size,
                    max_size=max_size,
                    overlap_percent=overlap,
                    fold=fold,
                    is_train=True
                )
            )
            create_contig(
                factory=FCGREmbeddingAbstractFactory(
                    data_dir=config.START_HERE,
                    output_dir=config.CONTIG_OUTPUT_DATA_DIR_FIX_BUG,
                    min_size=min_size,
                    max_size=max_size,
                    overlap_percent=overlap,
                    fold=fold,
                    is_train=False
                )
            )
