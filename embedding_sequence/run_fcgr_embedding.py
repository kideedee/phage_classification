import os

import h5py
import numpy as np

from common.env_config import config
from embedding_sequence.embedding_abstract_factory import EmbeddingAbstractFactory, FCGREmbeddingAbstractFactory


def create_embedding(factory: EmbeddingAbstractFactory):
    if isinstance(factory, FCGREmbeddingAbstractFactory):
        embedder = factory.create_embedding(kmer=6)
        # df = embedder.load_data()
        # df = df.sample(frac=0.1).reset_index(drop=True)
        # df = embedder.clean_data(df)
        # x, y = embedder.window_data(df)
        # x_aug, y_aug = embedder.augment_data(x=x, y=y)
        # x_resampled, y_resampled = embedder.resample_data(x=x_aug, y=y_aug)

        if factory.is_train:
            input_dir = os.path.join(config.CONTIG_OUTPUT_DATA_DIR,
                                     f"{factory.min_size}_{factory.max_size}/fold_{fold}/train")
            output_dir = os.path.join(factory.output_dir,
                                      f"{factory.min_size}_{factory.max_size}/fold_{fold}/train")
        else:
            input_dir = os.path.join(config.CONTIG_OUTPUT_DATA_DIR,
                                     f"{factory.min_size}_{factory.max_size}/fold_{fold}/test")
            output_dir = os.path.join(factory.output_dir,
                                      f"{factory.min_size}_{factory.max_size}/fold_{fold}/test")

        x_resampled = np.load(os.path.join(input_dir, "fcgr_vectors.npy"), allow_pickle=True)
        y_resampled = np.load(os.path.join(input_dir, "fcgr_labels.npy"))

        print(f"Processing group: {factory.min_size}_{factory.max_size}, fold: {factory.fold}")
        x_embedding, y_embedding = embedder.run(x_resampled, y_resampled)
        # embedder.save_embedding(x_embedding, y_embedding)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with h5py.File(os.path.join(output_dir, "data.h5"), "w") as f:
            f.create_dataset('vectors', data=x_embedding)
            f.create_dataset('labels', data=y_embedding)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    for i in range(1):
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
            overlap = 30
        else:
            min_size = 1200
            max_size = 1800
            overlap = 30

        for j in range(1):
            fold = j + 1
            create_embedding(
                factory=FCGREmbeddingAbstractFactory(
                    data_dir=config.CONTIG_OUTPUT_DATA_DIR,
                    output_dir=config.HDFS_FCGR_EMBEDDING_OUTPUT_DIR,
                    min_size=min_size,
                    max_size=max_size,
                    overlap_percent=overlap,
                    fold=fold,
                    is_train=True
                )
            )
            create_embedding(
                factory=FCGREmbeddingAbstractFactory(
                    data_dir=config.CONTIG_OUTPUT_DATA_DIR,
                    output_dir=config.HDFS_FCGR_EMBEDDING_OUTPUT_DIR,
                    min_size=min_size,
                    max_size=max_size,
                    overlap_percent=overlap,
                    fold=fold,
                    is_train=False
                )
            )
