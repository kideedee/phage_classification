import os

import h5py
import numpy as np

from common.env_config import config
from embedding_sequence.embedding_abstract_factory import EmbeddingAbstractFactory, PFCGREmbeddingAbstractFactory


def create_embedding(factory: EmbeddingAbstractFactory):
    if isinstance(factory, PFCGREmbeddingAbstractFactory):
        embedder = factory.create_embedding(kmer=6)

        if factory.is_train:
            input_dir = os.path.join(config.CONTIG_OUTPUT_DATA_DIR,
                                     f"{factory.min_size}_{factory.max_size}/fold_{factory.fold}/train")
            output_dir = os.path.join(factory.output_dir,
                                      f"{factory.min_size}_{factory.max_size}/fold_{factory.fold}/train")
        else:
            input_dir = os.path.join(config.CONTIG_OUTPUT_DATA_DIR,
                                     f"{factory.min_size}_{factory.max_size}/fold_{factory.fold}/test")
            output_dir = os.path.join(factory.output_dir,
                                      f"{factory.min_size}_{factory.max_size}/fold_{factory.fold}/test")

        x_resampled = np.load(os.path.join(input_dir, "fcgr_vectors.npy"), allow_pickle=True)
        y_resampled = np.load(os.path.join(input_dir, "fcgr_labels.npy"))

        print(f"Processing group: {factory.min_size}_{factory.max_size}, fold: {factory.fold}")

        # Create output directory once before processing
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Process data in batches
        batch_num = 5
        batch_size = len(x_resampled) // batch_num  # Fixed: use len() instead of division

        # # Initialize lists to collect all embeddings
        # all_x_embeddings = []
        # all_y_embeddings = []

        for batch in range(batch_num):
            start = batch * batch_size
            # Handle last batch to include remaining samples
            if batch == batch_num - 1:
                end = len(x_resampled)
            else:
                end = (batch + 1) * batch_size

            x_batch = x_resampled[start:end]
            y_batch = y_resampled[start:end]

            print(f"Processing batch {batch + 1}/{batch_num}: samples {start} to {end}")

            # Fixed: use batch data instead of full dataset
            x_embedding, y_embedding = embedder.run(x_batch, y_batch)

            # # Collect embeddings from all batches
            # all_x_embeddings.append(x_embedding)
            # all_y_embeddings.append(y_embedding)
            #
            # # Combine all batch embeddings
            # final_x_embedding = np.concatenate(all_x_embeddings, axis=0)
            # final_y_embedding = np.concatenate(all_y_embeddings, axis=0)

            # Save combined embeddings to single file
            with h5py.File(os.path.join(output_dir, f"data_{batch+1}.h5"), "w") as f:
                f.create_dataset('vectors', data=x_embedding)
                f.create_dataset('labels', data=y_embedding)

            # print(f"Saved embeddings: {final_x_embedding.shape} vectors, {final_y_embedding.shape} labels")

    else:
        raise NotImplementedError("Factory type not supported")


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
            # train_fcgr_embedding_factory = FCGREmbeddingAbstractFactory(data_dir=config.CONTIG_OUTPUT_DATA_DIR,
            #                                                             output_dir=config.HDFS_FCGR_EMBEDDING_OUTPUT_DIR,
            #                                                             min_size=min_size,
            #                                                             max_size=max_size, overlap_percent=overlap,
            #                                                             fold=fold, is_train=True)
            # test_fcgr_embedding_factory = FCGREmbeddingAbstractFactory(data_dir=config.CONTIG_OUTPUT_DATA_DIR,
            #                                                            output_dir=config.HDFS_FCGR_EMBEDDING_OUTPUT_DIR,
            #                                                            min_size=min_size,
            #                                                            max_size=max_size, overlap_percent=overlap,
            #                                                            fold=fold,
            #                                                            is_train=False)
            train_embedding_factory = PFCGREmbeddingAbstractFactory(data_dir=config.CONTIG_OUTPUT_DATA_DIR,
                                                                    output_dir=config.HDFS_PFCGR_EMBEDDING_OUTPUT_DIR,
                                                                    min_size=min_size,
                                                                    max_size=max_size, overlap_percent=overlap,
                                                                    fold=fold, is_train=True)
            test_embedding_factory = PFCGREmbeddingAbstractFactory(data_dir=config.CONTIG_OUTPUT_DATA_DIR,
                                                                   output_dir=config.HDFS_PFCGR_EMBEDDING_OUTPUT_DIR,
                                                                   min_size=min_size,
                                                                   max_size=max_size, overlap_percent=overlap,
                                                                   fold=fold,
                                                                   is_train=False)

            create_embedding(factory=train_embedding_factory)
            create_embedding(factory=test_embedding_factory)
