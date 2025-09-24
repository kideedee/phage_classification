import os

import h5py
import numpy as np
import torch
from sklearn.utils import shuffle
from tqdm import tqdm

from common.env_config import config
from embedding_sequence.embedding_abstract_factory import EmbeddingAbstractFactory, PFCGREmbeddingAbstractFactory, \
    OneHotEmbeddingAbstractFactory, DNABert2EmbeddingAbstractFactory, CNNEmbeddingAbstractFactory
from logger.phg_cls_log import log


def aggregate_h5_files(data_dir, input_files, output_file):
    """
    Aggregate multiple H5 files into a single file (memory efficient)
    """
    # First pass: determine total dimensions
    total_samples = 0
    vector_shape = None
    label_shape = None

    print("Scanning files to determine dimensions...")
    for input_file in input_files:
        input_path = os.path.join(data_dir, input_file)
        with h5py.File(input_path, 'r') as f_in:
            vectors = f_in['vectors']
            labels = f_in['labels']

            if vector_shape is None:
                vector_shape = vectors.shape[1:]  # All dimensions except first
                label_shape = labels.shape[1:] if len(labels.shape) > 1 else ()

            total_samples += vectors.shape[0]
            print(f"  {input_file}: {vectors.shape[0]} samples")

    print(f"Total samples to aggregate: {total_samples}")
    print(f"Vector shape per sample: {vector_shape}")

    # Create output file with pre-allocated datasets
    output_path = os.path.join(data_dir, output_file)
    with h5py.File(output_path, 'w') as f_out:
        # Create datasets with known total size
        if len(vector_shape) > 0:
            full_vector_shape = (total_samples,) + vector_shape
        else:
            full_vector_shape = (total_samples,)

        if len(label_shape) > 0:
            full_label_shape = (total_samples,) + label_shape
        else:
            full_label_shape = (total_samples,)

        vectors_dset = f_out.create_dataset('vectors', shape=full_vector_shape,
                                            dtype=np.float16)  # Adjust dtype as needed
        labels_dset = f_out.create_dataset('labels', shape=full_label_shape,
                                           dtype=np.float16)  # Adjust dtype as needed

        # Second pass: copy data chunk by chunk
        current_idx = 0
        for i, input_file in enumerate(input_files):
            input_path = os.path.join(data_dir, input_file)
            print(f"Processing {input_file}...")

            with h5py.File(input_path, 'r') as f_in:
                vectors = f_in['vectors']
                labels = f_in['labels']

                num_samples = vectors.shape[0]
                end_idx = current_idx + num_samples

                # Copy data directly without loading into memory
                vectors_dset[current_idx:end_idx] = vectors[:]
                labels_dset[current_idx:end_idx] = labels[:]

                current_idx = end_idx
                print(f"  Copied {num_samples} samples (total so far: {current_idx})")

        # # Add metadata
        # f_out.attrs['source_files'] = [f.encode('utf-8') for f in input_files]
        # f_out.attrs['num_source_files'] = len(input_files)
        # f_out.attrs['total_samples'] = total_samples

    print(f"Successfully aggregated {len(input_files)} files into {output_file}")
    print(f"Total samples: {total_samples}")


def build_input_output_dir(factory: EmbeddingAbstractFactory):
    if factory.is_train:
        input_dir = os.path.join(factory.data_dir,
                                 f"{factory.min_size}_{factory.max_size}/fold_{factory.fold}/train")
        output_dir = os.path.join(factory.output_dir,
                                  f"{factory.min_size}_{factory.max_size}/fold_{factory.fold}/train")
    else:
        input_dir = os.path.join(factory.data_dir,
                                 f"{factory.min_size}_{factory.max_size}/fold_{factory.fold}/test")
        output_dir = os.path.join(factory.output_dir,
                                  f"{factory.min_size}_{factory.max_size}/fold_{factory.fold}/test")

    return input_dir, output_dir


def create_embedding(factory: EmbeddingAbstractFactory):
    if isinstance(factory, PFCGREmbeddingAbstractFactory):
        embedder = factory.create_embedding(kmer=6)

        input_dir, output_dir = build_input_output_dir(factory)

        x = np.load(os.path.join(input_dir, "fcgr_vectors.npy"), allow_pickle=True)
        y = np.load(os.path.join(input_dir, "fcgr_labels.npy"))

        # n_samples = 10000  # adjust as needed
        # random_indices = np.random.choice(len(x), size=n_samples, replace=False)
        # x = x[random_indices]
        # y = y[random_indices]

        x, y = shuffle(x, y, random_state=42)

        print(f"Processing group: {factory.min_size}_{factory.max_size}, fold: {factory.fold}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        batch_num = 5
        batch_size = len(x) // batch_num

        for batch in range(batch_num):
            start = batch * batch_size
            if batch == batch_num - 1:
                end = len(x)
            else:
                end = (batch + 1) * batch_size

            x_batch = x[start:end]
            y_batch = y[start:end]

            print(f"Processing batch {batch + 1}/{batch_num}: samples {start} to {end}")

            x_embedding, y_embedding = embedder.run(x_batch, y_batch)
            with h5py.File(os.path.join(output_dir, f"data_{batch + 1}.h5"), "w") as f:
                f.create_dataset('vectors', data=x_embedding)
                f.create_dataset('labels', data=y_embedding)

            del x_batch, y_batch

        del x, y
        print(f"Start aggregating files...")
        input_files = ['data_1.h5', 'data_2.h5', 'data_3.h5', 'data_4.h5', 'data_5.h5']
        aggregate_h5_files(output_dir, input_files, 'data.h5')
        for file in input_files:
            file_path = os.path.join(output_dir, file)
            os.remove(file_path)

    elif isinstance(factory, OneHotEmbeddingAbstractFactory):
        embedder = factory.create_embedding()

        input_dir, output_dir = build_input_output_dir(factory)

        x = np.load(os.path.join(input_dir, "fcgr_vectors.npy"), allow_pickle=True)
        y = np.load(os.path.join(input_dir, "fcgr_labels.npy"))

        # n_samples = 10000  # adjust as needed
        # random_indices = np.random.choice(len(x), size=n_samples, replace=False)
        # x = x[random_indices]
        # y = y[random_indices]

        x, y = shuffle(x, y, random_state=42)

        print(f"Processing group: {factory.min_size}_{factory.max_size}, fold: {factory.fold}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_embedding, y_embedding = embedder.run(x, y)
        np.save(os.path.join(output_dir, f"vectors.npy"), x_embedding)
        np.save(os.path.join(output_dir, f"labels.npy"), y_embedding)
        with h5py.File(os.path.join(output_dir, f"data.h5"), "w") as f:
            f.create_dataset('vectors', data=x_embedding)
            f.create_dataset('labels', data=y_embedding)

    elif isinstance(factory, DNABert2EmbeddingAbstractFactory):
        embedder = factory.create_embedding()
        input_dir, output_dir = build_input_output_dir(factory)
        x = np.load(os.path.join(input_dir, "fcgr_vectors.npy"), allow_pickle=True)
        y = np.load(os.path.join(input_dir, "fcgr_labels.npy"))
        x, y = shuffle(x, y, random_state=42)

        # n_samples = 1000  # adjust as needed
        # random_indices = np.random.choice(len(x), size=n_samples, replace=False)
        # x = x[random_indices]
        # y = y[random_indices]

        log.info(
            f"DNABert2 embedding is processing data, group: {factory.min_size}_{factory.max_size}, fold: {factory.fold}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_embedding, y_embedding = embedder.run(x, y)
        with h5py.File(os.path.join(output_dir, f"data.h5"), "w") as f:
            f.create_dataset('vectors', data=x_embedding)
            f.create_dataset('labels', data=y_embedding)
    elif isinstance(factory, CNNEmbeddingAbstractFactory):
        log.info(
            f"CNN embedding is processing data, group: {factory.min_size}_{factory.max_size}, fold: {factory.fold}")

        embedder = factory.create_embedding()
        batch_size = 128
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_dir, output_dir = build_input_output_dir(factory)
        with h5py.File(os.path.join(input_dir, "data.h5"), "r") as f:
            length = len(f['labels'])

        all_features = []
        all_labels = []
        total_batches = (length + batch_size - 1) // batch_size
        for i in tqdm(range(0, length, batch_size), desc="Processing batches", total=total_batches):
            with h5py.File(os.path.join(input_dir, "data.h5"), "r") as f:
                batch_vectors = f['vectors'][i:i + batch_size]
                batch_labels = f['labels'][i:i + batch_size]

                batch_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in batch_vectors])
                batch_tensor = batch_tensor.to(device)  # Move data to GPU

                with torch.no_grad():
                    features = embedder.model(batch_tensor)
                    batch_features = features.cpu().numpy()

                all_features.extend(batch_features)
                all_labels.extend(batch_labels)

        # n_samples = 1000  # adjust as needed
        # random_indices = np.random.choice(len(x), size=n_samples, replace=False)
        # x = x[random_indices]
        # y = y[random_indices]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # x_embedding, y_embedding = embedder.run(x, y)
        with h5py.File(os.path.join(output_dir, f"data.h5"), "w") as f:
            f.create_dataset('vectors', data=np.array(all_features))
            f.create_dataset('labels', data=np.array(all_labels))
    else:
        raise NotImplementedError("Factory type not supported")


if __name__ == '__main__':
    for i in range(4, 8):
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
        elif i == 3:
            min_size = 1200
            max_size = 1800
            overlap = 30
        elif i == 4:
            min_size = 50
            max_size = 100
            overlap = 10
        elif i == 5:
            min_size = 100
            max_size = 200
            overlap = 10
        elif i == 6:
            min_size = 200
            max_size = 300
            overlap = 10
        elif i == 7:
            min_size = 300
            max_size = 400
            overlap = 10
        else:
            raise RuntimeError("invalid index")

        for j in range(5):
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
            # train_embedding_factory = PFCGREmbeddingAbstractFactory(data_dir=config.CONTIG_OUTPUT_DATA_DIR,
            #                                                         output_dir=config.NORMALIZED_HDFS_PFCGR_EMBEDDING_OUTPUT_DIR,
            #                                                         min_size=min_size,
            #                                                         max_size=max_size, overlap_percent=overlap,
            #                                                         fold=fold, is_train=True)
            # test_embedding_factory = PFCGREmbeddingAbstractFactory(data_dir=config.CONTIG_OUTPUT_DATA_DIR,
            #                                                        output_dir=config.NORMALIZED_HDFS_PFCGR_EMBEDDING_OUTPUT_DIR,
            #                                                        min_size=min_size,
            #                                                        max_size=max_size, overlap_percent=overlap,
            #                                                        fold=fold,
            #                                                        is_train=False)

            train_embedding_factory = OneHotEmbeddingAbstractFactory(data_dir=config.CONTIG_OUTPUT_DATA_DIR_FIX_BUG,
                                                                     output_dir=config.EMBEDDING_OUTPUT_DATA_DIR + "/onehot_after_fix_bug_contig",
                                                                     min_size=min_size,
                                                                     max_size=max_size, overlap_percent=overlap,
                                                                     fold=fold, is_train=True)
            test_embedding_factory = OneHotEmbeddingAbstractFactory(data_dir=config.CONTIG_OUTPUT_DATA_DIR_FIX_BUG,
                                                                    output_dir=config.EMBEDDING_OUTPUT_DATA_DIR + "/onehot_after_fix_bug_contig",
                                                                    min_size=min_size,
                                                                    max_size=max_size, overlap_percent=overlap,
                                                                    fold=fold,
                                                                    is_train=False)

            # train_embedding_factory = DNABert2EmbeddingAbstractFactory(data_dir=config.CONTIG_OUTPUT_DATA_DIR,
            #                                                            output_dir=config.DNA_BERT_2_EMBEDDING_OUTPUT_DIR,
            #                                                            min_size=min_size,
            #                                                            max_size=max_size, overlap_percent=overlap,
            #                                                            fold=fold, is_train=True)
            # test_embedding_factory = DNABert2EmbeddingAbstractFactory(data_dir=config.CONTIG_OUTPUT_DATA_DIR,
            #                                                           output_dir=config.DNA_BERT_2_EMBEDDING_OUTPUT_DIR,
            #                                                           min_size=min_size,
            #                                                           max_size=max_size, overlap_percent=overlap,
            #                                                           fold=fold,
            #                                                           is_train=False)

            # train_embedding_factory = CNNEmbeddingAbstractFactory(data_dir=config.ONE_HOT_EMBEDDING_OUT_PUT_DIR,
            #                                                       output_dir=config.CNN_EMBEDDING_OUTPUT_DIR,
            #                                                       min_size=min_size,
            #                                                       max_size=max_size, overlap_percent=overlap,
            #                                                       fold=fold, is_train=True)
            # test_embedding_factory = CNNEmbeddingAbstractFactory(data_dir=config.ONE_HOT_EMBEDDING_OUT_PUT_DIR,
            #                                                      output_dir=config.CNN_EMBEDDING_OUTPUT_DIR,
            #                                                      min_size=min_size,
            #                                                      max_size=max_size, overlap_percent=overlap,
            #                                                      fold=fold,
            #                                                      is_train=False)

            create_embedding(factory=train_embedding_factory)
            create_embedding(factory=test_embedding_factory)
