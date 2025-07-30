import os

from common.env_config import config
from embedding_sequence.embedding_abstract_factory import EmbeddingAbstractFactory, FCGREmbeddingAbstractFactory, \
    CodonEmbeddingAbstractFactory


def create_embedding(factory: EmbeddingAbstractFactory):
    if isinstance(factory, FCGREmbeddingAbstractFactory):
        embedder = factory.create_embedding(kmer=6, resolution=16)
        df = embedder.load_data()
        df = df.sample(frac=0.1).reset_index(drop=True)
        df = embedder.clean_data(df)
        x, y = embedder.window_data(df)
        x_aug, y_aug = embedder.augment_data(x=x, y=y)
        x_resampled, y_resampled = embedder.resample_data(x=x_aug, y=y_aug)
        x_embedding, y_embedding = embedder.run(x_resampled, y_resampled)
        embedder.save_embedding(x_embedding, y_embedding)
    elif isinstance(factory, CodonEmbeddingAbstractFactory):
        embedder = factory.create_embedding(preprocessing_method="padding")

        df = embedder.load_data()
        df = df.sample(frac=0.1).reset_index(drop=True)
        df = embedder.clean_data(df)
        x, y = embedder.window_data(df)
        x_aug, y_aug = embedder.augment_data(x=x, y=y)
        x_resampled, y_resampled = embedder.resample_data(x=x_aug, y=y_aug)
        codon_df = embedder.run(x_resampled, y_resampled)
        codon_df.to_csv(os.path.join(embedder.output_dir, "data.csv"), index=False)
    else:
        raise NotImplementedError()

    # embedder.experiment_with_parameters(x_resampled, y_resampled)

    # fcgr_matrix = x_embedding[0]
    # plt.figure(figsize=(8, 8))
    # plt.imshow(fcgr_matrix, cmap='hot', interpolation='nearest')
    # plt.colorbar(label='Frequency')
    # plt.xlabel('X coordinate')
    # plt.ylabel('Y coordinate')
    # plt.show()


if __name__ == '__main__':
    for i in range(4):
        if i == 0:
            min_size = 100
            max_size = 400
        elif i == 1:
            min_size = 400
            max_size = 800
        elif i == 2:
            min_size = 800
            max_size = 1200
        else:
            min_size = 1200
            max_size = 1800

        for j in range(5):
            fold = j + 1
            create_embedding(
                factory=CodonEmbeddingAbstractFactory(
                    data_dir=config.START_HERE,
                    output_dir=config.CODON_EMBEDDING_OUTPUT_DIR,
                    min_size=min_size,
                    max_size=max_size,
                    overlap_percent=30,
                    fold=fold,
                    is_train=True
                )
            )
            create_embedding(
                factory=CodonEmbeddingAbstractFactory(
                    data_dir=config.START_HERE,
                    output_dir=config.CODON_EMBEDDING_OUTPUT_DIR,
                    min_size=min_size,
                    max_size=max_size,
                    overlap_percent=30,
                    fold=fold,
                    is_train=False
                )
            )
    # create_embedding(
    #     factory=OneHotEmbeddingAbstractFactory(
    #         embedding_type="onehot",
    #         min_size=100,
    #         max_size=400,
    #         overlap_percent=30,
    #         fold=1,
    #         is_train=False,
    #     )
    # )
