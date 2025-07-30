from common.env_config import config
from embedding_sequence.embedding_abstract_factory import EmbeddingAbstractFactory, FCGREmbeddingAbstractFactory


def create_contig(factory: EmbeddingAbstractFactory):
    embedder = factory.create_embedding()
    df = embedder.load_data()
    x, y = embedder.window_data(df)
    x_aug, y_aug = embedder.augment_data(x=x, y=y)
    x_resampled, y_resampled = embedder.resample_data(x=x_aug, y=y_aug)
    embedder.save_embedding(x_resampled, y_resampled)


if __name__ == '__main__':
    for i in range(4):
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

        for j in range(5):
            fold = j + 1
            create_contig(
                factory=FCGREmbeddingAbstractFactory(
                    data_dir=config.START_HERE,
                    output_dir=config.CONTIG_OUTPUT_DATA_DIR,
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
                    output_dir=config.CONTIG_OUTPUT_DATA_DIR,
                    min_size=min_size,
                    max_size=max_size,
                    overlap_percent=overlap,
                    fold=fold,
                    is_train=False
                )
            )
