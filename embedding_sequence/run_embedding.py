from embedding_sequence.embedding_abstract_factory import EmbeddingAbstractFactory, FCGREmbeddingAbstractFactory
from logger.phg_cls_log import embedding_log as log


def create_embedding(factory: EmbeddingAbstractFactory):
    log.info("One hot embedding")

    if isinstance(factory, FCGREmbeddingAbstractFactory):
        train_embedding = factory.create_embedding(kmer=6, resolution=64)
    else:
        train_embedding = factory.create_embedding()

    train_df = train_embedding.load_data()
    # train_df = train_df.sample(frac=0.1).reset_index(drop=True)
    train_df = train_embedding.clean_data(train_df)
    x_train, y_train = train_embedding.window_data(train_df)
    x_train_aug, y_train_aug = train_embedding.augment_data(x=x_train, y=y_train)
    x_train_resampled, y_train_resampled = train_embedding.resample_data(x=x_train_aug, y=y_train_aug)

    x_embedding, y_embedding = train_embedding.encode_sequences(x_train_resampled, y_train_resampled)
    train_embedding.save_embedding(x_embedding, y_embedding)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(x_train_resampled[0])
    print(y_train_resampled[0])

    print("==============")

    print(x_embedding[0])
    print(y_embedding[0])


if __name__ == '__main__':
    create_embedding(
        factory=FCGREmbeddingAbstractFactory(
            min_size=100,
            max_size=400,
            overlap_percent=30,
            fold=1,
            is_train=True
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
