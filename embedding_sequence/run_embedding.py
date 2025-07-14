from embedding_sequence.embedding_abstract_factory import EmbeddingAbstractFactory, FCGREmbeddingAbstractFactory


def create_embedding(factory: EmbeddingAbstractFactory):
    if isinstance(factory, FCGREmbeddingAbstractFactory):
        train_embedding = factory.create_embedding(kmer=6, resolution=16)
    else:
        raise NotImplementedError()

    train_df = train_embedding.load_data()
    # train_df = train_df.sample(frac=0.1).reset_index(drop=True)
    train_df = train_embedding.clean_data(train_df)
    x_train, y_train = train_embedding.window_data(train_df)
    x_train_aug, y_train_aug = train_embedding.augment_data(x=x_train, y=y_train)
    x_train_resampled, y_train_resampled = train_embedding.resample_data(x=x_train_aug, y=y_train_aug)
    x_embedding, y_embedding = train_embedding.encode_sequences(x_train_resampled, y_train_resampled)
    train_embedding.save_embedding(x_embedding, y_embedding)

    # train_embedding.experiment_with_parameters(x_train_resampled, y_train_resampled)

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
                factory=FCGREmbeddingAbstractFactory(
                    min_size=min_size,
                    max_size=max_size,
                    overlap_percent=30,
                    fold=fold,
                    is_train=True
                )
            )
            create_embedding(
                factory=FCGREmbeddingAbstractFactory(
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
