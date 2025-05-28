from embedding_sequence.embedding_abstract_factory import EmbeddingAbstractFactory, FCGREmbeddingAbstractFactory


def client_code(factory: EmbeddingAbstractFactory):
    client = factory.create_fcgr_embedding()
    client.load_data("123")


if __name__ == '__main__':
    client_code(factory=FCGREmbeddingAbstractFactory())
