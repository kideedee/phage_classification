from abc import ABC, abstractmethod

from embedding_sequence.abstract_embedding import AbstractEmbedding
from embedding_sequence.fcgr.fcgr_embedding import FCGREmbedding


class EmbeddingAbstractFactory(ABC):

    @abstractmethod
    def create_fcgr_embedding(self) -> AbstractEmbedding:
        pass


class FCGREmbeddingAbstractFactory(EmbeddingAbstractFactory):
    def create_fcgr_embedding(self) -> FCGREmbedding:
        return FCGREmbedding()
