from abc import ABC, abstractmethod

from embedding_sequence.abstract_embedding import AbstractEmbedding
from embedding_sequence.fcgr.fcgr_embedding import FCGREmbedding
from embedding_sequence.one_hot.one_hot_embedding import OneHotEmbedding


class EmbeddingAbstractFactory(ABC):
    def __init__(self, embedding_type, min_size, max_size, overlap_percent, fold, is_train):
        self.embedding_type = embedding_type
        self.min_size = min_size
        self.max_size = max_size
        self.overlap_percent = overlap_percent
        self.fold = fold
        self.is_train = is_train

    @abstractmethod
    def create_embedding(self) -> AbstractEmbedding:
        pass


class FCGREmbeddingAbstractFactory(EmbeddingAbstractFactory):
    def create_embedding(self, kmer=6, resolution=64) -> FCGREmbedding:
        return FCGREmbedding(
            embedding_type=self.embedding_type,
            min_size=self.min_size,
            max_size=self.max_size,
            overlap_percent=self.overlap_percent,
            fold=self.fold,
            is_train=self.is_train,
            kmer=kmer,
            resolution=resolution
        )


class OneHotEmbeddingAbstractFactory(EmbeddingAbstractFactory):
    def create_embedding(self):
        return OneHotEmbedding(
            embedding_type=self.embedding_type,
            min_size=self.min_size,
            max_size=self.max_size,
            overlap_percent=self.overlap_percent,
            fold=self.fold,
            is_train=self.is_train,
        )
