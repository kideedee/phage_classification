from abc import ABC, abstractmethod

from embedding_sequence.abstract_embedding import AbstractEmbedding
from embedding_sequence.cnn_embedding.cnn_embedding import CNNEmbedding
from embedding_sequence.codon.codon_embedding import CodonEmbedding
from embedding_sequence.dnabert2.dnabert_2_embedding import DNABert2Embedding
from embedding_sequence.fcgr.fcgr_embedding import FCGREmbedding
from embedding_sequence.fcgr.pfcgr_embedding import PFCGREmbedding
from embedding_sequence.one_hot.one_hot_embedding import OneHotEmbedding


class EmbeddingAbstractFactory(ABC):
    def __init__(self, data_dir, output_dir, min_size, max_size, overlap_percent, fold, is_train):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.min_size = min_size
        self.max_size = max_size
        self.overlap_percent = overlap_percent
        self.fold = fold
        self.is_train = is_train

    @abstractmethod
    def create_embedding(self) -> AbstractEmbedding:
        pass


class FCGREmbeddingAbstractFactory(EmbeddingAbstractFactory):
    def create_embedding(self, kmer=6) -> FCGREmbedding:
        return FCGREmbedding(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            min_size=self.min_size,
            max_size=self.max_size,
            overlap_percent=self.overlap_percent,
            fold=self.fold,
            is_train=self.is_train,
            kmer=kmer
        )


class OneHotEmbeddingAbstractFactory(EmbeddingAbstractFactory):
    def create_embedding(self):
        return OneHotEmbedding(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            min_size=self.min_size,
            max_size=self.max_size,
            overlap_percent=self.overlap_percent,
            fold=self.fold,
            is_train=self.is_train,
        )


class CodonEmbeddingAbstractFactory(EmbeddingAbstractFactory):
    def create_embedding(self, preprocessing_method="padding"):
        return CodonEmbedding(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            min_size=self.min_size,
            max_size=self.max_size,
            overlap_percent=self.overlap_percent,
            fold=self.fold,
            is_train=self.is_train,
            preprocess_method=preprocessing_method
        )


class PFCGREmbeddingAbstractFactory(EmbeddingAbstractFactory):
    def create_embedding(self, kmer=6) -> PFCGREmbedding:
        print(f"Start creating PFCGREmbedding...")
        return PFCGREmbedding(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            min_size=self.min_size,
            max_size=self.max_size,
            overlap_percent=self.overlap_percent,
            fold=self.fold,
            is_train=self.is_train,
            kmer=kmer
        )

class DNABert2EmbeddingAbstractFactory(EmbeddingAbstractFactory):
    def create_embedding(self, kmer=6) -> DNABert2Embedding:
        print(f"Start creating DNABert2 Embedding...")
        return DNABert2Embedding(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            min_size=self.min_size,
            max_size=self.max_size,
            overlap_percent=self.overlap_percent,
            fold=self.fold,
            is_train=self.is_train
        )


class CNNEmbeddingAbstractFactory(EmbeddingAbstractFactory):
    def create_embedding(self, kmer=6) -> CNNEmbedding:
        print(f"Start creating CNN Embedding...")
        return CNNEmbedding(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            min_size=self.min_size,
            max_size=self.max_size,
            overlap_percent=self.overlap_percent,
            fold=self.fold,
            is_train=self.is_train
        )