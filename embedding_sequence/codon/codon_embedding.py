from typing import List

import pandas as pd
from Bio.Seq import Seq
from joblib import Parallel, delayed

from embedding_sequence.abstract_embedding import AbstractEmbedding


class CodonEmbedding(AbstractEmbedding):
    def __init__(self, data_dir, output_dir, min_size, max_size, overlap_percent, fold, is_train, preprocess_method):
        super().__init__(
            embedding_type="codon",
            data_dir=data_dir,
            output_dir=output_dir,
            min_size=min_size,
            max_size=max_size,
            overlap_percent=overlap_percent,
            is_train=is_train,
            fold=fold
        )

        self.preprocess_method = preprocess_method

    def run(self, sequences: List[str], labels: List[str]) -> pd.DataFrame:
        proteins = []
        result_labels = []
        df = pd.DataFrame(zip(sequences, labels), columns=["sequence", "label"])
        results = Parallel(n_jobs=-1)(
            delayed(self._sequence_to_codon)(
                (idx, row)
            ) for idx, row in df.iterrows()
        )
        for result in results:
            if result:
                proteins.append(result[0])
                result_labels.append(result[1])

        return pd.DataFrame(zip(proteins, result_labels), columns=["sequence", "label"])

    def _sequence_to_codon(self, row_tuple):
        idx, row = row_tuple
        sequence = row["sequence"]
        label = row["label"]
        dna_seq = self._preprocessing_sequence(sequence)
        protein = Seq(dna_seq).translate()
        return protein, label

    def _preprocessing_sequence(self, dna_seq):
        if self.preprocess_method == 'padding':
            remainder = len(dna_seq) % 3
            if remainder != 0:
                padding = 3 - remainder
                dna_padded = dna_seq + "N" * padding
            else:
                dna_padded = dna_seq

            return dna_padded
        elif self.preprocess_method == 'trim':
            trimmed_length = len(dna_seq) - (len(dna_seq) % 3)
            dna_trimmed = dna_seq[:trimmed_length]
            return dna_trimmed
        else:
            raise NotImplementedError(f"{self.preprocess_method} codon preprocessing method is not implemented.")
