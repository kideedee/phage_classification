import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from embedding_sequence.abstract_embedding import AbstractEmbedding
from embedding_sequence.fcgr.pfcgr import PFCGR


class PFCGREmbedding(AbstractEmbedding):
    def __init__(self, data_dir=None, output_dir=None, min_size=None, max_size=None, overlap_percent=None, kmer=6,
                 fold=1, is_train=True):
        super().__init__(
            embedding_type="pfcgr",
            data_dir=data_dir,
            output_dir=output_dir,
            min_size=min_size,
            max_size=max_size,
            overlap_percent=overlap_percent,
            fold=fold,
            is_train=is_train
        )

        self.kmer = kmer
        self.pfcgr = PFCGR(kmer)
        self.counter = 0  # count number of time a sequence is converted to pfcgr

    def run(self, sequences, labels):
        pfcgr_representations = []
        result_labels = []

        df = pd.DataFrame(zip(sequences, labels), columns=['sequence', 'label'])
        results = Parallel(n_jobs=-1, verbose=True)(
            delayed(self._sequence_to_pfcgr)(
                (idx, row)
            ) for idx, row in df.iterrows()
        )

        for result in results:
            if result is not None and len(result) > 0:
                pfcgr_representations.append(result[0])
                result_labels.append(result[1])

        return np.array(pfcgr_representations), np.array(result_labels)

    def _sequence_to_pfcgr(self, idx_row):
        idx, row = idx_row
        sequence = row['sequence']
        label = row['label']
        seq = self.pfcgr.preprocessing(sequence)
        chaos = self.pfcgr.get_multichannel_image(seq)
        self.counter += 1
        return chaos, label
