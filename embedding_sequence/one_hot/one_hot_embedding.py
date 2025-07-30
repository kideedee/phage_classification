from typing import List, Any

import numpy as np
from tqdm import tqdm

from embedding_sequence.abstract_embedding import AbstractEmbedding


class OneHotEmbedding(AbstractEmbedding):

    def __init__(self, data_dir, output_dir, min_size, max_size, overlap_percent, is_train, fold):
        super().__init__(
            embedding_type="one_hot",
            data_dir=data_dir,
            output_dir=output_dir,
            min_size=min_size,
            max_size=max_size,
            overlap_percent=overlap_percent,
            is_train=is_train,
            fold=fold)

    def run(self, sequences: List[str], labels: List[str]) -> tuple[
        np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]
    ]:
        x = np.array(
            [self.one_hot_encode_sequence(seq) for seq in tqdm(sequences, desc="Encoding sequences")]
        )
        return x, np.array(labels)

    def one_hot_encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Mã hóa một chuỗi DNA theo định dạng one-hot encoding theo DeePhage.
        """
        # Chuẩn hóa chuỗi DNA (chuyển về chữ in hoa)
        sequence = sequence.upper()

        # Xác định độ dài tối đa
        max_length = self.max_size

        # Tạo ma trận kết quả (tất cả là 0)
        one_hot_matrix = np.zeros((max_length, 4), dtype=np.float32)

        # Tạo mapping từ nucleotide sang one-hot vector
        mapping = {
            'A': [0, 0, 0, 1],
            'C': [0, 0, 1, 0],
            'G': [0, 1, 0, 0],
            'T': [1, 0, 0, 0],
            'N': [0, 0, 0, 0]  # Xử lý trường hợp 'N'
        }

        # Điền ma trận với giá trị one-hot
        for i, nucleotide in enumerate(sequence[:max_length]):
            if nucleotide in mapping:
                one_hot_matrix[i] = mapping[nucleotide]

        return one_hot_matrix
