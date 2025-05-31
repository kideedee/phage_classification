"""
Chương trình sử dụng mô hình dna2vec đã được huấn luyện sẵn để embedding chuỗi DNA
"""
import os
import sys

import numpy as np
from gensim.models import KeyedVectors

from common.env_config import config


class SingleKModel:
    """
    Mô hình cho một độ dài k-mer cụ thể.
    """

    def __init__(self, model):
        self.model = model
        self.vocab_lst = sorted(model.key_to_index.keys())


class MultiKModel:
    """
    Mô hình tổng hợp cho nhiều độ dài k-mer khác nhau.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.logger = None  # Sẽ được khởi tạo nếu cần logging

        # Tải mô hình từ file
        self.aggregate = KeyedVectors.load_word2vec_format(filepath, binary=False)

        # Tách mô hình thành các mô hình con cho từng độ dài k-mer
        self.k_models = {}
        for vocab in self.aggregate.key_to_index:
            k = len(vocab)
            if k not in self.k_models:
                self.k_models[k] = {}
            self.k_models[k][vocab] = self.aggregate[vocab]

        # Tạo các mô hình SingleKModel
        self.models = {k: self._make_dict_model(k) for k in self.k_models}

    def _make_dict_model(self, k):
        """
        Tạo mô hình từ điển cho một độ dài k cụ thể.
        """
        vectors = np.array([self.k_models[k][vocab] for vocab in sorted(self.k_models[k].keys())])

        # Tạo một KeyedVectors model mới
        model = KeyedVectors(self.aggregate.vector_size)
        model.add_vectors(sorted(self.k_models[k].keys()), vectors)

        return SingleKModel(model)

    def model(self, k):
        """
        Trả về mô hình cho độ dài k.
        """
        return self.models[k]

    def vector(self, km):
        """
        Trả về vector của k-mer.
        """
        return self.aggregate[km]

    def cosine_distance(self, km1, km2):
        """
        Tính khoảng cách cosine giữa hai k-mer.
        """
        return self.aggregate.similarity(km1, km2)


def embed_dna_sequence(sequence, model, k_min=3, k_max=8, method='average'):
    """
    Embedding một chuỗi DNA bằng cách chia thành các k-mer và tổng hợp các vector.

    Args:
        sequence: Chuỗi DNA cần embedding
        model: Mô hình MultiKModel đã được tải
        k_min: Độ dài k-mer tối thiểu
        k_max: Độ dài k-mer tối đa
        method: Phương pháp tổng hợp ('average', 'sum', 'concat')

    Returns:
        Vector biểu diễn của chuỗi DNA
    """
    # Loại bỏ các ký tự không hợp lệ
    sequence = ''.join([c for c in sequence.upper() if c in 'ACGT'])

    # Tạo các k-mer từ chuỗi DNA
    kmers = []
    for k in range(k_min, k_max + 1):
        if k > len(sequence):
            continue
        kmers.extend([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

    # Lấy vector của từng k-mer
    kmer_vectors = []
    for kmer in kmers:
        try:
            vec = model.vector(kmer)
            kmer_vectors.append(vec)
        except KeyError:
            # Bỏ qua nếu k-mer không có trong mô hình
            continue

    # Tổng hợp các vector
    if not kmer_vectors:
        return None

    if method == 'average':
        return np.mean(kmer_vectors, axis=0)
    elif method == 'sum':
        return np.sum(kmer_vectors, axis=0)
    elif method == 'concat':
        # Tạo một vector lớn bằng cách nối tất cả các vector lại
        # Giới hạn số lượng k-mer để tránh vector quá lớn
        max_kmers = 100
        selected_kmers = kmer_vectors[:max_kmers]
        return np.concatenate(selected_kmers)
    else:
        raise ValueError(f"Phương pháp {method} không được hỗ trợ")


def calculate_similarity(seq1, seq2, model, method='average'):
    """
    Tính độ tương đồng giữa hai chuỗi DNA.

    Args:
        seq1, seq2: Hai chuỗi DNA cần so sánh
        model: Mô hình MultiKModel
        method: Phương pháp tổng hợp vector

    Returns:
        Độ tương đồng cosine giữa hai chuỗi
    """
    vec1 = embed_dna_sequence(seq1, model, method=method)
    vec2 = embed_dna_sequence(seq2, model, method=method)

    if vec1 is None or vec2 is None:
        return None

    # Tính độ tương đồng cosine
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity


def main():
    # Đường dẫn đến file mô hình
    # Bạn cần tải file mô hình từ trang của tác giả hoặc từ GitHub
    model_path = os.path.join(config.MODEL_DIR, "dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v")

    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file mô hình tại {model_path}")
        print("Vui lòng tải mô hình từ trang GitHub của tác giả: https://github.com/pnpnpn/dna2vec")
        sys.exit(1)

    # Tải mô hình
    print("Đang tải mô hình dna2vec...")
    model = MultiKModel(model_path)
    print("Đã tải mô hình thành công!")

    # Thử nghiệm với một số k-mer đơn giản
    print("\nThử nghiệm với các k-mer đơn giản:")
    try:
        print(f"Vector của AAA: {model.vector('AAA')[:5]}...")  # Chỉ hiển thị 5 giá trị đầu tiên
        print(f"Độ tương đồng giữa AAA và GCT: {model.cosine_distance('AAA', 'GCT')}")
        print(f"Độ tương đồng giữa AAA và AAAA: {model.cosine_distance('AAA', 'AAAA')}")
    except Exception as e:
        print(f"Lỗi khi thử nghiệm: {e}")

    # Thử nghiệm với các chuỗi DNA dài hơn
    print("\nThử nghiệm với các chuỗi DNA dài hơn:")
    seq1 = "ATGCTAGCTAGCTAGCTGATCGATCG"
    seq2 = "ATGCTAGCTAGCTAGCTGATCGAAAA"
    seq3 = "TTTTTTTTTTTTTTTTTTTTTTTTTT"

    print(f"Chuỗi 1: {seq1}")
    print(f"Chuỗi 2: {seq2}")
    print(f"Chuỗi 3: {seq3}")

    print("\nĐộ tương đồng giữa các chuỗi:")
    similarity_12 = calculate_similarity(seq1, seq2, model)
    similarity_13 = calculate_similarity(seq1, seq3, model)
    similarity_23 = calculate_similarity(seq2, seq3, model)

    print(f"Độ tương đồng chuỗi 1 và 2: {similarity_12}")
    print(f"Độ tương đồng chuỗi 1 và 3: {similarity_13}")
    print(f"Độ tương đồng chuỗi 2 và 3: {similarity_23}")

    # So sánh các phương pháp tổng hợp vector khác nhau
    print("\nSo sánh các phương pháp tổng hợp vector:")
    for method in ['average', 'sum']:
        similarity = calculate_similarity(seq1, seq2, model, method=method)
        print(f"Phương pháp {method}: {similarity}")


if __name__ == "__main__":
    main()
