"""
Chương trình sử dụng mô hình dna2vec đã được huấn luyện sẵn để embedding chuỗi DNA
Phiên bản có hỗ trợ GPU
"""
import os
import sys

import numpy as np
import torch
from gensim.models import KeyedVectors

from common.env_config import config


class SingleKModel:
    """
    Mô hình cho một độ dài k-mer cụ thể, có hỗ trợ GPU.
    """

    def __init__(self, model, device):
        self.model = model
        self.vocab_lst = sorted(model.key_to_index.keys())
        self.device = device

        # Tạo cache để lưu các vector đã chuyển sang GPU
        self.vector_cache = {}


class MultiKModel:
    """
    Mô hình tổng hợp cho nhiều độ dài k-mer khác nhau, có hỗ trợ GPU.
    """

    def __init__(self, filepath, device=None):
        self.filepath = filepath
        self.logger = None  # Sẽ được khởi tạo nếu cần logging

        # Thiết lập device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")

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

        # Cache cho vectors trên GPU
        self.vector_cache = {}

    def _make_dict_model(self, k):
        """
        Tạo mô hình từ điển cho một độ dài k cụ thể.
        """
        vectors = np.array([self.k_models[k][vocab] for vocab in sorted(self.k_models[k].keys())])

        # Tạo một KeyedVectors model mới
        model = KeyedVectors(self.aggregate.vector_size)
        model.add_vectors(sorted(self.k_models[k].keys()), vectors)

        return SingleKModel(model, self.device)

    def model(self, k):
        """
        Trả về mô hình cho độ dài k.
        """
        return self.models[k]

    def vector(self, km):
        """
        Trả về vector của k-mer, đã chuyển sang GPU.
        """
        # Kiểm tra xem vector đã cache chưa
        if km not in self.vector_cache:
            # Lấy vector từ mô hình gensim (NumPy array)
            numpy_vector = self.aggregate[km]
            # Chuyển đổi sang PyTorch tensor và đưa lên GPU
            self.vector_cache[km] = torch.tensor(numpy_vector, dtype=torch.float32).to(self.device)

        return self.vector_cache[km]

    def cosine_distance(self, km1, km2):
        """
        Tính khoảng cách cosine giữa hai k-mer trên GPU.
        """
        # Lấy vectors từ GPU cache
        vec1 = self.vector(km1)
        vec2 = self.vector(km2)

        # Tính cosine similarity trên GPU
        similarity = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        return similarity


def embed_dna_sequence(sequence, model, k_min=3, k_max=8, method='average'):
    """
    Embedding một chuỗi DNA bằng cách chia thành các k-mer và tổng hợp các vector trên GPU.

    Args:
        sequence: Chuỗi DNA cần embedding
        model: Mô hình MultiKModel đã được tải
        k_min: Độ dài k-mer tối thiểu
        k_max: Độ dài k-mer tối đa
        method: Phương pháp tổng hợp ('average', 'sum', 'concat')

    Returns:
        Vector biểu diễn của chuỗi DNA trên GPU
    """
    # Loại bỏ các ký tự không hợp lệ
    sequence = ''.join([c for c in sequence.upper() if c in 'ACGT'])

    # Tạo các k-mer từ chuỗi DNA
    kmers = []
    for k in range(k_min, k_max + 1):
        if k > len(sequence):
            continue
        kmers.extend([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

    # Lấy vector của từng k-mer và đưa lên GPU
    kmer_vectors = []
    for kmer in kmers:
        try:
            # vector đã được chuyển sang GPU trong hàm model.vector()
            vec = model.vector(kmer)
            kmer_vectors.append(vec)
        except KeyError:
            # Bỏ qua nếu k-mer không có trong mô hình
            continue

    # Tổng hợp các vector
    if not kmer_vectors:
        return None

    # Xếp tất cả các vector thành một tensor
    vectors_tensor = torch.stack(kmer_vectors)

    if method == 'average':
        return torch.mean(vectors_tensor, dim=0)
    elif method == 'sum':
        return torch.sum(vectors_tensor, dim=0)
    elif method == 'concat':
        # Tạo một vector lớn bằng cách nối tất cả các vector lại
        # Giới hạn số lượng k-mer để tránh vector quá lớn
        max_kmers = 100
        selected_kmers = vectors_tensor[:max_kmers]
        return torch.flatten(selected_kmers)
    else:
        raise ValueError(f"Phương pháp {method} không được hỗ trợ")


def calculate_similarity(seq1, seq2, model, method='average'):
    """
    Tính độ tương đồng giữa hai chuỗi DNA trên GPU.

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

    # Tính độ tương đồng cosine trên GPU
    similarity = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    return similarity


def batch_similarity(sequences, reference_seq, model, method='average', batch_size=100):
    """
    Tính độ tương đồng giữa nhiều chuỗi DNA với một chuỗi tham chiếu, tận dụng GPU để xử lý hàng loạt.

    Args:
        sequences: Danh sách các chuỗi DNA cần so sánh
        reference_seq: Chuỗi DNA tham chiếu
        model: Mô hình MultiKModel
        method: Phương pháp tổng hợp vector
        batch_size: Kích thước batch để xử lý cùng lúc

    Returns:
        Danh sách các giá trị độ tương đồng
    """
    # Tính vector embedding cho chuỗi tham chiếu
    ref_vec = embed_dna_sequence(reference_seq, model, method=method)
    if ref_vec is None:
        return [None] * len(sequences)

    # Chuẩn bị ref_vec cho so sánh batch
    ref_vec = ref_vec.unsqueeze(0)  # [1, dim]

    results = []

    # Xử lý theo batch
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_vecs = []

        # Tính vectors cho tất cả các chuỗi trong batch
        for seq in batch:
            vec = embed_dna_sequence(seq, model, method=method)
            if vec is not None:
                batch_vecs.append(vec)
            else:
                batch_vecs.append(torch.zeros_like(ref_vec[0]))  # Vector 0 cho các chuỗi không hợp lệ

        # Stack tất cả vectors trong batch
        if batch_vecs:
            batch_tensor = torch.stack(batch_vecs)  # [batch_size, dim]

            # Tính cosine similarity cho cả batch
            batch_similarities = torch.nn.functional.cosine_similarity(batch_tensor, ref_vec)

            # Chuyển kết quả về CPU và thêm vào danh sách kết quả
            results.extend(batch_similarities.cpu().numpy().tolist())
        else:
            # Nếu không có vector hợp lệ trong batch
            results.extend([None] * len(batch))

    return results


def main():
    # Đường dẫn đến file mô hình
    model_path = os.path.join(config.MODEL_DIR, "dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v")

    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file mô hình tại {model_path}")
        print("Vui lòng tải mô hình từ trang GitHub của tác giả: https://github.com/pnpnpn/dna2vec")
        sys.exit(1)

    # Kiểm tra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
        print(f"Bộ nhớ GPU khả dụng: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Không tìm thấy GPU, sử dụng CPU")

    # Tải mô hình
    print("Đang tải mô hình dna2vec...")
    model = MultiKModel(model_path, device=device)
    print("Đã tải mô hình thành công!")

    # Thử nghiệm với một số k-mer đơn giản
    print("\nThử nghiệm với các k-mer đơn giản:")
    try:
        vec = model.vector('AAA')
        print(f"Vector của AAA: {vec.cpu().numpy()[:5]}...")  # Chỉ hiển thị 5 giá trị đầu tiên
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

    # Thử nghiệm tính toán batch với GPU
    print("\nThử nghiệm tính toán hàng loạt với GPU:")
    test_sequences = [seq1, seq2, seq3] * 100  # Tạo một danh sách lớn các chuỗi để thử nghiệm

    # Đo thời gian thực hiện
    import time
    start_time = time.time()

    batch_results = batch_similarity(test_sequences, seq1, model, batch_size=50)

    elapsed_time = time.time() - start_time
    print(f"Thời gian hoàn thành cho {len(test_sequences)} chuỗi: {elapsed_time:.4f} giây")
    print(f"Kết quả đầu tiên: {batch_results[:3]}")


if __name__ == "__main__":
    main()