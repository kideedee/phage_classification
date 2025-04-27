import numpy as np
from Bio.Seq import Seq
from gensim.models import Word2Vec


def generate_kmers(sequence, k=6):
    """Tạo k-mers từ một chuỗi DNA bằng cách dùng sliding window."""
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def prepare_sequences_for_word2vec(sequences, k=6):
    """Chuẩn bị chuỗi cho Word2Vec bằng cách chuyển đổi thành các kmer."""
    corpus = []
    for seq in sequences:
        # Tạo k-mers và chỉ giữ những k-mer hợp lệ (chỉ chứa A, C, G, T)
        valid_kmers = [kmer for kmer in generate_kmers(seq, k)
                       if all(nucleotide in "ACGT" for nucleotide in kmer)]
        corpus.append(valid_kmers)
    return corpus


def reverse_complement_augmentation(sequences, labels):
    """Tạo dữ liệu bổ sung bằng cách đảo ngược bổ sung chuỗi DNA."""
    augmented_sequences = []
    augmented_labels = []

    for seq, label in zip(sequences, labels):
        # Thêm chuỗi gốc
        augmented_sequences.append(seq)
        augmented_labels.append(label)

        # Thêm chuỗi đảo ngược bổ sung
        reverse_comp = str(Seq(seq).reverse_complement())
        augmented_sequences.append(reverse_comp)
        augmented_labels.append(label)

    return np.array(augmented_sequences), np.array(augmented_labels)


# Áp dụng reverse complement augmentation
X_train_aug, y_train_aug = reverse_complement_augmentation(X_train, y_train)

print(f"Số phage trong tập huấn luyện sau augmentation: {len(X_train_aug)}")

# Chuẩn bị dữ liệu cho Word2Vec
corpus = prepare_sequences_for_word2vec(X_train_aug, k=6)

# Huấn luyện mô hình Word2Vec với Skip-gram
word2vec_model = Word2Vec(
    sentences=corpus,
    vector_size=300,
    window=5,
    min_count=1,
    sample=1e-3,
    sg=1,  # Skip-gram model
    hs=0,  # Dùng negative sampling thay vì hierarchical softmax
    epochs=20,
    negative=5,
    workers=4,
    seed=42
)

# Lưu mô hình
word2vec_model.save("phage_word2vec_model.bin")


# Tạo vector cho mỗi chuỗi bằng cách tính trung bình các vector từ các k-mer
def sequence_to_vector(sequence, word2vec_model, k=6):
    """Chuyển đổi một chuỗi DNA thành vector đặc trưng sử dụng Word2Vec."""
    kmers = generate_kmers(sequence, k)
    valid_kmers = [kmer for kmer in kmers if kmer in word2vec_model.wv.key_to_index]

    if not valid_kmers:
        return np.zeros(word2vec_model.vector_size)

    # Tính trung bình các vector
    vectors = [word2vec_model.wv[kmer] for kmer in valid_kmers]
    return np.mean(vectors, axis=0)


# Chuyển đổi chuỗi thành vector đặc trưng
X_train_vectors = np.array([sequence_to_vector(seq, word2vec_model) for seq in X_train_aug])
X_test_vectors = np.array([sequence_to_vector(seq, word2vec_model) for seq in X_test])

print(f"Kích thước vector đặc trưng: {X_train_vectors.shape}")
