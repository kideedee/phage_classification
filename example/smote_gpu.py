import gc

import numpy as np
import torch
from cuml import LinearSVC
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import CondensedNearestNeighbour, EditedNearestNeighbours
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle
from cuml.neighbors import NearestNeighbors


def balance_by_ratio(X, y, target_ratio=3.0):
    """
    Cân bằng dữ liệu dựa trên tỷ lệ mong muốn giữa lớp đa số và lớp thiểu số.

    Tham số:
    X -- dữ liệu đặc trưng
    y -- nhãn
    target_ratio -- tỷ lệ mong muốn giữa lớp đa số và lớp thiểu số (mặc định: 3.0)

    Trả về:
    X_balanced -- dữ liệu đặc trưng sau khi cân bằng
    y_balanced -- nhãn sau khi cân bằng
    """
    # Đếm số lượng mỗi lớp
    unique_labels, counts = np.unique(y, return_counts=True)

    # Xác định lớp đa số và lớp thiểu số
    majority_class = unique_labels[np.argmax(counts)]
    minority_class = unique_labels[np.argmin(counts)]

    majority_count = max(counts)
    minority_count = min(counts)

    # Tìm các chỉ số của hai lớp
    majority_indices = np.where(y == majority_class)[0]
    minority_indices = np.where(y == minority_class)[0]

    # Tính toán số lượng mẫu cần giữ lại cho lớp đa số
    target_majority_count = int(minority_count * target_ratio)

    # Nếu lớp đa số đã ít hơn hoặc bằng tỷ lệ mong muốn, không cần giảm
    if majority_count <= target_majority_count:
        print(
            f"Lớp đa số ({majority_count} mẫu) đã có tỷ lệ ít hơn hoặc bằng {target_ratio} lần lớp thiểu số ({minority_count} mẫu)")
        return X, y

    # Xáo trộn các chỉ số của lớp đa số và chọn số lượng cần thiết
    np.random.shuffle(majority_indices)
    selected_majority_indices = majority_indices[:target_majority_count]

    # Kết hợp các chỉ số đã chọn từ lớp đa số và tất cả chỉ số từ lớp thiểu số
    balanced_indices = np.concatenate([selected_majority_indices, minority_indices])

    # Xáo trộn lại dữ liệu
    balanced_indices = shuffle(balanced_indices)

    # Tạo tập dữ liệu mới
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    return X_balanced, y_balanced


# Hàm để in ra phân phối lớp
def print_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    distribution = {}
    for label, count in zip(unique, counts):
        distribution[label] = count
        print(f"Nhãn {label}: {count} mẫu")

    # Tính tỷ lệ giữa các lớp
    if len(unique) == 2:
        ratio = max(counts) / min(counts)
        print(f"Tỷ lệ giữa lớp đa số và lớp thiểu số: {ratio:.2f}")

    return distribution


if __name__ == '__main__':
    x_train = np.load("../experiment/custom/re_sampling/word2vec_train_vector.npy")
    y_train = np.load("../experiment/custom/re_sampling/y_train.npy")

    # # Ví dụ sử dụng:
    # print("Phân phối lớp ban đầu:")
    # initial_distribution = print_class_distribution(y_train)
    #
    # # Cân bằng dữ liệu với tỷ lệ mong muốn là 3
    # target_ratio = 1.5
    # X_balanced, y_balanced = balance_by_ratio(x_train, y_train, target_ratio)
    #
    # print(f"\nPhân phối lớp sau khi cân bằng (tỷ lệ mục tiêu: {target_ratio}):")
    # final_distribution = print_class_distribution(y_balanced)
    #
    # X_balanced, y_balanced = balance_by_ratio(x_train, y_train)

    gc.collect()
    torch.cuda.empty_cache()
    #
    nn1 = NearestNeighbors(n_neighbors=6)
    nn2 = NearestNeighbors(n_neighbors=4)
    # X_resampled, y_resampled = ADASYN(n_neighbors=nn).fit_resample(X_balanced, y_balanced)
    smote = SMOTE(k_neighbors=nn1)
    enn = EditedNearestNeighbours(n_neighbors=nn2)
    cnn = SMOTEENN(enn=enn, smote=smote)
    # cnn.estimator = nn  # extra step for this resampler
    X_resampled, y_resampled = cnn.fit_resample(x_train, y_train)
    np.save(f"X_resampled_enn_majority.npy", X_resampled)
    np.save(f"y_resampled_enn_majority.npy", y_resampled)
    print("\n")
    print_class_distribution(y_resampled)


    # # Thiết lập SVM làm estimator cho RFECV
    # svm = LinearSVC(class_weight="balanced")
    #
    # # Thiết lập RFECV
    # rfecv = RFECV(
    #     estimator=svm,
    #     step=1,
    #     cv=10,
    #     scoring="roc_auc",
    #     min_features_to_select=10,
    #     n_jobs=-1
    # )
    #
    # # Thực hiện feature selection
    # rfecv.fit(X_balanced, y_balanced)
    #
    # # Lấy các đặc trưng được chọn
    # X_train_selected = rfecv.transform(X_balanced)
    # # X_test_selected = rfecv.transform(X_test_vectors)
    #
    # print(f"Số lượng đặc trưng tối ưu: {rfecv.n_features_}")
    # print(f"Kích thước dữ liệu sau khi chọn đặc trưng: {X_train_selected.shape}")
    #
    # # Vẽ đồ thị số lượng đặc trưng vs. điểm số cross-validation
    # plt.figure(figsize=(10, 6))
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (ROC AUC)")
    # plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
    # plt.savefig("optimal_features.png")
    # plt.show()
    # plt.close()
