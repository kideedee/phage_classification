import numpy as np
from cuml import SVC
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

if __name__ == '__main__':

    X_train_vectors = np.load("../experiment/custom/re_sampling/word2vec_train_vector.npy")
    y_train = np.load("../experiment/custom/re_sampling/y_train.npy")

    # Thiết lập SVM làm estimator cho RFECV
    svm = SVC(kernel="linear", class_weight="balanced")

    # Thiết lập RFECV
    rfecv = RFECV(
        estimator=svm,
        step=1,
        cv=10,
        scoring="roc_auc",
        min_features_to_select=10,
        n_jobs=-1
    )

    # Thực hiện feature selection
    rfecv.fit(X_train_vectors, y_train)

    # Lấy các đặc trưng được chọn
    X_train_selected = rfecv.transform(X_train_vectors)
    # X_test_selected = rfecv.transform(X_test_vectors)

    print(f"Số lượng đặc trưng tối ưu: {rfecv.n_features_}")
    print(f"Kích thước dữ liệu sau khi chọn đặc trưng: {X_train_selected.shape}")

    # Vẽ đồ thị số lượng đặc trưng vs. điểm số cross-validation
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (ROC AUC)")
    plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
    plt.savefig("optimal_features.png")
    plt.show()
    plt.close()