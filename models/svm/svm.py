import os
import time
from turtledemo.penrose import start

import h5py
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

if __name__ == '__main__':
    max_length = [100, 400, 800, 1200, 1800]

    train_matrix = None
    test_matrix = None
    train_label = None
    test_label = None

    start_main = time.time()
    for i in range(len(max_length)):
        if i == 0:
            continue
        length = max_length[i]
        print(f"Max length: {length}")
        group = f"{max_length[i - 1]}_{length}"

        start_group = time.time()
        for r in range(5):
            start_fold = time.time()

            j = r + 1
            root_data_dir = f"../../data"
            root_data_dir = os.path.join(root_data_dir, group)
            train_dir = os.path.join(root_data_dir, "train")
            test_dir = os.path.join(root_data_dir, "test")
            print("root_data_dir: ", root_data_dir)
            # Load dee_phage_data
            train_sequence_file = [f for f in os.listdir(os.path.join(train_dir, "sequences")) if f'_{j}.mat' in f][0]
            train_label_file = [f for f in os.listdir(os.path.join(train_dir, "labels")) if f'_{j}.mat' in f][0]
            test_sequence_file = [f for f in os.listdir(os.path.join(test_dir, "sequences")) if f'_{j}.mat' in f][0]
            test_label_file = [f for f in os.listdir(os.path.join(test_dir, "labels")) if f'_{j}.mat' in f][0]

            train_matrix = h5py.File(os.path.join(train_dir, f'sequences/{train_sequence_file}'), 'r')['P_train_ds'][:]
            train_label = h5py.File(os.path.join(train_dir, f'labels/{train_label_file}'), 'r')['T_train_ds'][:]
            test_matrix = h5py.File(os.path.join(test_dir, f'sequences/{test_sequence_file}'), 'r')['P_test'][:]
            test_label = h5py.File(os.path.join(test_dir, f'labels/{test_label_file}'), 'r')['T_test'][:]

            train_matrix = train_matrix.transpose()
            train_label = train_label.transpose()
            test_matrix = test_matrix.transpose()
            test_label = test_label.transpose()

            train_matrix = train_matrix.reshape(-1, length, 4)
            test_matrix = test_matrix.reshape(-1, length, 4)

            train_matrix_flat = train_matrix.reshape(train_matrix.shape[0], -1)
            test_matrix_flat = test_matrix.reshape(test_matrix.shape[0], -1)

            clf = svm.SVC()
            clf.fit(train_matrix_flat, train_label.ravel())
            test_predictions = clf.predict(test_matrix_flat)
            print(classification_report(test_label, test_predictions))

            end_fold = time.time() - start_fold
            print(f"Fold {j} runtime: {end_fold}")


        group_runtime = time.time() - start_group
        print(f"Group {length} runtime: {group_runtime}")

    main_runtime = time.time() - start_main
    print(f"Main runtime: {main_runtime}")
