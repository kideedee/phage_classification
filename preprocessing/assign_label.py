import os

import h5py
import numpy as np
import pandas as pd


def assign_train_label():
    mode = 'trim'
    data_types = ['train']
    for data_type in data_types:
        data_dir = f"../data/my_data/protbert_embedding/{mode}/{data_type}"
        result_dir = f"../data/my_data/protbert_embedding_with_label/{mode}/{data_type}"

        if data_type == 'train':
            record_num = 40000
        else:
            record_num = 20000

        for folder in os.listdir(data_dir):
            if not os.path.exists(f"{result_dir}/{folder}"):
                os.makedirs(f"{result_dir}/{folder}", exist_ok=True)
            files = os.listdir(f"{data_dir}/{folder}")
            for fold in range(1, 6):
                temp = [file for file in files if file.__contains__(f"temp_{folder}_{fold}.h5")][0]
                viru = [file for file in files if file.__contains__(f"viru_{folder}_{fold}.h5")][0]

                with (h5py.File(os.path.join(data_dir, folder, temp), 'r') as f1,
                      h5py.File(os.path.join(data_dir, folder, viru), 'r') as f2):
                    # print(f1.keys())
                    # Get data from both files using the same key
                    temp_embeddings = f1['embeddings'][:]
                    viru_embeddings = f2['embeddings'][:]

                    temp_labels = np.array([0] * 40000)
                    viru_labels = np.array([1] * 40000)

                    # Concatenate the data
                    combined_embeddings = np.concatenate([temp_embeddings, viru_embeddings], axis=0)
                    combined_labels = np.concatenate([temp_labels, viru_labels], axis=0)

                    # Create new file and save combined data
                    with h5py.File(f'{result_dir}/{folder}/train_{folder}_{fold}.h5', 'w') as f_out:
                        f_out.create_dataset('embeddings', data=combined_embeddings)
                        f_out.create_dataset('labels', data=combined_labels)

                    with h5py.File(f'{result_dir}/{folder}/train_{folder}_{fold}.h5', 'r') as f:
                        data = f['embeddings'][:]
                        print(f"Combined shape: {data.shape}")


def assign_test_label():
    mode = 'trim'
    data_types = ['test']
    for data_type in data_types:
        data_dir = f"../data/my_data/protbert_embedding/{mode}/{data_type}"
        result_dir = f"../data/my_data/protbert_embedding_with_label/{mode}/{data_type}"

        for folder in os.listdir(data_dir):
            if not os.path.exists(f"{result_dir}/{folder}"):
                os.makedirs(f"{result_dir}/{folder}", exist_ok=True)

            for file in os.listdir(f"{data_dir}/{folder}"):
                print(file)
                label_file = file.replace('.h5', '_label.csv')
                test_labels = pd.read_csv(f"../data/my_data/original_data/{data_type}/label/{label_file}", header=None)
                labels = test_labels.iloc[:, 0].to_numpy()
                print(labels.shape)
                with h5py.File(os.path.join(data_dir, folder, file), 'r') as f:
                    print(list(f.keys()))
                    print(f['embeddings'].shape)

                    if f.keys().__contains__('labels'):
                        print("Dataset already contains labels")
                        continue
                    with h5py.File(os.path.join(result_dir, folder, file), 'w') as f_out:
                        f_out.create_dataset('embeddings', data=f['embeddings'][:])
                        f_out.create_dataset('labels', data=labels)

                    with h5py.File(os.path.join(result_dir, folder, file), 'r') as f:
                        print(f['embeddings'].shape)
                        print(f['labels'].shape)



if __name__ == '__main__':
    # assign_train_label()
    assign_test_label()
