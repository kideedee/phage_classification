import os.path

import h5py

from common.env_config import config

# Mở file H5
with h5py.File(os.path.join(config.DATA_DIR, "deephage_data/original_data_with_label/train/100_400/train_100_400_1.h5"), 'r') as f:
    # Liệt kê tất cả các nhóm trong file
    print("Các nhóm trong file:", list(f.keys()))

    # Truy cập vào một dataset cụ thể (giả sử có dataset tên 'data')
    if 'labels' in f:
        data = f['labels'][()]
        print("Kích thước data:", data.shape)


    # Duyệt qua tất cả các nhóm và dataset
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")


    # In ra cấu trúc của file
    f.visititems(print_attrs)

    sequences = f['sequences']
    print(sequences)