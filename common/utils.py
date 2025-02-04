import os

import h5py


def log_and_append_file(file_path, message):
    print(message)
    with open(file_path, 'a') as f:
        f.write(message)
        f.write('\n')
        f.close()