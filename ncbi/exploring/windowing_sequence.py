import random

import pandas as pd


def sliding_window_with_skip(sequences, window_size=100, skip_step=1):
    windows = []
    for seq in sequences:
        if type(seq) is not str:
            continue
        if len(seq) < window_size:
            continue

        for i in range(0, len(seq) - window_size + 1, skip_step):
            window = seq[i:i + window_size]
            if all(base in "ATGC" for base in window):
                windows.append(window)

    return windows


def preprocess_data(lysogenic_seqs, lytic_seqs, window_size=100):
    lysogenic_windows = sliding_window_with_skip(lysogenic_seqs, window_size=window_size, skip_step=1)
    lytic_windows = sliding_window_with_skip(lytic_seqs, window_size=window_size, skip_step=125)

    print(f"Generated {len(lysogenic_windows)} lysogenic windows and {len(lytic_windows)} lytic windows")
    # min_count = min(len(lysogenic_windows), len(lytic_windows))

    # if len(lysogenic_windows) > min_count:
    #     lysogenic_windows = random.sample(lysogenic_windows, min_count)
    #
    # if len(lytic_windows) > min_count:
    #     lytic_windows = random.sample(lytic_windows, min_count)

    return {
        'lysogenic_windows': lysogenic_windows,
        'lytic_windows': lytic_windows
    }


if __name__ == '__main__':
    window_size = 250

    lysogenic_train_df = pd.read_csv('lysogenic_train.csv')
    lytic_train_df = pd.read_csv('lytic_train.csv')
    lysogenic_val_df = pd.read_csv('lysogenic_val.csv')
    lytic_val_df = pd.read_csv('lytic_val.csv')
    lysogenic_test_df = pd.read_csv('lysogenic_test.csv')
    lytic_test_df = pd.read_csv('lytic_test.csv')

    prepared_train_data = preprocess_data(lysogenic_train_df['sequence_filled'].values,
                                          lytic_train_df['sequence'].values,
                                          window_size=window_size)
    prepared_val_data = preprocess_data(lysogenic_val_df['sequence'].values, lytic_val_df['sequence'].values,
                                        window_size=window_size)
    prepared_test_data = preprocess_data(lysogenic_test_df['sequence'].values, lytic_test_df['sequence'].values,
                                         window_size=window_size)

    # Train data
    columns = ['sequence', 'label']
    labels = [0] * len(prepared_train_data['lysogenic_windows'])
    lysogenic_df = pd.DataFrame(zip(prepared_train_data['lysogenic_windows'], labels), columns=columns)
    print(lysogenic_df.size)

    labels = [1] * len(prepared_train_data['lytic_windows'])
    lytic_df = pd.DataFrame(zip(prepared_train_data['lytic_windows'], labels), columns=columns)
    print(lytic_df.size)

    combined_df = pd.concat([lysogenic_df, lytic_df], ignore_index=True)
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(shuffled_df.size)

    shuffled_df.to_csv("../../data/dnabert_2_preparation/train.csv", index=False)

    # Validation data
    columns = ['sequence', 'label']
    labels = [0] * len(prepared_val_data['lysogenic_windows'])
    lysogenic_val_df = pd.DataFrame(zip(prepared_val_data['lysogenic_windows'], labels), columns=columns)
    print(lysogenic_val_df.size)

    labels = [1] * len(prepared_val_data['lytic_windows'])
    lytic_val_df = pd.DataFrame(zip(prepared_val_data['lytic_windows'], labels), columns=columns)
    print(lytic_val_df.size)

    combined_val_df = pd.concat([lysogenic_val_df, lytic_val_df], ignore_index=True)
    shuffled_val_df = combined_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(shuffled_val_df.size)

    shuffled_val_df.to_csv("../../data/dnabert_2_preparation/val.csv", index=False)

    # Test data
    columns = ['sequence', 'label']
    labels = [0] * len(prepared_test_data['lysogenic_windows'])
    lysogenic_test_df = pd.DataFrame(zip(prepared_test_data['lysogenic_windows'], labels), columns=columns)
    print(lysogenic_test_df.size)

    labels = [1] * len(prepared_test_data['lytic_windows'])
    lytic_test_df = pd.DataFrame(zip(prepared_test_data['lytic_windows'], labels), columns=columns)
    print(lytic_test_df.size)

    combined_test_df = pd.concat([lysogenic_test_df, lytic_test_df], ignore_index=True)
    shuffled_test_df = combined_test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(shuffled_test_df.size)

    shuffled_test_df.to_csv("../../data/dnabert_2_preparation/test.csv", index=False)
