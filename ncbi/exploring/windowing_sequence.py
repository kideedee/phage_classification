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
    lysogenic_windows = sliding_window_with_skip(lysogenic_seqs, window_size=window_size, skip_step=85)
    lytic_windows = sliding_window_with_skip(lytic_seqs, window_size=window_size, skip_step=256)

    # print(f"Generated {len(lysogenic_windows)} lysogenic windows and {len(lytic_windows)} lytic windows")
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

def run():
    window_size = 512

    lysogenic_train_df = pd.read_csv('lysogenic_train.csv')
    lytic_train_df = pd.read_csv('lytic_train.csv')
    lysogenic_val_df = pd.read_csv('lysogenic_val.csv')
    lytic_val_df = pd.read_csv('lytic_val.csv')
    lysogenic_test_df = pd.read_csv('lysogenic_test.csv')
    lytic_test_df = pd.read_csv('lytic_test.csv')

    print(f"Lysogenic training set shape: {lysogenic_train_df.shape}")
    print(f"Lytic training set shape: {lytic_train_df.shape}")
    print(f"Lysogenic validation set shape: {lysogenic_val_df.shape}")
    print(f"Lytic validation set shape: {lytic_val_df.shape}")
    print(f"Lysogenic test set shape: {lysogenic_test_df.shape}")
    print(f"Lytic test set shape: {lytic_test_df.shape}\n")

    print("Preprocessing data...")

    prepared_train_data = preprocess_data(lysogenic_train_df['sequence'].values,
                                          lytic_train_df['sequence'].values,
                                          window_size=window_size)
    prepared_val_data = preprocess_data(lysogenic_val_df['sequence'].values, lytic_val_df['sequence'].values,
                                        window_size=window_size)
    prepared_test_data = preprocess_data(lysogenic_test_df['sequence'].values, lytic_test_df['sequence'].values,
                                         window_size=window_size)

    print("Data preprocessing completed!\n")
    # Train data
    print("Generating training data...")
    columns = ['sequence', 'label']
    labels = [0] * len(prepared_train_data['lysogenic_windows'])
    lysogenic_df = pd.DataFrame(zip(prepared_train_data['lysogenic_windows'], labels), columns=columns)
    print(f"Generated {len(lysogenic_df)} lysogenic windows in training data")

    labels = [1] * len(prepared_train_data['lytic_windows'])
    lytic_df = pd.DataFrame(zip(prepared_train_data['lytic_windows'], labels), columns=columns)
    print(f"Generated {len(lytic_df)} lytic windows in training data")

    combined_df = pd.concat([lysogenic_df, lytic_df], ignore_index=True)
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Generated {len(shuffled_df)} windows in training data")

    shuffled_df.to_csv("../../data/dnabert_2_preparation/train.csv", index=False)

    # Validation data
    print("Generating validation data...")
    columns = ['sequence', 'label']
    labels = [0] * len(prepared_val_data['lysogenic_windows'])
    lysogenic_val_df = pd.DataFrame(zip(prepared_val_data['lysogenic_windows'], labels), columns=columns)
    print(f"Generated {len(lysogenic_val_df)} lysogenic windows in validation data")

    labels = [1] * len(prepared_val_data['lytic_windows'])
    lytic_val_df = pd.DataFrame(zip(prepared_val_data['lytic_windows'], labels), columns=columns)
    print(f"Generated {len(lytic_val_df)} lytic windows in validation data")

    combined_val_df = pd.concat([lysogenic_val_df, lytic_val_df], ignore_index=True)
    shuffled_val_df = combined_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Generated {len(shuffled_val_df)} windows in validation data")

    shuffled_val_df.to_csv("../../data/dnabert_2_preparation/val.csv", index=False)

    # Test data
    print("Generating test data...")
    columns = ['sequence', 'label']
    labels = [0] * len(prepared_test_data['lysogenic_windows'])
    lysogenic_test_df = pd.DataFrame(zip(prepared_test_data['lysogenic_windows'], labels), columns=columns)
    print(f"Generated {len(lysogenic_test_df)} lysogenic windows in test data")

    labels = [1] * len(prepared_test_data['lytic_windows'])
    lytic_test_df = pd.DataFrame(zip(prepared_test_data['lytic_windows'], labels), columns=columns)
    print(f"Generated {len(lytic_test_df)} lytic windows in test data")

    combined_test_df = pd.concat([lysogenic_test_df, lytic_test_df], ignore_index=True)
    shuffled_test_df = combined_test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Generated {len(shuffled_test_df)} windows in test data")

    shuffled_test_df.to_csv("../../data/dnabert_2_preparation/test.csv", index=False)

if __name__ == '__main__':
    run()