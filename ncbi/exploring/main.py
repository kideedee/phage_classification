
import random


def generate_kmer_6(sequence):
    """Convert a sequence into k-mer 6 representation"""
    kmers = []
    for i in range(len(sequence) - 5):
        kmer = sequence[i:i + 6]
        if 'N' not in kmer:  # Skip k-mers with non-ATGC characters
            kmers.append(kmer)
    return kmers


def sliding_window_with_skip(sequences, window_size=100, skip_step=1):
    """
    Generate sliding windows from a list of sequences
    Args:
        sequences: List of DNA sequences
        window_size: Size of the sliding window (default: 100 bp)
        skip_step: Step size for sliding (default: 1)
    Returns:
        List of window sequences
    """
    windows = []
    for seq in sequences:
        if type(seq) is not str:
            continue
        if type(seq) is float:
            continue
        if len(seq) < window_size:
            continue

        # Apply sliding window with specified skip step
        for i in range(0, len(seq) - window_size + 1, skip_step):
            window = seq[i:i + window_size]
            # Check if window contains only ATGC
            if all(base in "ATGC" for base in window):
                windows.append(window)

    return windows


def clean_sequence(sequence):
    """Replace non-ATGC characters with random ATGC"""
    bases = list(sequence)
    for i, base in enumerate(bases):
        if base not in "ATGC":
            bases[i] = random.choice("ATGC")
    return ''.join(bases)


def preprocess_data(lysogenic_seqs, lytic_seqs, window_size=100):
    """
    Preprocess phage sequences to create a balanced training dataset
    Args:
        lysogenic_files: List of lysogenic gene files in FASTA format
        lytic_files: List of lytic phage genome files in FASTA format
        window_size: Size of the sliding window
    Returns:
        Balanced list of lysogenic and lytic sequences
    """

    # Apply sliding window with skip_step=1 for lysogenic sequences
    lysogenic_windows = sliding_window_with_skip(lysogenic_seqs, window_size=window_size, skip_step=1)

    # Apply sliding window with skip_step=91 for lytic sequences
    # Note: We might need to adjust skip_step to balance the dataset
    lytic_windows = sliding_window_with_skip(lytic_seqs, window_size=window_size, skip_step=91)

    # Convert sequences to k-mer 6 representation
    lysogenic_kmers = [generate_kmer_6(window) for window in lysogenic_windows]
    lytic_kmers = [generate_kmer_6(window) for window in lytic_windows]

    print(f"Generated {len(lysogenic_windows)} lysogenic windows and {len(lytic_windows)} lytic windows")

    # Return balanced dataset by sampling if needed
    min_count = min(len(lysogenic_windows), len(lytic_windows))

    if len(lysogenic_windows) > min_count:
        lysogenic_windows = random.sample(lysogenic_windows, min_count)
        lysogenic_kmers = [generate_kmer_6(window) for window in lysogenic_windows]

    if len(lytic_windows) > min_count:
        lytic_windows = random.sample(lytic_windows, min_count)
        lytic_kmers = [generate_kmer_6(window) for window in lytic_windows]

    return {
        'lysogenic_windows': lysogenic_windows,
        'lytic_windows': lytic_windows,
        'lysogenic_kmers': lysogenic_kmers,
        'lytic_kmers': lytic_kmers
    }

import pandas as pd

lysogenic_df = pd.read_csv('lysogenic_train.csv')
lytic_df = pd.read_csv('lytic_train.csv')

prepared_data = preprocess_data(lysogenic_df['sequence_filled'].values, lytic_df['sequence'].values, window_size=100)