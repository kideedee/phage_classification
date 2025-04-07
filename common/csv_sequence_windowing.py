import pandas as pd
import numpy as np
from functools import partial
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='windowing_process.log'
)
logger = logging.getLogger('sequence_windowing')


def generate_windows_for_sequence(sequence, target, seq_id, distribution_type="normal",
                                  range_width=6, min_size=200, max_size=800,
                                  step_size=None, overlap_percent=None):
    """Helper function to generate windows for a single sequence"""

    windowed_data = []

    # Validate sequence is a string
    if not isinstance(sequence, str):
        error_msg = f"Row ID {seq_id}: Expected sequence to be a string, got {type(sequence)} with value {sequence}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Continue with processing if sequence is valid
    mean_size = np.mean([min_size, max_size])
    std_dev = (max_size - min_size) / range_width
    seq_length = len(sequence)

    # Start position for the first window
    start = 0

    while start < seq_length:
        # Generate window size based on selected distribution
        if distribution_type.lower() == "normal":
            window_size = int(np.random.normal(mean_size, std_dev))
        elif distribution_type.lower() == "uniform":
            window_size = int(np.random.uniform(min_size, max_size))
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}. Use 'normal' or 'uniform'.")

        # Ensure window size is within allowed range
        window_size = max(min_size, min(window_size, max_size))

        end = start + window_size
        if end > seq_length:
            end = seq_length

        # Skip windows that are too small
        if end - start < min_size:
            break

        # Extract the window
        window_seq = sequence[start:end]

        # Create a new record with positional information
        windowed_data.append({
            'original_id': seq_id,
            'sequence': window_seq,
            'target': target,  # Keep the original target/label
            'window_start': start + 1,
            'window_end': end,
            'window_size': end - start,
            'distribution': distribution_type
        })

        # Update start position for next window
        if overlap_percent is not None:
            # Calculate step based on overlap percentage
            overlap_amount = int(window_size * (overlap_percent / 100))
            step = window_size - overlap_amount
            start += max(1, step)
        elif step_size is None:
            # Random step size between 1/4 and 3/4 of the window size
            step = int(np.random.uniform(0.25, 0.75) * window_size)
            start += max(1, step)  # Ensure we move at least 1 base
        else:
            start += step_size

    return windowed_data


def window_sequences_with_distribution(df, distribution_type="normal",
                                       range_width=6, min_size=200, max_size=800,
                                       step_size=None, overlap_percent=None):
    """
    Window sequences in a DataFrame with window sizes following a specified distribution.
    Preserves the target/label for each windowed sequence. Uses apply for processing.

    Parameters:
    - df: DataFrame with 'sequence' and 'target' columns
    - distribution_type: Type of distribution ('normal' or 'uniform')
    - range_width: Controls standard deviation for normal distribution
    - min_size: Minimum allowed window size
    - max_size: Maximum allowed window size
    - step_size: Fixed step size between windows (if None, will use random step sizes)
    - overlap_percent: Percentage of overlap between windows (overrides step_size if provided)

    Returns:
    - DataFrame with windowed sequences and their corresponding targets
    """
    # Pre-process DataFrame to validate data types
    df = preprocess_dataframe(df)

    # Create a partial function with the fixed parameters
    window_function = partial(
        generate_windows_for_sequence,
        distribution_type=distribution_type,
        range_width=range_width,
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        overlap_percent=overlap_percent
    )

    all_windows = []

    # Apply the function to each row with error handling
    for idx, row in df.iterrows():
        try:
            windows = window_function(
                sequence=row['sequence'],
                target=row['target'],
                seq_id=idx
            )
            all_windows.extend(windows)
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            continue

    # Convert to DataFrame
    if all_windows:
        windowed_df = pd.DataFrame(all_windows)
        return windowed_df
    else:
        logger.warning("No windows were generated!")
        return pd.DataFrame()


def process_row(row_tuple, distribution_type="normal",
                range_width=6, min_size=200, max_size=800,
                step_size=None, overlap_percent=None):
    """Process a single row for parallel execution"""
    idx, row = row_tuple
    try:
        return generate_windows_for_sequence(
            sequence=row['sequence'],
            target=row['target'],
            seq_id=idx,
            distribution_type=distribution_type,
            range_width=range_width,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            overlap_percent=overlap_percent
        )
    except Exception as e:
        logger.error(f"Error processing row {idx}: {str(e)}")
        logger.error(traceback.format_exc())
        return []  # Return empty list for failed rows


def preprocess_dataframe(df):
    """
    Preprocess the DataFrame to validate data and fix common issues
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Check for missing values
    if df['sequence'].isna().any():
        missing_rows = df[df['sequence'].isna()].index.tolist()
        logger.warning(f"Found {len(missing_rows)} rows with missing sequences at indices: {missing_rows}")

    # Check for non-string types in sequence column
    non_string_rows = df[~df['sequence'].apply(lambda x: isinstance(x, str) if pd.notna(x) else True)].index.tolist()
    if non_string_rows:
        logger.warning(
            f"Found {len(non_string_rows)} rows with non-string sequence types at indices: {non_string_rows}")
        for idx in non_string_rows:
            logger.warning(
                f"Row {idx}: sequence = {df.loc[idx, 'sequence']} (type: {type(df.loc[idx, 'sequence']).__name__})")

    # Convert sequences to strings where possible (handles int/float types)
    for idx in non_string_rows:
        if pd.notna(df.loc[idx, 'sequence']):
            try:
                # Try to convert to string
                df.loc[idx, 'sequence'] = str(df.loc[idx, 'sequence'])
                logger.info(f"Converted sequence at row {idx} to string")
            except:
                # If conversion fails, mark as NaN
                df.loc[idx, 'sequence'] = np.nan
                logger.warning(f"Could not convert sequence at row {idx} to string, setting to NaN")

    # Drop rows with missing sequences after conversion attempts
    original_count = len(df)
    df = df.dropna(subset=['sequence'])
    dropped_count = original_count - len(df)
    if dropped_count > 0:
        logger.warning(f"Dropped {dropped_count} rows with invalid sequences")

    return df


def window_sequences_parallel(df, distribution_type="normal",
                              range_width=6, min_size=200, max_size=800,
                              step_size=None, overlap_percent=None,
                              n_jobs=-1):
    """
    Parallel implementation using joblib for even faster processing.
    Requires: pip install joblib

    Parameters:
    - df: DataFrame with 'sequence' and 'target' columns
    - distribution_type: Type of distribution ('normal' or 'uniform')
    - range_width: Controls standard deviation for normal distribution
    - min_size: Minimum allowed window size
    - max_size: Maximum allowed window size
    - step_size: Fixed step size between windows (if None, will use random step sizes)
    - overlap_percent: Percentage of overlap between windows (overrides step_size if provided)
    - n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
    - DataFrame with windowed sequences and their corresponding targets
    """
    from joblib import Parallel, delayed

    # Preprocess DataFrame
    df = preprocess_dataframe(df)

    if len(df) == 0:
        logger.warning("No valid rows to process after preprocessing")
        return pd.DataFrame()

    logger.info(f"Starting parallel processing of {len(df)} sequences with {n_jobs} jobs")

    # Process in parallel with row index included
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_row)(
            (idx, row),
            distribution_type=distribution_type,
            range_width=range_width,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            overlap_percent=overlap_percent
        ) for idx, row in df.iterrows()
    )

    # Flatten the list of lists, filtering out empty lists from failed rows
    all_windows = [window for windows in results if windows for window in windows]

    logger.info(f"Generated {len(all_windows)} windows from {len(df)} sequences")

    if not all_windows:
        logger.warning("No windows were generated!")
        return pd.DataFrame()

    # Convert to DataFrame
    windowed_df = pd.DataFrame(all_windows)

    return windowed_df

# Example usage:
# df = pd.read_csv('your_sequences.csv')
#
# # Option 1: Using apply (safer, handles errors per-row)
# windowed_df = window_sequences_with_distribution(df, overlap_percent=50)
#
# # Option 2: Using parallel processing (fastest, requires joblib)
# windowed_df = window_sequences_parallel(df, overlap_percent=50)
#
# windowed_df.to_csv('windowed_sequences.csv', index=False)