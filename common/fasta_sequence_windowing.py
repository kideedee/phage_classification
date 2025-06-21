import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def window_fasta(input_file, window_size, step_size=None):
    # If step_size is not provided, use non-overlapping windows
    if step_size is None:
        step_size = window_size

    windowed_records = []

    # Parse the original FASTA file
    for record in SeqIO.parse(input_file, "fasta"):
        seq_id = record.id
        description = record.description
        sequence = str(record.seq)
        seq_length = len(sequence)

        # Create windows
        for start in range(0, seq_length - window_size + 1, step_size):
            end = start + window_size
            if end > seq_length:
                break

            # Extract the window
            window_seq = sequence[start:end]

            # Create a new record with positional information
            new_id = f"{seq_id}_{start + 1}_{end}"
            new_description = f"{description} | window: {start + 1}-{end}"

            # Create new SeqRecord
            new_record = SeqRecord(
                seq=record.seq[start:end],
                id=new_id,
                description=new_description
            )

            windowed_records.append(new_record)

    return windowed_records


def window_fasta_with_distribution(input_file, output_file=None, distribution_type="normal",
                                   range_width=6, min_size=200, max_size=800,
                                   step_size=None, overlap_percent=None):
    """
    Window a FASTA file with window sizes following a specified distribution.

    Parameters:
    - input_file: Path to input FASTA file
    - output_file: Path to output FASTA file
    - distribution_type: Type of distribution ('normal' or 'uniform')
    - std_dev: Standard deviation (for normal distribution)
    - min_size: Minimum allowed window size
    - max_size: Maximum allowed window size
    - step_size: Fixed step size between windows (if None, will use random step sizes)
    - overlap_percent: Percentage of overlap between windows (overrides step_size if provided)

    Returns:
    - List of SeqRecord objects with windowed sequences
    """
    windowed_records = []
    mean_size = np.mean([min_size, max_size])
    std_dev = (max_size - min_size) / range_width

    # Parse the original FASTA file
    for record in SeqIO.parse(input_file, "fasta"):
        seq_id = record.id
        description = record.description
        sequence = str(record.seq)
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
            new_id = f"{seq_id}_{start + 1}_{end}"
            new_description = f"{description} | window: {start + 1}-{end} | size: {end - start} | dist: {distribution_type}"

            # Create new SeqRecord
            new_record = SeqRecord(
                seq=record.seq[start:end],
                id=new_id,
                description=new_description
            )

            windowed_records.append(new_record)

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

    # Write the records to the output file
    if output_file is not None:
        SeqIO.write(windowed_records, output_file, "fasta")

    return windowed_records


def analyze_window_sizes(windowed_records, plot_filename=None):
    """
    Analyze the size distribution of windowed sequences

    Parameters:
    - windowed_records: List of SeqRecord objects
    - plot_filename: If provided, save a histogram to this file
    """
    sizes = [len(record.seq) for record in windowed_records]
    mean_size = np.mean(sizes)
    std_size = np.std(sizes)
    min_size = np.min(sizes)
    max_size = np.max(sizes)

    print(f"Thống kê kích thước window:")
    print(f"- Trung bình: {mean_size:.2f}")
    print(f"- Độ lệch chuẩn: {std_size:.2f}")
    print(f"- Tối thiểu: {min_size}")
    print(f"- Tối đa: {max_size}")
    print(f"- Số lượng window: {len(sizes)}")

    if plot_filename:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(sizes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Phân phối kích thước window')
            plt.xlabel('Kích thước (bp)')
            plt.ylabel('Số lượng window')
            plt.grid(axis='y', alpha=0.75)
            plt.savefig(plot_filename)
            print(f"Đã lưu biểu đồ phân phối tại: {plot_filename}")
        except ImportError:
            print("Không thể tạo biểu đồ. Vui lòng cài đặt matplotlib.")
