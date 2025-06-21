import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from Bio import Entrez
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ncbi_download.log"),
        logging.StreamHandler()
    ]
)

Entrez.email = "your_email@example.com"  # Replace with your email
Entrez.tool = "ncbi_batch_downloader"

# Output directory
base_dir = "../../data/ncbi_data/gen_bank"
os.makedirs(base_dir, exist_ok=True)

# Create cache directory for tracking completed downloads
cache_dir = os.path.join(base_dir, ".cache")
os.makedirs(cache_dir, exist_ok=True)


def download_genbank(accession, output_dir, output_file=None, retries=3, retry_delay=5):
    """
    Download GenBank record from NCBI and save to file with retry mechanism

    Args:
        accession: Accession number of the sequence
        output_file: Output filename (optional)
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Path to the saved file or None if failed
    """
    # Skip if already downloaded
    if output_file:
        cache_file = os.path.join(cache_dir, f"{output_file}.done")
        final_output = os.path.join(output_dir, f"{output_file}.gb")
        if os.path.exists(cache_file) and os.path.exists(final_output):
            logging.info(f"Skipping {accession} - already downloaded")
            return final_output

    for attempt in range(retries):
        try:
            # Handle connection errors with retries
            try:
                handle = Entrez.esearch(db="nucleotide", term=accession)
                record = Entrez.read(handle)
                handle.close()
            except Exception as e:
                logging.warning(f"Search error for {accession} (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(retry_delay)
                continue

            # Check if record was found
            if not record["IdList"]:
                logging.warning(f"No record found for {accession}")
                return None

            id_list = record["IdList"]

            # Fetch the GenBank record with retry
            try:
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=id_list[0],
                    rettype="gb",
                    retmode="text"
                )
                content = handle.read()
                handle.close()
            except Exception as e:
                logging.warning(f"Fetch error for {accession} (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(retry_delay)
                continue

            # Set output filename
            if output_file is None:
                final_output = os.path.join(output_dir, f"{accession}.gb")
            else:
                final_output = os.path.join(output_dir, f"{output_file}.gb")

            # Write to file
            with open(final_output, "w") as out_handle:
                out_handle.write(content)

            # Mark as complete
            if output_file:
                with open(os.path.join(cache_dir, f"{output_file}.done"), "w") as f:
                    f.write("done")

            logging.info(f"Successfully downloaded {accession} to {final_output}")
            return final_output

        except Exception as e:
            logging.error(f"Error downloading {accession} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

    logging.error(f"Failed to download {accession} after {retries} attempts")
    return None


def download_worker(args):
    """Worker function for parallel downloads"""
    accession, output_dir, output_file = args
    return download_genbank(accession, output_dir, output_file)


def download_batch_from_excel(file_path, is_train=True, max_workers=5, batch_size=200):
    """
    Download GenBank data for all accessions in Excel file with parallel processing
    and batch processing to avoid overwhelming NCBI servers

    Args:
        file_path: Path to Excel file
        is_train: Whether this is training data
        max_workers: Number of parallel download threads
        batch_size: Size of batches to process with pauses between them
    """
    try:
        excel_df = pd.read_excel(file_path)

        if 'Accession number' not in excel_df.columns:
            logging.error("Error: Excel file must have 'Accession number' column")
            return

        if 'Lifecycle' not in excel_df.columns:
            logging.error("Error: Excel file must have 'Lifecycle' column")
            return

        download_tasks = []
        for index, row in excel_df.iterrows():
            accession = str(row['Accession number']).strip()
            life_cycle = str(row["Lifecycle"]).strip()

            if is_train:
                if "5-fold cross-validation" not in excel_df.columns:
                    logging.error("Error: Training data Excel file must have '5-fold cross-validation' column")
                    return
                group = str(row["5-fold cross-validation"]).strip()
                output_dir = os.path.join(base_dir, f"train/{life_cycle}/{group}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                output_file = f"train_{accession}_{life_cycle}_{group}"
            else:
                output_dir = os.path.join(base_dir, f"test/{life_cycle}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                output_file = f"test_{accession}_{life_cycle}"

            download_tasks.append((accession, output_dir, output_file))

        # Process in batches with progress bar
        total_tasks = len(download_tasks)
        logging.info(f"Starting download of {total_tasks} GenBank records from {file_path}")

        for i in range(0, total_tasks, batch_size):
            batch = download_tasks[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}/{(total_tasks + batch_size - 1) // batch_size}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(download_worker, batch), total=len(batch), desc="Downloading"))

            if i + batch_size < total_tasks:
                logging.info(f"Pausing between batches for 10 seconds...")
                time.sleep(10)

        logging.info(f"Completed downloading GenBank records from {file_path}")

    except Exception as e:
        logging.error(f"Error processing Excel file: {e}")
        raise


def resume_download(file_path, is_train=True):
    """Resume a previously interrupted download"""
    logging.info(f"Resuming download from {file_path}")
    download_batch_from_excel(file_path, is_train)


if __name__ == "__main__":
    # Example usage:

    # 1. Download batch data from Excel
    download_batch_from_excel("../../data/deep_pl_data/train_dataset.xlsx", is_train=True, max_workers=3, batch_size=100)
    download_batch_from_excel("../../data/deep_pl_data/test_dataset.xlsx", is_train=False, max_workers=3, batch_size=100)

    # 2. Resume a previously interrupted download
    # resume_download("train_dataset.xlsx", is_train=True)

    # 3. Download a single accession
    # download_genbank("NC_003313", "single_sample")
