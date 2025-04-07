import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from Bio import Entrez
from tqdm import tqdm

from common.env_config import config
from logger.phg_cls_log import log

Entrez.email = "your_email@example.com"  # Replace with your email
Entrez.tool = "ncbi_batch_downloader"

# Output directory
base_dir = config.GEN_BANK_DIR

# Create cache directory for tracking completed downloads
cache_dir = os.path.join(base_dir, ".cache")
os.makedirs(cache_dir, exist_ok=True)


def download_genbank(accession, output_dir, output_file=None, retries=3, retry_delay=5):
    """
    Download GenBank record from NCBI and save to file with retry mechanism

    Args:
        accession: Accession number of the sequence
        output_dir: Output directory
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
        if os.path.exists(final_output):
            log.info(f"Skipping {accession} - already downloaded")
            return final_output

    for attempt in range(retries):
        try:
            # Handle connection errors with retries
            try:
                handle = Entrez.esearch(db="nucleotide", term=accession)
                record = Entrez.read(handle)
                handle.close()
            except Exception as e:
                log.warning(f"Search error for {accession} (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(retry_delay)
                continue

            # Check if record was found
            if not record["IdList"]:
                log.warning(f"No record found for {accession}")
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
                log.warning(f"Fetch error for {accession} (attempt {attempt + 1}/{retries}): {e}")
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

            log.info(f"Successfully downloaded {accession} to {final_output}")
            return final_output

        except Exception as e:
            log.error(f"Error downloading {accession} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

    log.error(f"Failed to download {accession} after {retries} attempts")
    return None


def download_worker(args):
    """Worker function for parallel downloads"""
    accession, output_dir, output_file = args
    return download_genbank(accession, output_dir, output_file)


def download_batch(file_path, is_train=True, max_workers=5, batch_size=200):
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
        df = pd.read_csv(file_path)

        # if 'Accession number' not in df.columns:
        #     log.error("Error: Excel file must have 'Accession number' column")
        #     return

        if 'label' not in df.columns:
            log.error("Error: Excel file must have 'label' column")
            return

        download_tasks = []
        for index, row in df.iterrows():
            accession = str(row['accession_number']).strip()
            life_cycle = str(row["label"]).strip()

            output_dir = base_dir
            output_file = f"{accession}_{life_cycle}"

            download_tasks.append((accession, output_dir, output_file))

        # Process in batches with progress bar
        total_tasks = len(download_tasks)
        log.info(f"Starting download of {total_tasks} GenBank records from {file_path}")

        for i in range(0, total_tasks, batch_size):
            batch = download_tasks[i:i + batch_size]
            log.info(f"Processing batch {i // batch_size + 1}/{(total_tasks + batch_size - 1) // batch_size}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(download_worker, batch), total=len(batch), desc="Downloading"))

            # if i + batch_size < total_tasks:
            #     log.info(f"Pausing between batches for 10 seconds...")
            #     time.sleep(10)

        log.info(f"Completed downloading GenBank records from {file_path}")

    except Exception as e:
        log.error(f"Error processing Excel file: {e}")
        raise


def resume_download(file_path, is_train=True):
    """Resume a previously interrupted download"""
    log.info(f"Resuming download from {file_path}")
    download_batch(file_path, is_train)


if __name__ == "__main__":
    # Example usage:

    # 1. Download batch data from Excel
    download_batch("../../data/custom/combined_ds.csv", is_train=True, max_workers=3,
                   batch_size=100)
    # download_batch_from_excel("../../data/deep_pl_data/test_dataset.xlsx", is_train=False, max_workers=3,
    #                           batch_size=100)

    # 2. Resume a previously interrupted download
    # resume_download("train_dataset.xlsx", is_train=True)

    # 3. Download a single accession
    # download_genbank("NC_003313", "single_sample")
