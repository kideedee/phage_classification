import os
import time

from Bio import Entrez
from log.custom_log import logger

from common.env_config import config

Entrez.email = "quangsonvu9699@gmail.com"  # Replace with your email
Entrez.tool = "ncbi_batch_downloader"


def download_genbank_record(accession, retries=3, retry_delay=5, file_type="gb"):
    """
    Download GenBank record from NCBI with retry mechanism

    Args:
        accession: Accession number of the sequence
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds
        file_type: File type to save (gb or fasta)

    Returns:
        GenBank record content as string or None if failed
    """
    if file_type not in ["gb", "fasta"]:
        logger.error(f"Invalid record type: {file_type}")
        raise ValueError(f"Invalid record type: {file_type}")

    for attempt in range(retries):
        try:
            # Handle connection errors with retries
            try:
                handle = Entrez.esearch(db="nucleotide", term=accession)
                record = Entrez.read(handle)
                handle.close()
            except Exception as e:
                logger.warning(f"Search error for {accession} (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(retry_delay)
                continue

            # Check if record was found
            if not record["IdList"]:
                logger.warning(f"No record found for {accession}")
                return None

            id_list = record["IdList"]

            # Fetch the GenBank record with retry
            try:
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=id_list[0],
                    rettype=file_type,
                    retmode="text"
                )
                content = handle.read()
                handle.close()

                logger.info(f"Successfully downloaded {accession}")
                return content

            except Exception as e:
                logger.warning(f"Fetch error for {accession} (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(retry_delay)
                continue

        except Exception as e:
            logger.error(f"Error downloading {accession} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

    logger.error(f"Failed to download {accession} after {retries} attempts")
    return None


def save_genbank_record(content, accession, final_output, file_name=None):
    """
    Save GenBank record content to file

    Args:
        content: GenBank record content as string
        accession: Accession number of the sequence
        final_output: Path to the output file
        file_name: Output filename (optional)

    Returns:
        Path to the saved file or None if failed
    """

    try:
        # Write to file
        with open(final_output, "w") as out_handle:
            out_handle.write(content)

        # Mark as complete
        if file_name:
            with open(os.path.join(config.CACHE_FOLDER, f"{file_name}.done"), "w") as f:
                f.write("done")

        logger.info(f"Successfully saved {accession} to {final_output}")
        return final_output

    except Exception as e:
        logger.error(f"Error saving {accession}: {e}")
        return None


def download_genbank_by_accession_number(accession, output_dir, file_name=None, file_type="gb", retries=3,
                                         retry_delay=5):
    """
    Download GenBank record from NCBI and save to file with retry mechanism

    Args:
        accession: Accession number of the sequence
        output_dir: Directory to save the file
        file_name: Output filename (optional)
        file_type: File type to save (gb or fasta)
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Path to the saved file or None if failed
    """
    if file_type not in ["gb", "fasta"]:
        logger.error(f"Invalid record type: {file_type}")
        raise ValueError(f"Invalid record type: {file_type}")

    # Skip if already downloaded
    if file_name:
        cache_file = os.path.join(config.CACHE_FOLDER, f"{file_name}.done")
        final_output = os.path.join(output_dir, f"{file_name}.{file_type}")
        if os.path.exists(cache_file) and os.path.exists(final_output):
            logger.info(f"Skipping {accession} - already downloaded")
            return

    # Download the GenBank record
    content = download_genbank_record(accession, retries, retry_delay, file_type)
    if content is None:
        return None

    # Save the record to file
    return save_genbank_record(content, accession, final_output, file_name)
