import concurrent.futures
import logging
import os
import time

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ncbi_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Entrez.email = "your_email@example.com"  # Thay bằng email của bạn
# Entrez.api_key = "your_api_key"  # Đăng ký API key tại: https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/

output_directory = "../data/ncbi_data"
os.makedirs(output_directory, exist_ok=True)

MAX_WORKERS = 3
REQUEST_DELAY = 0.5  # giây


def download_and_modify_fasta(row, file_name=None):
    """
    Download FASTA files from NCBI and add custom labels and descriptions

    :param row: Row data from DataFrame
    :param file_name: Output file name
    :return: Tuple (accession_number, status, error_message_or_temp_file_path)
    """
    accession_number = row['Accession number']

    custom_labels = {
        "lifecycle": row["Lifecycle"],
        "usage": row["Usage"],
        # "cross_validation_group": row["5-fold cross-validation"]
    }

    output_file = os.path.join(output_directory, f"{file_name}.fasta")

    try:
        time.sleep(REQUEST_DELAY)

        handle = Entrez.esearch(db="nucleotide", term=accession_number)
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            return accession_number, False, "Record not found"

        id_list = record["IdList"]

        handle = Entrez.efetch(
            db="nucleotide",
            id=id_list[0],
            rettype="fasta",
            retmode="text"
        )

        seq_record = SeqIO.read(handle, "fasta")
        handle.close()

        new_description = seq_record.description

        label_str = ""
        for label, value in custom_labels.items():
            label_str += f" [{label}={value}]"

        new_description += label_str

        seq_data = SeqRecord(
            seq=seq_record.seq,
            id=seq_record.id,
            description=new_description
        )

        temp_file = os.path.join(output_directory, "temp_files", f"temp_{accession_number}_{file_name}.fasta")

        # Ghi vào file tạm của riêng nó
        with open(temp_file, "w") as out_file:
            SeqIO.write(seq_data, out_file, "fasta")

        return accession_number, True, temp_file

    except Exception as e:
        error_message = str(e)
        return accession_number, False, error_message


def download_data_parallel(file_path, output_filename):
    """
    Download data from NCBI in parallel using ThreadPoolExecutor

    :param file_path: Path to Excel file containing list of accession numbers
    :param output_filename: Output file name (without extension)
    :return: Tuple (successful_count, failed_count, total_count, failed_records_list)
    """
    try:
        excel_df = pd.read_excel(file_path)
        total_records = len(excel_df)

        output_file = os.path.join(output_directory, f"{output_filename}.fasta")
        if os.path.exists(output_file):
            os.remove(output_file)

        temp_dir = os.path.join(output_directory, "temp_files")
        os.makedirs(temp_dir, exist_ok=True)

        logger.info(f"Starting to download {total_records} sequences from file {file_path}")

        successful = 0
        failed = 0
        failed_records = []
        successful_temp_files = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_row = {
                executor.submit(download_and_modify_fasta, row, output_filename): row['Accession number']
                for _, row in excel_df.iterrows()
            }

            with tqdm(total=total_records) as pbar:
                for future in concurrent.futures.as_completed(future_to_row):
                    accession_number = future_to_row[future]
                    try:
                        acc_num, status, result = future.result()
                        print(f"acc_num: {acc_num}, status: {status}")
                        if status:
                            successful += 1
                            successful_temp_files.append(result)  # result là đường dẫn đến file tạm
                        else:
                            failed += 1
                            failed_records.append((acc_num, result))  # result là thông báo lỗi
                            logger.warning(f"Error downloading {acc_num}: {result}")
                    except Exception as e:
                        failed += 1
                        failed_records.append((accession_number, str(e)))
                        logger.error(f"Unexpected error with {accession_number}: {e}")

                    pbar.update(1)

        # Merge all temporary files into a final file
        logger.info(f"Merging {len(successful_temp_files)} temporary files into the final output...")
        with open(output_file, "w") as final_file:
            for temp_file in successful_temp_files:
                try:
                    records = list(SeqIO.parse(temp_file, "fasta"))
                    SeqIO.write(records, final_file, "fasta")
                    os.remove(temp_file)
                except Exception as e:
                    logger.error(f"Error when merging file {temp_file}: {e}")

        try:
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

        # Log the list of failed records
        if failed_records:
            logger.warning(f"List of failed records ({len(failed_records)}):")
            for acc_num, error in failed_records:
                logger.warning(f"  - {acc_num}: {error}")

        logger.info(f"Download completed: {successful}/{total_records} successful, {failed}/{total_records} failed")

        return successful, failed, total_records, failed_records

    except Exception as e:
        logger.error(f"Lỗi khi xử lý file {file_path}: {e}")
        return 0, 0, 0


def retry_failed_downloads(failed_list, output_filename, max_retries=3):
    """
    Retry downloading failed records

    :param failed_list: List of failed records (accession_number, error)
    :param output_filename: Output file name
    :param max_retries: Maximum number of retry attempts
    :return: Tuple (successful_count, still_failed_count)
    """
    if not failed_list:
        return 0, 0

    logger.info(f"Retrying download of {len(failed_list)} failed records...")

    retry_df = pd.DataFrame({
        'Accession number': [item[0] for item in failed_list],
        'Lifecycle': ['unknown'] * len(failed_list),
        'Usage': ['retry'] * len(failed_list),
        # '5-fold cross-validation': ['retry'] * len(failed_list)
    })

    successful = 0
    still_failed = []

    temp_dir = os.path.join(output_directory, "temp_retry_files")
    os.makedirs(temp_dir, exist_ok=True)

    output_file = os.path.join(output_directory, f"{output_filename}.fasta")

    for retry in range(max_retries):
        if not retry_df.empty:
            logger.info(f"Retry attempt {retry + 1}/{max_retries} for {len(retry_df)} records...")

            successful_temp_files = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_row = {
                    executor.submit(download_and_modify_fasta, row, f"{output_filename}_retry{retry + 1}"): row[
                        'Accession number']
                    for _, row in retry_df.iterrows()
                }

                for future in concurrent.futures.as_completed(future_to_row):
                    accession_number = future_to_row[future]
                    try:
                        acc_num, status, result = future.result()
                        if status:
                            successful += 1
                            successful_temp_files.append(result)
                            retry_df = retry_df[retry_df['Accession number'] != acc_num]
                        else:
                            logger.warning(
                                f"Still failed ({retry + 1}/{max_retries}) when downloading {acc_num}: {result}")
                    except Exception as e:
                        logger.error(f"Unexpected error ({retry + 1}/{max_retries}) with {accession_number}: {e}")

            with open(output_file, "a") as final_file:
                for temp_file in successful_temp_files:
                    try:
                        records = list(SeqIO.parse(temp_file, "fasta"))
                        SeqIO.write(records, final_file, "fasta")

                        os.remove(temp_file)
                    except Exception as e:
                        logger.error(f"Lỗi khi gộp file {temp_file}: {e}")

            if retry < max_retries - 1 and not retry_df.empty:
                time.sleep(REQUEST_DELAY * 5)

    still_failed = retry_df['Accession number'].tolist()

    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except:
        pass

    return successful, len(still_failed)


if __name__ == "__main__":
    start_time = time.time()

    # Create temp directory to store individual files
    temp_dir = os.path.join(output_directory, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)

    # Download data from training file
    # logger.info("=== Downloading training dataset ===")
    # train_success, train_failed, train_total, train_failed_records = download_data_parallel("train_dataset.xlsx",
    #                                                                                         "train_dataset")

    # Retry failed downloads from training set (if any)
    # if train_failed > 0:
    #     logger.info(f"Retrying {train_failed} failed records from training dataset...")
    #     retry_success, retry_still_failed = retry_failed_downloads(train_failed_records, "train_dataset")
    #     # Cập nhật số lượng thành công
    #     train_success += retry_success
    #     train_failed = retry_still_failed

    # Download data from test file
    logger.info("\n=== Downloading test dataset ===")
    test_success, test_failed, test_total, test_failed_records = download_data_parallel(
        "../../data/deep_pl_data/test_dataset.xlsx",
                                                                                        "test_dataset")

    if test_failed > 0:
        logger.info(f"Retrying {test_failed} failed records from test dataset...")
        retry_success, retry_still_failed = retry_failed_downloads(test_failed_records, "test_dataset")

        test_success += retry_success
        test_failed = retry_still_failed

    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Could not delete temp directory: {e}")

    # Summary
    total_time = time.time() - start_time
    logger.info("\n=== Summary ===")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    # logger.info(
    #     f"Training dataset: {train_success}/{train_total} sequences downloaded successfully ({train_failed} failed)")
    # logger.info(f"Test dataset: {test_success}/{test_total} sequences downloaded successfully ({test_failed} failed)")

    # Calculate overall success rate
    # overall_success_rate = ((train_success + test_success) / (train_total + test_total)) * 100
    overall_success_rate = (test_success / test_total) * 100
    logger.info(f"Overall success rate: {overall_success_rate:.2f}%")
