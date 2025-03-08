from Bio import Entrez
import os
import pandas as pd
import time

Entrez.email = "your_email@example.com"  # Thay bằng email của bạn


def download_fasta_files(accession_numbers, output_directory="fasta_files", dataset_type=""):
    """
    Tải nhiều file FASTA từ NCBI

    :param accession_numbers: Danh sách các accession number
    :param output_directory: Thư mục lưu file (mặc định là fasta_files)
    :param dataset_type: Loại dataset (Training/Test)
    :return: DataFrame chứa thông tin về các file đã tải
    """
    os.makedirs(output_directory, exist_ok=True)

    download_info = []

    for accession in accession_numbers:
        try:
            handle = Entrez.esearch(db="nucleotide", term=accession)
            record = Entrez.read(handle)
            handle.close()

            if record["IdList"]:
                id_list = record["IdList"]

                handle = Entrez.efetch(db="nucleotide",
                                       id=id_list[0],
                                       rettype="fasta",
                                       retmode="text")

                output_file = os.path.join(output_directory, f"{accession}.fasta")

                # Ghi file
                with open(output_file, "w") as out_handle:
                    out_handle.write(handle.read())

                handle.close()

                download_info.append({
                    "Accession number": accession,
                    "Filename": f"{accession}.fasta",
                    "Dataset": dataset_type
                })

                print(f"Tải thành công file {output_file}")

                time.sleep(0.5)

            else:
                print(f"Không tìm thấy bản ghi cho accession number {accession}")

        except Exception as e:
            print(f"Đã xảy ra lỗi với {accession}: {e}")

    df_download = pd.DataFrame(download_info)

    return df_download


excel_file = 'journal.pcbi.1012525.s001.xlsx'
training_df = pd.read_excel(excel_file, sheet_name='Training dataset')
test_df = pd.read_excel(excel_file, sheet_name='Test dataset')

training_fasta_info = download_fasta_files(
    training_df['Accession number'],
    output_directory="fasta_files/training",
    dataset_type="Training"
)

test_fasta_info = download_fasta_files(
    test_df['Accession number'],
    output_directory="fasta_files/test",
    dataset_type="Test"
)


def merge_download_info(original_df, download_info_df):
    merged_df = original_df.merge(
        download_info_df,
        on='Accession number',
        how='left'
    )
    return merged_df


training_full_info = merge_download_info(training_df, training_fasta_info)
test_full_info = merge_download_info(test_df, test_fasta_info)

training_full_info.to_csv('training_dataset_with_fasta.csv', index=False)
test_full_info.to_csv('test_dataset_with_fasta.csv', index=False)

print("Quá trình tải hoàn tất. Kiểm tra các file CSV và thư mục fasta_files.")