import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Literal

from Bio import SeqIO
from Bio.Seq import Seq

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DNAPreprocessor:
    def __init__(self, mode: str = 'trim'):
        """
        Khởi tạo DNA Preprocessor để chuyển đổi DNA sang Protein

        Args:
            mode: Chế độ xử lý chuỗi DNA không chia hết cho 3
                - 'trim': Cắt bỏ các nucleotide dư ở cuối
                - 'fill': Thêm 'N' vào cuối để đủ bộ ba
        """
        if mode not in ['trim', 'fill']:
            raise ValueError("Mode must be either 'trim' or 'fill'")

        logger.info(f"Initialized DNA Preprocessor with mode: {mode}")
        self.mode = mode

    def process_dna_sequence(self, dna_sequence: str) -> str:
        """
        Xử lý chuỗi DNA theo mode đã chọn

        Args:
            dna_sequence: Chuỗi DNA cần xử lý

        Returns:
            str: Chuỗi DNA đã xử lý
        """
        remainder = len(dna_sequence) % 3

        if remainder == 0:
            return dna_sequence

        if self.mode == 'trim':
            # Cắt sequence để độ dài chia hết cho 3
            trimmed_length = (len(dna_sequence) // 3) * 3
            processed_seq = dna_sequence[:trimmed_length]
            logger.debug(f"Trimmed {remainder} nucleotides")
            return processed_seq

        else:  # mode == 'fill'
            # Thêm 'N' để độ dài chia hết cho 3
            padding_length = 3 - remainder
            processed_seq = dna_sequence + 'N' * padding_length
            logger.debug(f"Added {padding_length} 'N' nucleotides")
            return processed_seq

    def dna_to_protein(self, dna_sequence: str) -> str:
        """
        Chuyển đổi một chuỗi DNA sang protein

        Args:
            dna_sequence: Chuỗi DNA cần chuyển đổi

        Returns:
            str: Chuỗi protein (đã loại bỏ stop codon)
        """
        try:
            # Xử lý DNA theo mode đã chọn
            processed_dna = self.process_dna_sequence(dna_sequence)

            # Chuyển đổi sang protein
            protein = str(Seq(processed_dna).translate())

            # Loại bỏ stop codon
            protein = protein.rstrip("*")  # ở cuối chuỗi
            protein = protein.replace("*", "")  # ở giữa chuỗi

            return protein

        except Exception as e:
            logger.error(f"Error converting DNA to protein: {str(e)}")
            return ""

    def read_fasta(self, fasta_file: str) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Đọc và xử lý file FASTA

        Args:
            fasta_file: Đường dẫn đến file FASTA

        Returns:
            Tuple chứa:
            - Dict[str, str]: dictionary {sequence_id: protein_sequence}
            - Dict[str, int]: thống kê về quá trình xử lý
        """
        if not Path(fasta_file).exists():
            raise FileNotFoundError(f"File not found: {fasta_file}")

        sequences = {}
        stats = {
            'total_sequences': 0,
            'processed_sequences': 0,
            'invalid_sequences': 0,
            'sequences_modified': 0  # Đổi tên từ sequences_trimmed
        }

        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                stats['total_sequences'] += 1
                original_length = len(record.seq)

                # Chuyển DNA sang protein
                protein = self.dna_to_protein(str(record.seq))

                if protein:
                    sequences[record.id] = protein
                    stats['processed_sequences'] += 1

                    # Đánh dấu sequences cần xử lý theo mode
                    if original_length % 3 != 0:
                        stats['sequences_modified'] += 1
                else:
                    stats['invalid_sequences'] += 1

            logger.info(f"Processing stats ({self.mode} mode): {stats}")

        except Exception as e:
            logger.error(f"Error processing FASTA file: {str(e)}")
            raise

        return sequences, stats


def save_proteins_to_fasta(proteins: Dict[str, str], output_file: str) -> None:
    """
    Lưu proteins vào file FASTA

    Args:
        proteins: Dictionary chứa cặp sequence_id và protein sequence
        output_file: Đường dẫn file output
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            for seq_id, protein in proteins.items():
                f.write(f'>{seq_id}\n')
                # Chia protein thành các dòng 60 ký tự
                for i in range(0, len(protein), 60):
                    f.write(f'{protein[i:i + 60]}\n')

        logger.info(f"Successfully saved proteins to {output_file}")

    except Exception as e:
        logger.error(f"Error saving proteins to FASTA file: {str(e)}")
        raise


def process_fasta_file(fasta_file: str, mode: str = 'trim') -> Tuple[
    Dict[str, str], Dict[str, int]]:
    """
    Hàm tiện ích để xử lý file FASTA

    Args:
        fasta_file: Đường dẫn đến file FASTA
        mode: Chế độ xử lý ('trim' hoặc 'fill')

    Returns:
        Tuple chứa proteins và thống kê
    """
    preprocessor = DNAPreprocessor(mode=mode)
    return preprocessor.read_fasta(fasta_file)


# Ví dụ sử dụng:
if __name__ == "__main__":
    # # Ví dụ sử dụng:
    # # 1. Xử lý một chuỗi DNA đơn lẻ với mode 'fill'
    # fill_preprocessor = DNAPreprocessor(mode='fill')
    # dna = "ATGGCCACTGAAGCTGGTA"  # DNA sequence có độ dài 19 (không chia hết cho 3)
    # protein = fill_preprocessor.dna_to_protein(dna)
    # print(f"Using fill mode:")
    # print(f"DNA: {dna}")
    # print(f"Processed DNA: {fill_preprocessor.process_dna_sequence(dna)}")
    # print(f"Protein: {protein}\n")
    #
    # # 2. Xử lý cùng chuỗi DNA với mode 'trim'
    # trim_preprocessor = DNAPreprocessor(mode='trim')
    # protein = trim_preprocessor.dna_to_protein(dna)
    # print(f"Using trim mode:")
    # print(f"DNA: {dna}")
    # print(f"Processed DNA: {trim_preprocessor.process_dna_sequence(dna)}")
    # print(f"Protein: {protein}\n")
    #
    # # 3. Xử lý file FASTA
    # try:
    #     input_fasta = "path/to/your/input.fasta"
    #
    #     # Xử lý với mode 'fill'
    #     print("\nProcessing with fill mode:")
    #     proteins_fill, stats_fill = process_fasta_file(input_fasta, mode='fill')
    #
    #     # Xử lý với mode 'trim'
    #     print("\nProcessing with trim mode:")
    #     proteins_trim, stats_trim = process_fasta_file(input_fasta, mode='trim')
    #
    # except FileNotFoundError:
    #     print("Please provide a valid FASTA file path")
    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")

    mode = 'fill'
    type = 'test'
    data_dir = f"../data/my_data/original_data/{type}"
    result_dir = f"../data/my_data/protein_format/{mode}/{type}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".fna"):
                    input_fna = os.path.join(folder_path, file)
                    output_fna = os.path.join(result_dir, folder, file.replace(".fna", ".fasta"))
                    if not os.path.exists(os.path.join(result_dir, folder)):
                        os.mkdir(os.path.join(result_dir, folder))
                    try:
                        proteins, stats = process_fasta_file(input_fna, mode=mode)
                        save_proteins_to_fasta(proteins, output_fna)
                    except FileNotFoundError:
                        print("Please provide a valid FASTA file path")
                    except Exception as e:
                        print(f"An error occurred: {str(e)}")
