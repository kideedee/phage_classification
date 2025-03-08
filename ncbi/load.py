from Bio import SeqIO


def read_modified_fasta(fasta_file):
    """
    Đọc file FASTA và trích xuất các thông tin nhãn tùy chỉnh

    :param fasta_file: Đường dẫn đến file FASTA cần đọc
    :return: List các dict chứa thông tin trình tự và nhãn
    """

    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_id = record.id
            sequence = str(record.seq)
            description = record.description

            labels = {}

            # Tìm tất cả các nhãn có định dạng [label=value]
            import re
            label_matches = re.findall(r'\[([^=]+)=([^\]]+)\]', description)

            for label, value in label_matches:
                labels[label] = value

            output_file = f'{labels["lifecycle"]}_{labels["usage"]}_{labels["cross_validation_group"]}'
            with open(output_file, "a") as output_file:
                SeqIO.write(record, output_file, "fasta")

            # In thông tin để kiểm tra
            print(f"ID: {seq_id}")
            print(f"Description: {description}")
            print(f"Sequence length: {len(sequence)}")
            print("Custom labels:")
            for label, value in labels.items():
                print(f"  - {label}: {value}")
            print()

    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file FASTA: {e}")
        raise e


# Ví dụ sử dụng
if __name__ == "__main__":
    modified_file = "../data/ncbi_data/train_dataset.fasta"

    sequences = read_modified_fasta(modified_file)

    # for seq in sequences:
    #     # Kiểm tra xem có nhãn "Lifecycle" không
    #     if "Lifecycle" in seq["labels"]:
    #         lifecycle = seq["labels"]["Lifecycle"]
    #         print(f"Trình tự {seq['id']} có lifecycle: {lifecycle}")
    #
    #     # Kiểm tra xem có nhãn "Cross validation group" không
    #     if "Cross validation group" in seq["labels"]:
    #         group = seq["labels"]["Cross validation group"]
    #         print(f"Trình tự {seq['id']} thuộc nhóm: {group}")
