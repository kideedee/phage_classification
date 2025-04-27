import argparse
import re
import sys

from Bio import SeqIO


def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Tìm kiếm các gen phage trong file GenBank')
    parser.add_argument('-i', '--input', required=True, help='Đường dẫn đến file GenBank đầu vào')
    parser.add_argument('-o', '--output', help='Đường dẫn đến file kết quả đầu ra (mặc định: phage_genes_results.txt)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Hiển thị thông tin chi tiết trong quá trình xử lý')
    return parser


def load_genbank_file(file_path):
    try:
        record = SeqIO.read(file_path, "genbank")
        return record
    except Exception as e:
        print(f"Lỗi khi đọc file GenBank: {e}")
        sys.exit(1)


def search_phage_genes(record, target_genes, verbose=False):
    results = {}
    gene_counter = {gene: 0 for gene in target_genes}

    if verbose:
        print(f"Đang tìm kiếm trong trình tự: {record.id}")
        print(f"Số lượng features: {len(record.features)}")

    patterns = {gene: re.compile(gene, re.IGNORECASE) for gene in target_genes}

    for feature in record.features:
        if feature.type not in ["gene", "CDS", "misc_feature"]:
            continue

        qualifiers_to_check = ["gene", "product", "note", "function", "protein_id"]
        feature_info = {}

        for qualifier in qualifiers_to_check:
            if qualifier in feature.qualifiers:
                feature_info[qualifier] = feature.qualifiers[qualifier][0]

        if not feature_info:
            continue

        # Tìm kiếm từng gen trong các thông tin của feature
        for gene_name, pattern in patterns.items():
            found = False

            for qualifier, value in feature_info.items():
                if pattern.search(value):
                    found = True
                    break

            if found:
                # Nếu tìm thấy, thêm vào kết quả
                if gene_name not in results:
                    results[gene_name] = []

                # Cập nhật bộ đếm
                gene_counter[gene_name] += 1

                # Thêm thông tin về gene này
                gene_result = {
                    'type': feature.type,
                    'location': str(feature.location),
                    'strand': '+' if feature.location.strand == 1 else '-',
                    'info': feature_info,
                    'count': gene_counter[gene_name]  # Thêm số thứ tự cho gen này
                }

                results[gene_name].append(gene_result)

                if verbose:
                    print(f"Tìm thấy: {gene_name} #{gene_counter[gene_name]} tại {feature.location}")

    return results


def format_results(results):
    """
    Định dạng kết quả tìm kiếm để hiển thị hoặc lưu vào file

    Args:
        results: Từ điển chứa kết quả tìm kiếm

    Returns:
        Chuỗi đã được định dạng
    """
    if not results:
        return "Không tìm thấy gen phage nào trong file GenBank đã cho."

    # Tính tổng số gen tìm thấy
    total_genes_found = sum(len(findings) for findings in results.values())

    output = "KẾT QUẢ TÌM KIẾM GEN PHAGE\n"
    output += "=" * 50 + "\n"
    output += f"TỔNG SỐ GEN TÌM THẤY: {total_genes_found}\n\n"

    # Tạo bảng tóm tắt số lượng của mỗi loại gen
    output += "BẢNG THỐNG KÊ:\n"
    output += "-" * 50 + "\n"
    output += f"{'TÊN GEN':<30} {'SỐ LƯỢNG':>10}\n"
    output += "-" * 50 + "\n"

    for gene_name, findings in sorted(results.items()):
        output += f"{gene_name:<30} {len(findings):>10}\n"

    output += "-" * 50 + "\n\n"

    # Chi tiết về từng gen
    output += "CHI TIẾT:\n"
    output += "=" * 50 + "\n\n"

    for gene_name, findings in sorted(results.items()):
        output += f"GEN: {gene_name}\n"
        output += "-" * 50 + "\n"

        if not findings:
            output += "    Không tìm thấy\n"
        else:
            output += f"    Số lượng tìm thấy: {len(findings)}\n\n"

            for finding in findings:
                output += f"    Kết quả #{finding['count']}:\n"
                output += f"    - Loại: {finding['type']}\n"
                output += f"    - Vị trí: {finding['location']}\n"
                output += f"    - Chuỗi: {finding['strand']}\n"

                for key, value in finding['info'].items():
                    output += f"    - {key}: {value}\n"

                output += "\n"

        output += "\n"

    return output


def save_results_to_file(formatted_results, output_file):
    """Lưu kết quả đã định dạng vào file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_results)
        print(f"Đã lưu kết quả vào file: {output_file}")
    except Exception as e:
        print(f"Lỗi khi lưu kết quả: {e}")


def main():
    # Danh sách các gen phage cần tìm kiếm
    target_genes = [
        "integrase",
        "excisionase",
        "recombinase",
        "regulatory protein cro",
        "antitermination protein Q",
        "cI repressor",
        "cII protein",
        "cIII protein",
        "replication protein O",
        "replication protein P",
        "recombination protein Bet"
    ]

    # parser = setup_argument_parser()
    # args = parser.parse_args()

    # output_file = args.output if args.output else "phage_genes_results.txt"
    output_file = "phage_genes_results.txt"

    record = load_genbank_file("search_lysogenic/train_NC_000902_Lysogneic_Group1.gb")

    results = search_phage_genes(record, target_genes, True)

    formatted_results = format_results(results)
    print("\n" + formatted_results)

    save_results_to_file(formatted_results, output_file)


if __name__ == "__main__":
    main()
