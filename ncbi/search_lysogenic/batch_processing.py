import argparse
import os
import re
import sys

from Bio import SeqIO


def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Tìm kiếm các gen phage trong file GenBank')
    parser.add_argument('-i', '--input', required=True, help='Đường dẫn đến file GenBank đầu vào')
    parser.add_argument('-o', '--output', help='Đường dẫn đến file kết quả đầu ra (mặc định: phage_genes_results.txt)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Hiển thị thông tin chi tiết trong quá trình xử lý')
    parser.add_argument('-p', '--pattern', help='Đường dẫn đến file JSON chứa các pattern tìm kiếm tùy chỉnh')
    return parser


def load_genbank_file(file_path):
    try:
        record = SeqIO.read(file_path, "genbank")
        return record
    except Exception as e:
        print(f"Lỗi khi đọc file GenBank: {e}")
        sys.exit(1)


def search_phage_genes(record, target_genes_dict, verbose=False):
    results = {}
    gene_counter = {gene: 0 for gene in target_genes_dict.keys()}

    if verbose:
        print(f"Đang tìm kiếm trong trình tự: {record.id}")
        print(f"Số lượng features: {len(record.features)}")
        print(f"Tìm kiếm các gen sau:")
        for gene, variants in target_genes_dict.items():
            print(f"  - {gene}: {len(variants)} biến thể")

    patterns = {}
    for gene_name, variants in target_genes_dict.items():
        pattern_strs = [re.escape(variant) for variant in variants]
        combined_pattern = '|'.join(pattern_strs)
        patterns[gene_name] = re.compile(combined_pattern, re.IGNORECASE)

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

        for gene_name, pattern in patterns.items():
            found = False

            for qualifier, value in feature_info.items():
                if pattern.search(value):
                    found = True
                    break

            if found:

                if gene_name not in results:
                    results[gene_name] = []

                gene_counter[gene_name] += 1

                gene_result = {
                    'type': feature.type,
                    'location': str(feature.location),
                    'strand': '+' if feature.location.strand == 1 else '-',
                    'info': feature_info,
                    'count': gene_counter[gene_name]
                }

                results[gene_name].append(gene_result)

                if verbose:
                    print(f"Tìm thấy: {gene_name} #{gene_counter[gene_name]} tại {feature.location}")

    return results


def format_results(results):
    if not results:
        return "Không tìm thấy gen phage nào trong file GenBank đã cho."

    total_genes_found = sum(len(findings) for findings in results.values())

    output = "KẾT QUẢ TÌM KIẾM GEN PHAGE\n"
    output += "=" * 50 + "\n"
    output += f"TỔNG SỐ GEN TÌM THẤY: {total_genes_found}\n\n"

    output += "BẢNG THỐNG KÊ:\n"
    output += "-" * 50 + "\n"
    output += f"{'TÊN GEN':<30} {'SỐ LƯỢNG':>10}\n"
    output += "-" * 50 + "\n"

    for gene_name, findings in sorted(results.items()):
        output += f"{gene_name:<30} {len(findings):>10}\n"

    output += "-" * 50 + "\n\n"

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
    default_patterns = {
        "integrase": [
            "integrase", "int gene", "int protein", "site-specific integrase",
            "phage integrase", "tyrosine integrase", "serine integrase", "int"
        ],
        "excisionase": [
            "excisionase", "xis gene", "xis protein", "recombination directionality factor",
            "rdf", "excision protein", "xis"
        ],
        "recombinase": [
            "recombinase", "site-specific recombinase", "tyrosine recombinase",
            "serine recombinase", "dna recombinase"
        ],
        "regulatory protein cro": [
            "regulatory protein cro", "cro protein", "cro repressor", "cro",
            "transcriptional regulator cro"
        ],
        "antitermination protein Q": [
            "antitermination protein Q", "protein Q", "Q protein",
            "late gene regulator Q", "transcription antiterminator Q", "Q"
        ],
        "cI repressor": [
            "cI repressor", "ci repressor", "cI protein", "lambda repressor",
            "immunity repressor", "repressor protein cI", "cl protein", "cl repressor", "cI", "ci"
        ],
        "cII protein": [
            "cII protein", "cii protein", "protein cII", "transcription activator cII",
            "cII", "cii"
        ],
        "cIII protein": [
            "cIII protein", "ciii protein", "protein cIII", "cIII stabilization protein",
            "cIII", "ciii"
        ],
        "replication protein O": [
            "replication protein O", "O protein", "protein O", "replication initiation protein O",
            "phage replication protein O", "dna replication protein O", "O"
        ],
        "replication protein P": [
            "replication protein P", "P protein", "protein P", "replication initiation protein P",
            "phage replication protein P", "dna replication protein P", "P"
        ],
        "recombination protein Bet": [
            "recombination protein Bet", "Bet protein", "bet protein", "bet", "Bet", "BET",
            "ssDNA annealing protein", "single-strand DNA binding protein",
            "DNA single-strand annealing protein", "RecT-like recombination protein",
            "Redβ", "Redβ recombinase", "Red beta protein", "Red beta recombination protein",
            "Redbeta", "Red-beta", "single strand annealing protein", "SSAP", "Bet", "bet"
        ]
    }

    data_dir = "../../data/ncbi_data/gen_bank/train/Lysogneic"
    for group in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, group)):
            record = load_genbank_file(os.path.join(data_dir, group, file))
            # Tìm kiếm các gen phage với các biến thể
            results = search_phage_genes(record, default_patterns, True)

            # Định dạng và hiển thị kết quả
            formatted_results = format_results(results)
            print("\n" + formatted_results)


if __name__ == "__main__":
    main()
