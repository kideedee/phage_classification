#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tìm kiếm các gen phage trong file GenBank
"""

import argparse
import os
import re
import sys

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

output_dir = "data"


def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Tìm kiếm các gen phage trong file GenBank')
    parser.add_argument('-i', '--input', required=True, help='Đường dẫn đến file GenBank đầu vào')
    parser.add_argument('-o', '--output', help='Đường dẫn đến file kết quả đầu ra (mặc định: phage_genes_results.txt)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Hiển thị thông tin chi tiết trong quá trình xử lý')
    parser.add_argument('-p', '--pattern', help='Đường dẫn đến file JSON chứa các pattern tìm kiếm tùy chỉnh')
    return parser


def load_genbank_file(file_path):
    """Đọc file GenBank và trả về record"""
    try:
        record = SeqIO.read(file_path, "genbank")
        return record
    except Exception as e:
        print(f"Lỗi khi đọc file GenBank: {e}")
        sys.exit(1)


def search_phage_genes(record, target_genes_dict, verbose=False):
    """
    Tìm kiếm các gen phage trong record GenBank

    Args:
        record: Record GenBank từ Biopython
        target_genes_dict: Từ điển chứa các gen cần tìm và các biến thể của chúng
        verbose: Hiển thị thông tin chi tiết trong quá trình xử lý

    Returns:
        Từ điển chứa các gen tìm thấy và thông tin của chúng
    """
    results = {}
    gene_counter = {gene: 0 for gene in target_genes_dict.keys()}

    if verbose:
        print(f"Đang tìm kiếm trong trình tự: {record.id}")
        print(f"Số lượng features: {len(record.features)}")
        print(f"Tìm kiếm các gen sau:")
        for gene, variants in target_genes_dict.items():
            print(f"  - {gene}: {len(variants)} biến thể")

    # Tạo các pattern regex từ danh sách biến thể (không phân biệt chữ hoa/thường)
    patterns = {}
    for gene_name, variants in target_genes_dict.items():
        # Tạo pattern cho mỗi biến thể của gen
        pattern_strs = [re.escape(variant) for variant in variants]
        # Kết hợp tất cả biến thể bằng toán tử OR (|)
        combined_pattern = '|'.join(pattern_strs)
        # Biên dịch pattern với cờ IGNORECASE
        patterns[gene_name] = re.compile(combined_pattern, re.IGNORECASE)

    # Duyệt qua tất cả các features trong record
    for feature in record.features:
        # Chỉ quan tâm đến các features có type là gene, CDS hoặc misc_feature
        # if feature.type not in ["gene", "CDS", "misc_feature"]:
        if feature.type not in ["gene"]:
            continue

        # Lấy tất cả các qualifier có thể chứa thông tin về gen
        # qualifiers_to_check = ["gene", "product", "note", "function", "protein_id"]
        qualifiers_to_check = ["gene"]
        feature_info = {}

        for qualifier in qualifiers_to_check:
            if qualifier in feature.qualifiers:
                feature_info[qualifier] = feature.qualifiers[qualifier][0]

        # Nếu không có thông tin hữu ích, bỏ qua feature này
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

                # Trích xuất trình tự gen
                try:
                    gene_sequence = feature.location.extract(record.seq)
                except Exception as e:
                    print(f"Lỗi khi trích xuất trình tự gen {gene_name}: {e}")
                    gene_sequence = None

                # Thêm thông tin về gene này
                gene_result = {
                    'type': feature.type,
                    'location': str(feature.location),
                    'strand': '+' if feature.location.strand == 1 else '-',
                    'info': feature_info,
                    'count': gene_counter[gene_name]  # Thêm số thứ tự cho gen này
                }

                # Nếu có trình tự gen, thêm và lưu file FASTA
                if gene_sequence:
                    gene_result['sequence'] = str(gene_sequence)

                    # Tạo tên file FASTA
                    safe_gene_name = re.sub(r'[^\w\-_\.]', '_', gene_name)
                    fasta_filename = os.path.join(output_dir,
                                                  f"{record.id}_{safe_gene_name}_{gene_counter[gene_name]}.fasta")

                    # Tạo SeqRecord để lưu
                    gene_record = SeqRecord(
                        Seq(str(gene_sequence)),
                        id=f"{record.id}_{gene_name}_{gene_counter[gene_name]}",
                        description=f"Extracted {gene_name} gene from {record.id}"
                    )

                    # Lưu file FASTA
                    try:
                        SeqIO.write(gene_record, fasta_filename, "fasta")
                        gene_result['fasta_file'] = fasta_filename
                    except Exception as e:
                        print(f"Lỗi khi lưu file FASTA cho gen {gene_name}: {e}")

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


def save_patterns_to_json(patterns, filename="phage_gene_patterns.json"):
    """Lưu các pattern tìm kiếm vào file JSON để có thể tùy chỉnh sau này"""
    try:
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)
        print(f"Đã lưu các pattern tìm kiếm vào file: {filename}")
        print(f"Bạn có thể chỉnh sửa file này và sử dụng tham số -p để sử dụng các pattern tùy chỉnh.")
    except Exception as e:
        print(f"Lỗi khi lưu pattern: {e}")


def load_custom_patterns(pattern_file):
    """
    Đọc các pattern tìm kiếm tùy chỉnh từ file JSON

    Args:
        pattern_file: Đường dẫn đến file JSON

    Returns:
        Từ điển chứa các gen và các biến thể của chúng
    """
    try:
        import json
        with open(pattern_file, 'r', encoding='utf-8') as f:
            custom_patterns = json.load(f)
        return custom_patterns
    except Exception as e:
        print(f"Lỗi khi đọc file pattern: {e}")
        print("Sử dụng pattern mặc định.")
        return None


def main():
    # Danh sách các gen phage cần tìm kiếm với các biến thể
    default_patterns = {
        "integrase": ["int"],
        "excisionase": ["xis"],
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

    # Thiết lập đường dẫn file đầu ra
    output_file = args.output if args.output else "phage_genes_results.txt"

    # Đọc file GenBank
    record = load_genbank_file(args.input)

    # Kiểm tra xem có pattern tùy chỉnh không
    if args.pattern:
        custom_patterns = load_custom_patterns(args.pattern)
        if custom_patterns:
            target_patterns = custom_patterns
        else:
            target_patterns = default_patterns
    else:
        target_patterns = default_patterns

    # Tìm kiếm các gen phage với các biến thể
    results = search_phage_genes(record, target_patterns, args.verbose)

    # Định dạng và hiển thị kết quả
    formatted_results = format_results(results)
    print("\n" + formatted_results)

    # Lưu kết quả vào file
    save_results_to_file(formatted_results, output_file)

    # Lưu các pattern tìm kiếm vào file JSON để có thể tùy chỉnh sau này
    if not args.pattern:  # Chỉ lưu nếu không sử dụng pattern tùy chỉnh
        save_patterns_to_json(default_patterns)


if __name__ == "__main__":
    main()
