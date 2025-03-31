import glob
import os
import random
import re
from collections import Counter

import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction as GC
from sklearn.model_selection import train_test_split

lysogenic_train_dir = "../../data/ncbi_data/gen_bank/train/Lysogneic"
lytic_train_dir = "../../data/ncbi_data/gen_bank/train/Lytic"
lysogenic_test_dir = "../../data/ncbi_data/gen_bank/test/Lysogenic"
lytic_test_dir = "../../data/ncbi_data/gen_bank/test/Lytic"


def process_dna_sequence(row):
    seq = row['sequence'].upper()

    non_atgc = re.findall(r'[^ATGC]', seq)
    non_atgc_count = len(non_atgc)

    if 0 <= non_atgc_count <= 10:
        for char in set(non_atgc):
            replacement = random.choice(['A', 'T', 'G', 'C'])
            seq = seq.replace(char, replacement)

        return seq

    print(f"{seq}\n")
    return None


def analyze_genbank_file(file_path):
    try:
        record = SeqIO.read(file_path, "genbank")

        basic_info = {
            "file_name": os.path.basename(file_path),
            "accession": record.id,
            "name": record.name,
            "description": record.description,
            "length": len(record.seq),
            "gc_content": GC(record.seq),
            "num_features": len(record.features),
            "sequence": str(record.seq)
        }

        genes_data = []
        feature_types = Counter()

        for feature in record.features:
            feature_types[feature.type] += 1

            if feature.type == "CDS":
                locus_tag = feature.qualifiers.get("locus_tag", ["Unknown"])[0]
                gene_name = feature.qualifiers.get("gene", ["Unknown"])[0]
                product = feature.qualifiers.get("product", ["Unknown"])[0]
                protein_id = feature.qualifiers.get("protein_id", ["Unknown"])[0]
                translation = feature.qualifiers.get("translation", [""])[0]
                note = feature.qualifiers.get("note", [""])[0]
                function = feature.qualifiers.get("function", [""])[0]

                gene_info = {
                    "locus_tag": locus_tag,
                    "gene_name": gene_name,
                    "product": product,
                    "protein_id": protein_id,
                    "start": int(feature.location.start),
                    "end": int(feature.location.end),
                    "strand": "+" if feature.location.strand == 1 else "-",
                    "length": len(feature.location),
                    "protein_length": len(translation) if translation else 0,
                    "note": note,
                    "function": function,
                    "file_path": file_path
                }
                genes_data.append(gene_info)

        result = {
            "basic_info": basic_info,
            "genes_df": pd.DataFrame(genes_data) if genes_data else pd.DataFrame(),
            "list_distinct_gene_names": list(set(gene["gene_name"] for gene in genes_data)),
            "list_distinct_products": list(set(gene["product"] for gene in genes_data)),
            "num_genes": len(genes_data),
            "feature_types": feature_types
        }

        return result

    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
        return None


def process_genbank_folder(data_dir, is_train=True):
    gb_files = []
    if is_train:
        for group in os.listdir(data_dir):
            gb_files += glob.glob(os.path.join(data_dir, group, "*.gb"))
    else:
        gb_files += glob.glob(os.path.join(data_dir, "*.gb"))

    if not gb_files:
        print(f"Không tìm thấy file GenBank trong thư mục {data_dir}")
        return None

    print(f"Tìm thấy {len(gb_files)} file GenBank. Đang xử lý...")

    all_results = []
    for i, file_path in enumerate(gb_files):
        print(f"Đang xử lý file {i + 1}/{len(gb_files)}: {os.path.basename(file_path)}")
        result = analyze_genbank_file(file_path)
        if result:
            all_results.append(result)

    print(f"Đã xử lý xong {len(all_results)}/{len(gb_files)} file GenBank.")
    return all_results


def run(all_results):
    if not all_results:
        print("Không có dữ liệu để phân tích.")
        return

    basic_info_list = [r["basic_info"] for r in all_results]
    basic_df = pd.DataFrame(basic_info_list)

    all_genes_data = []
    for r in all_results:
        if not r["genes_df"].empty:
            genes_df = r["genes_df"].copy()
            genes_df["accession"] = r["basic_info"]["accession"]
            genes_df["phage_name"] = r["basic_info"]["name"]
            all_genes_data.append(genes_df)

    if all_genes_data:
        all_genes_df = pd.concat(all_genes_data, ignore_index=True)
    else:
        all_genes_df = pd.DataFrame()

    return {
        "basic_df": basic_df,
        "all_genes_df": all_genes_df
    }


def check_pattern(value, search_pattern):
    bounded_patterns = [r'\b' + re.escape(pattern) + r'\b' for pattern in search_pattern]
    pattern = re.compile('|'.join(bounded_patterns), re.IGNORECASE)
    return bool(pattern.search(str(value)))


def extract_sequence(row):
    try:
        record = SeqIO.read(row['file_path'], "genbank")
        start_pos = row['start']
        end_pos = row['end']
        seq_start = start_pos - 1
        sequence = record.seq[seq_start:end_pos]

        return str(sequence)
    except Exception as e:
        raise e


def process_lysogenic():
    all_results = process_genbank_folder(lysogenic_train_dir)

    if all_results:
        data_frames = run(all_results)

        all_gen_df = data_frames.get('all_genes_df')
    search_patterns = {
        "int": ["integrase", "int gene", "int protein", "site-specific integrase", "phage integrase",
                "tyrosine integrase", "serine integrase", "int", "intA", "intB"],
        "xis": ["excisionase", "xis gene", "xis protein", "recombination directionality factor", "rdf",
                "excision protein", "xis"],
        "rec": ["recombinase", "site-specific recombinase", "tyrosine recombinase", "serine recombinase",
                "dna recombinase", "rec", "recA"],
        "cro": ["regulatory protein cro", "cro protein", "cro repressor", "cro", "transcriptional regulator cro"],
        "q": ["antitermination protein Q", "protein Q", "Q protein", "late gene regulator Q",
              "transcription antiterminator Q"],
        "ci": ["cI repressor", "ci repressor", "cI protein", "lambda repressor", "immunity repressor",
               "repressor protein cI", "cl protein", "cl repressor"],
        "cii": ["cII protein", "cii protein", "protein cII", "transcription activator cII", "cII", "cii"],
        "ciii": ["cIII protein", "ciii protein", "protein cIII", "cIII stabilization protein", "cIII", "ciii"],
        "o": ["replication protein O", "O protein", "protein O", "replication initiation protein O",
              "phage replication protein O", "dna replication protein O"],
        "p": ["replication protein P", "P protein", "protein P", "replication initiation protein P",
              "phage replication protein P", "dna replication protein P"],
        "bet": ["recombination protein Bet", "Bet protein", "bet protein", "bet", "Bet", "BET",
                "ssDNA annealing protein", "single-strand DNA binding protein",
                "DNA single-strand annealing protein", "RecT-like recombination protein", "Redβ",
                "Redβ recombinase", "Red beta protein", "Red beta recombination protein", "Redbeta", "Red-beta",
                "single strand annealing protein", "SSAP", "Bet", "bet"]
    }

    dfs = []
    for key, patterns in search_patterns.items():
        df = all_gen_df[
            (all_gen_df['product'].apply(lambda value: check_pattern(value, patterns))) |
            (all_gen_df['gene_name'] == key)
            ]
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0).drop_duplicates(keep='first')
    combined_df['sequence'] = combined_df.apply(lambda row: extract_sequence(row), axis=1)
    combined_df['sequence_filled'] = combined_df.apply(lambda row: process_dna_sequence(row), axis=1)
    print(combined_df.shape)
    combined_df = combined_df.dropna(subset=['sequence_filled'])
    train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    train_df.to_csv("lysogenic_train.csv", index=False)
    val_df.to_csv("lysogenic_val.csv", index=False)


def process_without_extract(data_dir, is_train=False, data_class=None):
    all_results = process_genbank_folder(data_dir, is_train)
    data_frames = run(all_results)
    gen_df = data_frames.get('basic_df')

    if is_train:
        train_df, val_df = train_test_split(gen_df, test_size=0.2, random_state=42)
        train_df.to_csv(f"{data_class}_train.csv", index=False)
        val_df.to_csv(f"{data_class}_val.csv", index=False)

        return

    gen_df.to_csv(f"{data_class}_test.csv", index=False)



def process_test():
    process_without_extract(lysogenic_test_dir, False, "lysogenic")
    process_without_extract(lytic_test_dir, False, "lytic")


if __name__ == '__main__':
    # process_lysogenic()
    process_without_extract(lysogenic_train_dir, True, "lysogenic")
    process_without_extract(lytic_train_dir, True, "lytic")
    process_test()
