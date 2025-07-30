# """Create and FCGR.npy for each sequence.fasta in a folder"""
# from pathlib import Path
#
# from fcgr_embedding import FCGREmbedding
#
# if __name__ == '__main__':
#     DATA_DIR = "E:\master\\final_project\data\my_data\\fasta\\1200_1800\\1\\test\data.fa"
#     KMER = 3
#     OUTPUT_DIR = Path("E:\master\\final_project\data\my_data\output\\fcgr")
#
#     # Instantiate class to generate FCGR
#     generate_fcgr = FCGREmbedding(
#         destination_folder=OUTPUT_DIR,
#         kmer=KMER,
#     )
#
#     # Generate FCGR for a list of fasta files
#     generate_fcgr(list_fasta=DATA_DIR)
#
#     # count generated FCGR
#     N_seqs_to_gen = len(DATA_DIR)
#     with open(OUTPUT_DIR.joinpath("generated_fcgr.txt"), "w") as fp:
#         gen_seqs = len(list(OUTPUT_DIR.rglob("*.npy")))
#         fp.write(f"{gen_seqs}/{N_seqs_to_gen}")
