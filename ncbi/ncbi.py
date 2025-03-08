from Bio import Entrez, SeqIO

Entrez.email = "quangsonvu9699@gmail.com"

search_handle = Entrez.esearch(db="nucleotide",
                              term="bacteriophage[ORGN] AND complete genome[TITL]",
                              retmax=5000)
search_results = Entrez.read(search_handle)
search_handle.close()

id_list = search_results["IdList"]
print(f"Tìm thấy {len(id_list)} trình tự phage")

# Tải xuống (chia thành nhiều lô nếu có nhiều trình tự)
batch_size = 100
output_file = "phage_genomes.fasta"

with open(output_file, "w") as output_handle:
    for start in range(0, len(id_list), batch_size):
        end = min(start + batch_size, len(id_list))
        print(f"Đang tải các trình tự {start + 1}-{end}...")

        fetch_handle = Entrez.efetch(db="nucleotide",
                                     id=id_list[start:end],
                                     rettype="fasta",
                                     retmode="text")

        for seq_record in SeqIO.parse(fetch_handle, "fasta"):
            SeqIO.write(seq_record, output_handle, "fasta")

        fetch_handle.close()

print(f"Đã tải xong và lưu vào {output_file}")