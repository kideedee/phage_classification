from Bio import SeqIO

if __name__ == '__main__':
    # Path to your GenBank file
    genbank_file = "../data/gen_bank/AF074945_Lysogenic.gb"  # or .gbk or .genbank

    # Read the GenBank file
    record = SeqIO.read(genbank_file, "genbank")

    # Now you can access various parts of the record
    print(f"ID: {record.id}")
    print(f"Name: {record.name}")
    print(f"Description: {record.description}")
    print(f"Sequence length: {len(record.seq)}")
    print(f"Sequence: {record.seq}")

    # Access features (genes, CDS, etc.)
    # for feature in record.features:
    #     if feature.type == "gene":
    #         print(f"Gene: {feature.qualifiers.get('gene', ['Unknown'])[0]}")
    #         print(f"Location: {feature.location}")
