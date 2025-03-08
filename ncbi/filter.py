from Bio import SeqIO

# Đọc file GenBank
virulent_phages = []
temperate_phages = []
uncertain_phages = []

for record in SeqIO.parse("phage_genomes.fasta", "fasta"):
    phage_id = record.id
    description = record.description.lower()
    features = record.features

    lysogenic_genes = ["integrase", "excision", "lysogen", "temperate",
                       "cI repressor", "recombinase"]

    found_lysogenic = False

    for keyword in lysogenic_genes:
        if keyword in description:
            temperate_phages.append(phage_id)
            found_lysogenic = True
            break

    if found_lysogenic:
        continue

    for feature in features:
        if feature.type == "CDS" and 'product' in feature.qualifiers:
            product = feature.qualifiers['product'][0].lower()
            for keyword in lysogenic_genes:
                if keyword in product:
                    temperate_phages.append(phage_id)
                    found_lysogenic = True
                    break
            if found_lysogenic:
                break

    if not found_lysogenic:
        if "virulent" in description or "lytic" in description:
            virulent_phages.append(phage_id)
        else:
            uncertain_phages.append(phage_id)

print(f"Phage virulent: {len(virulent_phages)}")
print(f"Phage temperate: {len(temperate_phages)}")
print(f"Phage không xác định: {len(uncertain_phages)}")
