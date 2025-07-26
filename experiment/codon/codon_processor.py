class CodonProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def dna_to_codons(self, dna_sequence):
        """Convert DNA sequence to space-separated codons"""
        # Clean sequence
        dna_sequence = dna_sequence.upper().replace(' ', '').replace('\n', '')

        codons = []
        for i in range(0, len(dna_sequence) - 2, 3):
            codon = dna_sequence[i:i + 3]
            if len(codon) == 3 and all(base in 'ATGC' for base in codon):
                codons.append(codon)
        return ' '.join(codons)

    def tokenize_dna(self, dna_sequence, max_length=512):
        """Tokenize DNA sequence"""
        codon_text = self.dna_to_codons(dna_sequence)

        encoded = self.tokenizer(
            codon_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return encoded, codon_text

    def batch_tokenize(self, dna_sequences, max_length=512):
        """Batch tokenize multiple DNA sequences"""
        codon_texts = [self.dna_to_codons(seq) for seq in dna_sequences]

        encoded = self.tokenizer(
            codon_texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return encoded, codon_texts