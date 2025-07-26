import json
import os

from transformers import BertTokenizer


class CodonTokenizerBuilder:
    def __init__(self):
        self.bases = ['A', 'T', 'G', 'C']
        self.codons = self._generate_all_codons()
        self.stop_codons = ['TAA', 'TAG', 'TGA']  # Stop codons in DNA

    def _generate_all_codons(self):
        """Generate all possible codons"""
        codons = []
        for b1 in self.bases:
            for b2 in self.bases:
                for b3 in self.bases:
                    codons.append(b1 + b2 + b3)
        return sorted(codons)

    def create_vocab_file(self, output_path="vocab.txt"):
        """Create vocabulary file with stop codon mapping"""
        vocab_tokens = [
            '[PAD]',
            '[UNK]',
            '[CLS]',
            '[SEP]',
            '[MASK]',
            '*'  # Stop codon symbol
        ]
        vocab_tokens.extend(self.codons)

        with open(output_path, 'w') as f:
            for token in vocab_tokens:
                f.write(token + '\n')

        print(f"Vocab file saved to {output_path}")
        print(f"Total vocab size: {len(vocab_tokens)}")
        print(f"Stop codons ({', '.join(self.stop_codons)}) will be mapped to '*'")
        return output_path

    def create_codon_to_token_mapping(self):
        """Create mapping from codon to token, with stop codons mapped to '*'"""
        mapping = {}

        # Map stop codons to '*'
        for stop_codon in self.stop_codons:
            mapping[stop_codon] = '*'

        # Map regular codons to themselves
        for codon in self.codons:
            if codon not in self.stop_codons:
                mapping[codon] = codon

        return mapping

    def encode_sequence(self, dna_sequence):
        """
        Encode DNA sequence to tokens with stop codons as '*'

        Args:
            dna_sequence (str): DNA sequence (should be multiple of 3)

        Returns:
            list: List of tokens
        """
        if len(dna_sequence) % 3 != 0:
            print(f"Warning: Sequence length {len(dna_sequence)} is not multiple of 3")

        tokens = ['[CLS]']  # Start token
        mapping = self.create_codon_to_token_mapping()

        # Process sequence in groups of 3
        for i in range(0, len(dna_sequence), 3):
            codon = dna_sequence[i:i + 3]
            if len(codon) == 3:
                if codon in mapping:
                    tokens.append(mapping[codon])
                else:
                    tokens.append('[UNK]')  # Unknown codon
            else:
                # Handle incomplete codon at the end
                tokens.append('[UNK]')

        tokens.append('[SEP]')  # End token
        return tokens

    def decode_sequence(self, tokens):
        """
        Decode tokens back to DNA sequence
        Note: Stop codons encoded as '*' cannot be perfectly decoded
        """
        dna_sequence = ""
        mapping = self.create_codon_to_token_mapping()

        # Create reverse mapping (token to codon)
        reverse_mapping = {v: k for k, v in mapping.items()}
        # For '*', we'll use the first stop codon as default
        reverse_mapping['*'] = self.stop_codons[0]  # Default to TAA

        for token in tokens:
            if token in ['[CLS]', '[SEP]', '[PAD]', '[MASK]']:
                continue
            elif token == '[UNK]':
                dna_sequence += "NNN"  # Unknown codon
            elif token in reverse_mapping:
                dna_sequence += reverse_mapping[token]
            elif token == '*':
                dna_sequence += self.stop_codons[0]  # Default stop codon

        return dna_sequence

    def create_tokenizer_config(self, output_dir="./codon_tokenizer"):
        """Create complete tokenizer configuration with stop codon support"""
        os.makedirs(output_dir, exist_ok=True)

        # Create vocab file
        vocab_path = os.path.join(output_dir, "vocab.txt")
        self.create_vocab_file(vocab_path)

        # Tokenizer config
        tokenizer_config = {
            "model_type": "bert",
            "do_lower_case": False,
            "do_basic_tokenize": False,  # Important: don't split codons
            "never_split": None,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "tokenize_chinese_chars": False,
            "strip_accents": None,
            "model_max_length": 512,
            "special_tokens_map_file": None,
            "name_or_path": "codon_bert_tokenizer",
            "tokenizer_class": "BertTokenizer"
        }

        config_path = os.path.join(output_dir, "tokenizer_config.json")
        with open(config_path, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)

        # Special tokens map
        special_tokens_map = {
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]"
        }

        special_tokens_path = os.path.join(output_dir, "special_tokens_map.json")
        with open(special_tokens_path, 'w') as f:
            json.dump(special_tokens_map, f, indent=2)

        # Save codon mapping for reference
        mapping_path = os.path.join(output_dir, "codon_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(self.create_codon_to_token_mapping(), f, indent=2)

        print(f"Tokenizer config saved to {output_dir}")
        print(f"Codon mapping saved to {mapping_path}")
        return output_dir

    def load_tokenizer(self, tokenizer_dir):
        """Load tokenizer from directory"""
        return BertTokenizer.from_pretrained(tokenizer_dir)

    def demo_encoding(self):
        """Demonstrate encoding with stop codons"""
        # Example DNA sequence with stop codons
        test_sequence = "ATGAAATAAGTGCCCTAGTGA"  # Start codon + some codons + stop codons

        print("=== Demo: Encoding DNA sequence with stop codons ===")
        print(f"Original DNA: {test_sequence}")

        # Split into codons for visualization
        codons = [test_sequence[i:i + 3] for i in range(0, len(test_sequence), 3)]
        print(f"Codons: {' | '.join(codons)}")

        # Encode
        tokens = self.encode_sequence(test_sequence)
        print(f"Encoded tokens: {tokens}")

        # Show mapping
        mapping = self.create_codon_to_token_mapping()
        print("\nCodon -> Token mapping:")
        for codon in codons:
            if len(codon) == 3:
                print(f"  {codon} -> {mapping.get(codon, '[UNK]')}")

        # Decode back
        decoded = self.decode_sequence(tokens)
        print(f"\nDecoded DNA: {decoded}")

        return tokens


# Example usage
if __name__ == "__main__":
    # Create tokenizer builder
    builder = CodonTokenizerBuilder()

    # Demo the encoding with stop codons
    builder.demo_encoding()

    # Create tokenizer files
    tokenizer_dir = builder.create_tokenizer_config("./codon_tokenizer")

    # Load and test tokenizer
    tokenizer = builder.load_tokenizer(tokenizer_dir)

    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    print("Stop codon symbol '*' is in vocab:", '*' in tokenizer.vocab)