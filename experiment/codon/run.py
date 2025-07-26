import torch
from transformers import BertConfig, BertForSequenceClassification

from experiment.codon.codon_processor import CodonProcessor
from experiment.codon.codon_tokenizer import CodonTokenizerBuilder

if __name__ == "__main__":
    # 1. Create tokenizer
    builder = CodonTokenizerBuilder()
    tokenizer_dir = builder.create_tokenizer_config("./codon_tokenizer")

    # 2. Load tokenizer
    tokenizer = builder.load_tokenizer(tokenizer_dir)
    print(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")

    # 3. Create processor
    processor = CodonProcessor(tokenizer)

    # 4. Test with DNA sequence
    test_dna = "ATGGATCCATAAGGCTGAATCGATCGATCG"
    encoded, codon_text = processor.tokenize_dna(test_dna)

    print(f"Original DNA: {test_dna}")
    print(f"Codon text: {codon_text}")
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"First 10 tokens: {encoded['input_ids'][0][:10]}")

    # 5. Decode to verify
    decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
    print(f"Decoded: {decoded}")

    # 6. Create compatible BERT model
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = BertForSequenceClassification(config)

    # 7. Test forward pass
    with torch.no_grad():
        output = model(**encoded)
        print(f"Model output shape: {output.logits.shape}")
        print(f"Predictions: {torch.softmax(output.logits, dim=-1)}")

    # 8. Save tokenizer for future use
    tokenizer.save_pretrained("./final_codon_tokenizer")
    print("Tokenizer saved to ./final_codon_tokenizer")
