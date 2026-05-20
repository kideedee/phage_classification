import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


def debug_dnabert_zero_outputs():
    """Debug why DNABERT-2 outputs are all zeros"""

    # Load model and tokenizer
    print("Loading DNABERT-2...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # Test with a simple DNA sequence
    test_sequence = "ATCGATCGATCGATCG"
    print(f"\nTesting with sequence: {test_sequence}")

    # Check tokenization
    print("\n1. Checking tokenization...")
    tokens = tokenizer.tokenize(test_sequence)
    print(f"Tokens: {tokens}")

    token_ids = tokenizer.encode(test_sequence)
    print(f"Token IDs: {token_ids}")

    # Check input preparation
    print("\n2. Checking input preparation...")
    inputs = tokenizer(
        test_sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    print(f"Input keys: {inputs.keys()}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Attention mask: {inputs['attention_mask']}")

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Check model forward pass
    print("\n3. Checking model forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Output type: {type(outputs)}")
    print(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")

    # Check last hidden states
    if hasattr(outputs, 'last_hidden_state'):
        last_hidden_states = outputs.last_hidden_state
        print(f"Last hidden states shape: {last_hidden_states.shape}")
        print(f"Last hidden states dtype: {last_hidden_states.dtype}")
        print(f"Last hidden states device: {last_hidden_states.device}")
        print(f"Min value: {last_hidden_states.min().item()}")
        print(f"Max value: {last_hidden_states.max().item()}")
        print(f"Mean value: {last_hidden_states.mean().item()}")
        print(f"Std value: {last_hidden_states.std().item()}")

        # Check if all zeros
        is_all_zeros = torch.all(last_hidden_states == 0)
        print(f"All zeros: {is_all_zeros}")

        # Print first few values
        print(f"First 10 values of [CLS] token: {last_hidden_states[0, 0, :10]}")

    # Test with different sequence
    print("\n4. Testing with longer sequence...")
    long_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    inputs_long = tokenizer(
        long_sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs_long = {k: v.to(device) for k, v in inputs_long.items()}

    with torch.no_grad():
        outputs_long = model(**inputs_long)

    if hasattr(outputs_long, 'last_hidden_state'):
        last_hidden_states_long = outputs_long.last_hidden_state
        print(f"Long sequence - Min: {last_hidden_states_long.min().item()}")
        print(f"Long sequence - Max: {last_hidden_states_long.max().item()}")
        print(f"Long sequence - Mean: {last_hidden_states_long.mean().item()}")

    # Check model weights
    print("\n5. Checking model weights...")
    first_layer_weight = None
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}, Mean: {param.mean().item():.6f}")
        if first_layer_weight is None:
            first_layer_weight = param
        break  # Just check first parameter

    print(f"First layer weight stats - Min: {first_layer_weight.min().item():.6f}")
    print(f"First layer weight stats - Max: {first_layer_weight.max().item():.6f}")


# Additional debugging functions
def check_sequence_format(sequence):
    """Check if DNA sequence is properly formatted for DNABERT-2"""
    valid_chars = set('ATCGN')
    sequence_chars = set(sequence.upper())
    invalid_chars = sequence_chars - valid_chars

    print(f"Sequence: {sequence}")
    print(f"Valid characters: {valid_chars}")
    print(f"Sequence characters: {sequence_chars}")
    print(f"Invalid characters: {invalid_chars}")

    return len(invalid_chars) == 0


def test_with_multiple_sequences():
    """Test with various sequence types"""
    sequences = [
        "'AATGACAAGCAGAAAAATTATAATCCACACAAGCCATACCCCCTAAGGTGTTATTACCGTCCTGAAAGTCGATTCACTTGTATTGTTTAAACCTTCATACATTGTACTTGTCTTGTATGGAGTTGGAAATAGGACAATTTGATAGTTTTCTTTATCCT'",
        "ATCGATCGATCGATCG",
        # "AAAAAAAAAAAAAAAAA",
        # "NNNNNNNNNNNNNNN",
        # "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" * 5
    ]

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    for i, seq in enumerate(sequences):
        print(f"\nTesting sequence {i + 1}: {seq[:50]}{'...' if len(seq) > 50 else ''}")
        check_sequence_format(seq)

        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            print(f"Output shape: {hidden.shape}")
            print(f"Min: {hidden.min().item():.6f}, Max: {hidden.max().item():.6f}, Mean: {hidden.mean().item():.6f}")


if __name__ == "__main__":
    # debug_dnabert_zero_outputs()
    print("\n" + "=" * 50)
    test_with_multiple_sequences()