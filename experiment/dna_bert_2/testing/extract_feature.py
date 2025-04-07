import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# Default configuration
class Config:
    input_fasta = "../train_windowed_sequences.fasta"
    output_dir = './train_embeddings'
    model_name = 'zhihan1996/DNABERT-2-117M'
    max_length = 512
    batch_size = 512  # Increased from 8
    pooling_strategy = 'cls'
    num_workers = max(1, multiprocessing.cpu_count() - 2)  # Use all but one CPU core
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision = True  # Enable mixed precision for faster computation
    save_batch_size = 1000  # Save embeddings in chunks to reduce memory usage


# Custom dataset for DNA sequences
class DNASequenceDataset(Dataset):
    def __init__(self, ids, sequences, labels, tokenizer, max_length):
        self.ids = ids
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'sequence': self.sequences[idx],
            'sequence_length': len(self.sequences[idx]),
            'label': self.labels[idx]
        }

    def collate_fn(self, batch):
        ids = [item['id'] for item in batch]
        sequences = [item['sequence'] for item in batch]
        sequence_lengths = [item['sequence_length'] for item in batch]
        labels = [item['label'] for item in batch]

        # Tokenize sequences
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.max_length)

        return {
            'ids': ids,
            'sequence_lengths': sequence_lengths,
            'labels': labels,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'token_type_ids': inputs.get('token_type_ids', None)
        }


def read_fasta_with_labels(fasta_path):
    """Read sequences from a FASTA file using BioPython's SeqIO and assign labels based on ID content."""
    print(f"Reading sequences from {fasta_path}")
    ids = []
    sequences = []
    labels = []

    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Reading FASTA"):
        record_id = record.id
        ids.append(record_id)
        sequences.append(str(record.seq).upper())

        # Assign labels based on ID content
        if "Lytic" in record_id:
            labels.append(1)  # Lytic = 1
        elif "Lysogenic" in record_id:
            labels.append(0)  # Lysogenic = 0
        else:
            # For IDs that don't match either pattern, assign -1 (unknown)
            labels.append(-1)

    return ids, sequences, labels


def save_embeddings_chunk(ids, embeddings, labels, config, chunk_id):
    """Save a chunk of embeddings to disk."""
    output_file = os.path.join(config.output_dir, f'dnabert2_embeddings_chunk_{chunk_id}.npz')
    np.savez_compressed(
        output_file,
        ids=np.array(ids),
        embeddings=embeddings,
        labels=np.array(labels)
    )
    print(f"Saved embeddings chunk {chunk_id} to {output_file}")
    return output_file


def apply_pooling(last_hidden_state, attention_mask, pooling_strategy):
    """Apply pooling strategy to get sequence embeddings."""
    if pooling_strategy == 'cls':
        # Use the [CLS] token embedding
        return last_hidden_state[:, 0, :]
    elif pooling_strategy == 'mean':
        # Use mean of all token embeddings (masked mean)
        sum_embeddings = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
        return sum_embeddings / attention_mask.sum(dim=1, keepdim=True)
    elif pooling_strategy == 'max':
        # Use max pooling (with masking)
        masked = last_hidden_state.clone()
        masked[attention_mask.unsqueeze(-1).expand_as(masked) == 0] = -float('inf')
        return torch.max(masked, dim=1)[0]
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")


def extract_embeddings(config):
    """Extract embeddings from DNA sequences using DNA-BERT-2 with optimizations."""
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Load tokenizer and model
    print(f"Loading DNA-BERT-2 model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name, trust_remote_code=True)
    model = model.to(config.device)
    model.eval()

    # Read sequences with labels
    ids, sequences, labels = read_fasta_with_labels(config.input_fasta)
    print(f"Read {len(sequences)} sequences")
    print(f"Label distribution: {np.bincount(np.array([l for l in labels if l != -1]))}")

    # Create dataset and dataloader
    dataset = DNASequenceDataset(ids, sequences, labels, tokenizer, config.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    # Setup for mixed precision if available and requested
    amp_enabled = config.mixed_precision and config.device == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

    # Extract embeddings
    print("Extracting embeddings...")
    all_ids = []
    all_embeddings = []
    all_labels = []
    all_seq_lengths = []
    chunk_files = []
    current_chunk = 0

    # with torch.no_grad():
    #     for batch in tqdm(dataloader, desc="Processing batches"):
    #         batch_ids = batch['ids']
    #         batch_seq_lengths = batch['sequence_lengths']
    #         batch_labels = batch['labels']
    #
    #         # Move tensors to device
    #         input_ids = batch['input_ids'].to(config.device)
    #         attention_mask = batch['attention_mask'].to(config.device)
    #         token_type_ids = batch['token_type_ids'].to(config.device) if batch['token_type_ids'] is not None else None
    #
    #         # Forward pass with mixed precision if enabled
    #         if amp_enabled:
    #             with torch.cuda.amp.autocast():
    #                 outputs = model(
    #                     input_ids=input_ids,
    #                     attention_mask=attention_mask,
    #                     token_type_ids=token_type_ids,
    #                     output_hidden_states=False
    #                 )
    #         else:
    #             outputs = model(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 token_type_ids=token_type_ids,
    #                 output_hidden_states=False
    #             )
    #
    #         # Get embeddings from the last hidden state
    #         if isinstance(outputs, tuple):
    #             last_hidden_state = outputs[0]  # First element is typically the last hidden state
    #         else:
    #             last_hidden_state = outputs.last_hidden_state
    #
    #         # Apply pooling strategy
    #         embeddings = apply_pooling(last_hidden_state, attention_mask, config.pooling_strategy)
    #
    #         # Move embeddings to CPU to free up GPU memory
    #         embeddings_cpu = embeddings.cpu().numpy()
    #
    #         # Store batch results
    #         all_ids.extend(batch_ids)
    #         all_embeddings.append(embeddings_cpu)
    #         all_seq_lengths.extend(batch_seq_lengths)
    #         all_labels.extend(batch_labels)
    #
    #         # Save in chunks to manage memory
    #         if len(all_ids) >= config.save_batch_size:
    #             combined_embeddings = np.vstack(all_embeddings)
    #             chunk_file = save_embeddings_chunk(all_ids, combined_embeddings, all_labels, config, current_chunk)
    #             chunk_files.append(chunk_file)
    #
    #             # Reset for next chunk
    #             all_ids = []
    #             all_embeddings = []
    #             all_labels = []
    #             current_chunk += 1
    #
    # # Save any remaining embeddings
    # if all_embeddings:
    #     combined_embeddings = np.vstack(all_embeddings)
    #     chunk_file = save_embeddings_chunk(all_ids, combined_embeddings, all_labels, config, current_chunk)
    #     chunk_files.append(chunk_file)
    #
    # # If chunks were created, merge them
    # if chunk_files:
    #     merge_embedding_chunks(chunk_files, config)

    # Save metadata
    metadata = {
        'id': ids,
        'sequence_length': all_seq_lengths if all_seq_lengths else [len(seq) for seq in sequences],
        # 'embedding_size': combined_embeddings.shape[1] if 'combined_embeddings' in locals() else None,
        'label': labels
    }
    metadata_df = pd.DataFrame(metadata)
    metadata_file = os.path.join(config.output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_file, index=False)
    print(f"Saved metadata to {metadata_file}")

    # Save label distribution
    label_counts = np.bincount(np.array([l for l in labels if l != -1]))
    print(f"Label distribution:")
    print(f"Lysogenic (0): {label_counts[0] if len(label_counts) > 0 else 0}")
    print(f"Lytic (1): {label_counts[1] if len(label_counts) > 1 else 0}")
    print(f"Unknown (-1): {sum(1 for l in labels if l == -1)}")


# def merge_embedding_chunks(chunk_files, config):
#     """Merge embedding chunks into a single file."""
#     print("Merging embedding chunks...")
#     all_ids = []
#     all_embeddings = []
#     all_labels = []
#
#     # Load all chunks
#     for file in tqdm(chunk_files, desc="Loading chunks"):
#         data = np.load(file)
#         all_ids.extend(data['ids'])
#         all_embeddings.append(data['embeddings'])
#         all_labels.extend(data['labels'])
#
#     # Combine embeddings
#     combined_embeddings = np.vstack(all_embeddings)
#
#     # Save merged file
#     output_file = os.path.join(config.output_dir, 'dnabert2_embeddings.npz')
#     np.savez_compressed(
#         output_file,
#         ids=np.array(all_ids),
#         embeddings=combined_embeddings,
#         labels=np.array(all_labels)
#     )
#     print(f"Saved merged embeddings to {output_file}")
#
#     # Clean up chunk files
#     for file in chunk_files:
#         os.remove(file)
#         print(f"Removed temporary file: {file}")


def main():
    config = Config()
    extract_embeddings(config)


if __name__ == "__main__":
    main()