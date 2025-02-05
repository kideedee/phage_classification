import os
import h5py
import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Generator
from dataclasses import dataclass
from common.env_config import config
from log.custom_log import logger


@dataclass
class SequenceBatch:
    ids: List[str]
    descriptions: List[str]
    sequences: List[str]


class DNABertEmbedding:
    def __init__(self, model_name: str = "zhihan1996/DNA_bert_6", batch_size: int = 32):
        logger.info(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if self.device == 'cuda':
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable cuDNN benchmarking for optimal performance
            torch.backends.cudnn.benchmark = True

            # Initialize gradient scaler for mixed precision
            # self.scaler = torch.cuda.amp.GradScaler()

            # Set memory allocation strategy
            # torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory
            torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.embedding_dim = self.model.config.hidden_size

    def _split_sequence(self, sequence: str, kmer: int = 6) -> List[str]:
        return [sequence[i:i + kmer] for i in range(0, len(sequence) - kmer + 1)]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @torch.cuda.amp.autocast()
    def embed_batch(self, sequences: List[str]) -> np.ndarray:
        try:
            kmer_lists = [self._split_sequence(str(seq)) for seq in sequences]
            kmer_strs = [" ".join(kmers) for kmers in kmer_lists]

            inputs = self.tokenizer(kmer_strs, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            torch.cuda.empty_cache()
            return embeddings

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

    def sequence_generator(self, fna_path: str) -> Generator[SequenceBatch, None, None]:
        batch_ids, batch_desc, batch_seq = [], [], []

        for record in SeqIO.parse(fna_path, "fasta"):
            batch_ids.append(record.id)
            batch_desc.append(record.description)
            batch_seq.append(str(record.seq))

            if len(batch_ids) == self.batch_size:
                yield SequenceBatch(batch_ids, batch_desc, batch_seq)
                batch_ids, batch_desc, batch_seq = [], [], []

        if batch_ids:  # Process remaining sequences
            yield SequenceBatch(batch_ids, batch_desc, batch_seq)

    def process_and_save(self, fna_path: str, output_path: str):
        logger.info(f"Processing {fna_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Count sequences first
        with open(fna_path) as f:
            total_sequences = sum(1 for _ in SeqIO.parse(f, "fasta"))

        with h5py.File(output_path, 'w') as f:
            # Initialize datasets with compression
            embeddings_dataset = f.create_dataset(
                'embeddings',
                shape=(total_sequences, self.embedding_dim),
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
                chunks=True
            )

            dt = h5py.special_dtype(vlen=str)
            seq_ids_dataset = f.create_dataset('sequence_ids', shape=(total_sequences,), dtype=dt)
            desc_dataset = f.create_dataset('descriptions', shape=(total_sequences,), dtype=dt)
            seq_dataset = f.create_dataset('sequences', shape=(total_sequences,), dtype=dt)

            f.attrs['embedding_dim'] = self.embedding_dim
            f.attrs['num_sequences'] = total_sequences

            current_idx = 0
            for batch in tqdm(self.sequence_generator(fna_path),
                              total=(total_sequences + self.batch_size - 1) // self.batch_size,
                              desc="Processing batches"):
                batch_size = len(batch.ids)
                embeddings = self.embed_batch(batch.sequences)

                # Save batch data
                embeddings_dataset[current_idx:current_idx + batch_size] = embeddings
                seq_ids_dataset[current_idx:current_idx + batch_size] = batch.ids
                desc_dataset[current_idx:current_idx + batch_size] = batch.descriptions
                seq_dataset[current_idx:current_idx + batch_size] = batch.sequences

                current_idx += batch_size

        logger.info(f"Saved embeddings to {output_path}")
        return output_path


def process_folder(folder_path: str, result_dir: str, dna_embedder: DNABertEmbedding):
    try:
        for file in os.listdir(folder_path):
            if file.endswith('.fna'):
                fna_path = os.path.join(folder_path, file)
                output_path = os.path.join(
                    result_dir,
                    os.path.basename(folder_path),
                    file.replace(".fna", ".h5")
                )

                output_path = dna_embedder.process_and_save(fna_path, output_path)

                with h5py.File(output_path, 'r') as f:
                    logger.info(f"Processed {file}")
                    logger.info(f"Number of sequences: {f.attrs['num_sequences']}")
                    logger.info(f"Embeddings shape: {f['embeddings'].shape}")

    except Exception as e:
        logger.error(f"Error processing folder {folder_path}: {str(e)}")
        raise


def main():
    logger.info("Initializing DNABertEmbedding...")
    try:
        data_type = "train"
        data_dir = f"{config.DNA_BERT_INPUT_DATA_PATH}/{data_type}"
        result_dir = f"{config.DNA_BERT_OUTPUT_DATA_PATH}/{data_type}"

        dna_embedder = DNABertEmbedding(
            model_name=config.DNA_BERT_MODEL_PATH,
            batch_size=32  # Adjust based on GPU memory
        )

        with ThreadPoolExecutor() as executor:
            folder_paths = [
                os.path.join(data_dir, folder)
                for folder in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, folder))
            ]

            futures = [
                executor.submit(process_folder, folder_path, result_dir, dna_embedder)
                for folder_path in folder_paths
            ]

            for future in futures:
                future.result()  # Will raise any exceptions that occurred

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()