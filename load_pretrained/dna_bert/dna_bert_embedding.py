import os

import h5py
import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from common.env_config import config
from log.custom_log import logger

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


class DNABertEmbedding:
    def __init__(self, model_name="zhihan1996/DNA_bert_6", batch_size=32):
        logger.info(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        if torch.cuda.is_available():
            # Clear GPU cache before loading model
            torch.cuda.empty_cache()

            # Set memory allocation strategy
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory

            # Enable TF32 for better performance on Ampere GPUs (RTX 3000 series)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Disable benchmarking to reduce memory usage
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Reduce batch size if using GPU
        self.batch_size = batch_size // 2 if torch.cuda.is_available() else batch_size

    def _split_sequence(self, sequence, kmer=6):
        return [sequence[i:i + kmer] for i in range(0, len(sequence) - kmer + 1)]

    def embed_batch(self, sequences):
        try:
            kmer_lists = [self._split_sequence(str(seq)) for seq in sequences]
            kmer_strs = [" ".join(kmers) for kmers in kmer_lists]

            inputs = self.tokenizer(kmer_strs, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Add gradient checkpointing for memory efficiency
            self.model.gradient_checkpointing_enable()

            with torch.cuda.amp.autocast():  # Enable automatic mixed precision
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Clear GPU cache after processing each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return embeddings

        except RuntimeError as e:
            logger.error(f"GPU error in embed_batch: {str(e)}")
            # If we encounter a CUDA error, try processing with an even smaller batch
            if len(sequences) > 1:
                mid = len(sequences) // 2
                first_half = self.embed_batch(sequences[:mid])
                second_half = self.embed_batch(sequences[mid:])
                return np.vstack([first_half, second_half])
            else:
                raise e

    def process_and_save(self, fna_path, output_path):
        """
        Process .fna file and save embeddings to .h5 file using batched processing
        Args:
            fna_path: Path to input .fna file
            output_path: Path to output .h5 file
        """
        logger.info(f"Processing {fna_path} and saving to {output_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sequences = list(SeqIO.parse(fna_path, "fasta"))
        total_sequences = len(sequences)
        embeddings = []
        seq_ids = []
        descriptions = []
        sequences_str = []

        # Process in batches
        for i in tqdm(range(0, total_sequences, self.batch_size), desc="Processing batches"):
            batch_sequences = sequences[i:i + self.batch_size]
            batch_embeddings = self.embed_batch([seq.seq for seq in batch_sequences])

            embeddings.extend(batch_embeddings)
            seq_ids.extend([seq.id for seq in batch_sequences])
            descriptions.extend([seq.description for seq in batch_sequences])
            sequences_str.extend([str(seq.seq) for seq in batch_sequences])

        embeddings = np.array(embeddings)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('embeddings', data=embeddings)
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('sequence_ids', data=seq_ids, dtype=dt)
            f.create_dataset('descriptions', data=descriptions, dtype=dt)
            f.create_dataset('sequences', data=sequences_str, dtype=dt)
            f.attrs['embedding_dim'] = self.get_embedding_dim()
            f.attrs['num_sequences'] = len(sequences)

    def get_embedding_dim(self):
        return self.model.config.hidden_size


def main():
    logger.info("Initializing DNABertEmbedding...")
    try:
        data_type = "test"
        data_dir = f"{config.DNA_BERT_INPUT_DATA_PATH}/{data_type}"
        result_dir = f"{config.DNA_BERT_OUTPUT_DATA_PATH}/{data_type}"

        # Initialize embedder once outside the loop
        dna_embedder = DNABertEmbedding(model_name=config.DNA_BERT_MODEL_PATH)

        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if os.path.exists(os.path.join(result_dir, folder, file.replace(".fna", ".h5"))) or not file.endswith(".fna"):
                        logger.info(f"Skipping {file}")
                        continue

                    fna_path = os.path.join(folder_path, file)
                    output_path = os.path.join(result_dir, folder, file.replace(".fna", ".h5"))

                    dna_embedder.process_and_save(fna_path, output_path)

                    with h5py.File(output_path, 'r') as f:
                        logger.info(f"Processed {file}")
                        logger.info(f"Number of sequences: {f.attrs['num_sequences']}")
                        logger.info(f"Embeddings shape: {f['embeddings'].shape}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
