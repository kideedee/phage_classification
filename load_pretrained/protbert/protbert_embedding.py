import json
import logging
import os
import time
from datetime import datetime
from logging.handlers import MemoryHandler
from pathlib import Path
from subprocess import check_output
from typing import Dict, List, Union, Tuple, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '0'


class CheckpointHandler:
    def __init__(self, output_file: str, checkpoint_interval: int = 1000):
        self.output_file = output_file
        self.checkpoint_file = output_file + '.checkpoint'
        self.checkpoint_interval = checkpoint_interval
        self.logger = logging.getLogger(__name__)

    def load_checkpoint(self) -> Tuple[int, Optional[Dict]]:
        """
        Load existing checkpoint data
        Returns:
            Tuple containing:
            - Number of processed sequences
            - Dictionary with existing embeddings and sequence IDs if available
        """
        processed_sequences = 0
        existing_data = None

        if os.path.exists(self.checkpoint_file):
            try:
                with h5py.File(self.checkpoint_file, 'r') as f:
                    processed_sequences = f.attrs.get('processed_sequences', 0)
                    if 'embeddings' in f and 'sequence_ids' in f:
                        existing_data = {
                            'embeddings': f['embeddings'][:],
                            'sequence_ids': [s.decode() for s in f['sequence_ids'][:]]
                        }
                self.logger.info(f"Loaded checkpoint: {processed_sequences} sequences processed")
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {str(e)}")

        return processed_sequences, existing_data

    def save_checkpoint(self, processed_sequences: int,
                        embeddings: torch.Tensor,
                        sequence_ids: List[str]):
        """
        Save checkpoint data periodically
        """
        if processed_sequences % self.checkpoint_interval == 0:
            try:
                with h5py.File(self.checkpoint_file, 'w') as f:
                    f.attrs['processed_sequences'] = processed_sequences
                    f.create_dataset('embeddings', data=embeddings.cpu().numpy())
                    dt = h5py.special_dtype(vlen=str)
                    sequence_ids_dataset = f.create_dataset('sequence_ids',
                                                            (len(sequence_ids),), dtype=dt)
                    sequence_ids_dataset[:] = sequence_ids
                self.logger.info(f"Saved checkpoint at {processed_sequences} sequences")
            except Exception as e:
                self.logger.error(f"Error saving checkpoint: {str(e)}")

    def cleanup(self):
        """
        Clean up checkpoint file after successful completion
        """
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                self.logger.info("Checkpoint file cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up checkpoint: {str(e)}")


# Tạo thư mục logs nếu chưa tồn tại
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
monitor_file = log_dir / f"gpu_monitor_{current_time}.json"

# Logger cho thông tin chung
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console handler
    ]
)
logger = logging.getLogger(__name__)

# Logger riêng cho monitoring
monitor_logger = logging.getLogger('monitor')
monitor_logger.setLevel(logging.INFO)

# File handler
monitor_file_handler = logging.FileHandler(log_dir / f"monitor_{current_time}.log", mode='w')
monitor_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# Memory handler với buffer 1000 records
memory_handler = MemoryHandler(
    capacity=1000,  # Số lượng records giữ trong memory
    flushLevel=logging.ERROR,  # Flush ngay khi có ERROR
    target=monitor_file_handler  # Handler đích để ghi log
)

monitor_logger.addHandler(monitor_file_handler)
monitor_logger.propagate = False  # Không chuyển log lên logger cha


def save_monitor_data(data: Dict):
    """
    Lưu monitor data vào file JSON
    """
    try:
        if os.path.exists(monitor_file):
            with open(monitor_file, 'r') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        else:
            existing_data = []

        existing_data.append(data)

        with open(monitor_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving monitor data: {str(e)}")


def monitor_gpu_memory() -> Tuple[float, float, float]:
    """
    Monitor detailed GPU memory usage using PyTorch
    """
    if not torch.cuda.is_available():
        return None

    try:
        # Allocated memory
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB
        cached = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # Convert to GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        memory_stats = torch.cuda.memory_stats()
        active_blocks = memory_stats.get("num_alloc_retries", 0)

        memory_info = {
            "timestamp": datetime.now().isoformat(),
            "allocated_gb": round(allocated, 2),
            "cached_gb": round(cached, 2),
            "max_allocated_gb": round(max_allocated, 2),
            "active_blocks": active_blocks
        }

        # Log to monitor file only
        monitor_logger.info(
            f"GPU Memory - Allocated: {allocated:.2f}GB, "
            f"Cached: {cached:.2f}GB, "
            f"Max: {max_allocated:.2f}GB, "
            f"Blocks: {active_blocks}"
        )

        save_monitor_data({"memory": memory_info})
        return allocated, cached, max_allocated

    except Exception as e:
        logger.error(f"Error monitoring GPU memory: {str(e)}")
        return None


def monitor_system() -> float:
    """
    Monitor system and GPU status
    """
    try:
        gpu_info = check_output([
            'nvidia-smi',
            '--query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ])
        temp, util, power, mem_used, mem_total = gpu_info.decode().strip().split(',')

        # Convert values to appropriate types
        temp = float(temp)
        util = float(util)
        power = float(power)
        mem_used = float(mem_used)
        mem_total = float(mem_total)

        system_info = {
            "timestamp": datetime.now().isoformat(),
            "temperature": temp,
            "utilization": util,
            "power_draw": power,
            "memory_used_mb": mem_used,
            "memory_total_mb": mem_total
        }

        # Log to monitor file only
        monitor_logger.info(
            f"GPU Status - Temp: {temp}°C, "
            f"Util: {util}%, "
            f"Power: {power}W, "
            f"Memory: {mem_used}/{mem_total}MB"
        )

        # Add PyTorch memory monitoring
        memory_info = monitor_gpu_memory()

        # Save combined monitoring data
        save_monitor_data({
            "system": system_info,
            "timestamp": datetime.now().isoformat()
        })

        return temp

    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")
        return None


class ProteinEmbedder:
    def __init__(
            self,
            model_name: str = "Rostlab/prot_bert",
            device: str = None,
            batch_size: int = 32,
            max_length: int = 512
    ):
        """
        Khởi tạo Protein Embedder sử dụng ProtBERT
        """
        logger.info(
            f"Initializing ProteinEmbedder with: model={model_name}, batch_size={batch_size}, max_length={max_length}")

        # Xác định device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Memory Management Optimization
        if self.device == 'cuda':
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable cuDNN benchmarking for optimal performance
            torch.backends.cudnn.benchmark = True

            # Initialize gradient scaler for mixed precision
            self.scaler = torch.cuda.amp.GradScaler()

            # Set memory allocation strategy
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory
            torch.cuda.empty_cache()

        # Load model và tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        self.batch_size = batch_size
        self.max_length = max_length
        self.checkpoint_interval = 1000

    def _tokenize_sequences(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize một batch các protein sequences
        """
        sequences = [" ".join(list(seq)) for seq in sequences]
        return self.tokenizer(
            sequences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def get_embeddings(
            self,
            sequences: List[str],
            pooling_mode: str = 'mean',
            return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Tạo embeddings cho một list các protein sequences
        """
        all_embeddings = []
        monitor_interval = 10  # Monitor mỗi 10 batches

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), self.batch_size), desc="Creating embeddings"):
                # Monitor trước mỗi batch
                temp = monitor_system()

                # Check temperature threshold
                if temp and temp > 80:
                    logger.warning(f"High temperature detected: {temp}°C. Pausing for cooling...")
                    time.sleep(30)

                batch_sequences = sequences[i:i + self.batch_size]
                inputs = self._tokenize_sequences(batch_sequences)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Monitor sau khi tokenize
                # monitor_system()

                outputs = self.model(**inputs)

                # Monitor sau khi get embeddings
                # monitor_system()

                if pooling_mode == 'cls':
                    batch_embeddings = outputs.last_hidden_state[:, 0]
                elif pooling_mode == 'mean':
                    attention_mask = inputs['attention_mask']
                    batch_embeddings = torch.sum(
                        outputs.last_hidden_state * attention_mask.unsqueeze(-1),
                        dim=1
                    ) / torch.sum(attention_mask, dim=1, keepdim=True)
                elif pooling_mode == 'all':
                    batch_embeddings = outputs.last_hidden_state
                else:
                    raise ValueError("pooling_mode must be one of ['mean', 'cls', 'all']")

                batch_embeddings = batch_embeddings.cpu()
                torch.cuda.empty_cache()
                all_embeddings.append(batch_embeddings)

                # if i % 2 == 0:
                #     time.sleep(0.1)
                # del outputs
                # del inputs
                # torch.cuda.empty_cache()

        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings.numpy() if return_numpy else embeddings

    def embed_fasta(
            self,
            input_file: str,
            output_file: str,
            pooling_mode: str = 'mean'
    ) -> Dict[str, int]:
        """
        Tạo embeddings cho các sequences từ file FASTA
        """
        from Bio import SeqIO

        checkpoint_handler = CheckpointHandler(output_file, self.checkpoint_interval)
        processed_sequences, existing_data = checkpoint_handler.load_checkpoint()

        # Skip already processed sequences
        sequences = []
        sequence_ids = []

        for i, record in enumerate(SeqIO.parse(input_file, "fasta")):
            if i < processed_sequences:
                continue
            sequences.append(str(record.seq))
            sequence_ids.append(record.id)

        if sequences:  # If there are new sequences to process
            new_embeddings = self.get_embeddings(sequences, pooling_mode=pooling_mode, return_numpy=False)

            if existing_data:  # Combine with existing embeddings
                embeddings = torch.cat([
                    torch.tensor(existing_data['embeddings']),
                    new_embeddings
                ], dim=0)
                sequence_ids = existing_data['sequence_ids'] + sequence_ids
            else:
                embeddings = new_embeddings

            # Save checkpoint periodically
            checkpoint_handler.save_checkpoint(len(sequence_ids), embeddings, sequence_ids)

            # Save final results
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('embeddings', data=embeddings.numpy())
                dt = h5py.special_dtype(vlen=str)
                sequence_ids_dataset = f.create_dataset('sequence_ids',
                                                        (len(sequence_ids),), dtype=dt)
                sequence_ids_dataset[:] = sequence_ids
                f.attrs['pooling_mode'] = pooling_mode
                f.attrs['model_name'] = "Rostlab/prot_bert"
                f.attrs['embedding_dim'] = embeddings.shape[1]

            checkpoint_handler.cleanup()  # Remove checkpoint after successful completion

            stats = {
                'total_sequences': len(sequence_ids),
                'embedding_dim': embeddings.shape[1],
                'file_size_mb': Path(output_file).stat().st_size / (1024 * 1024)
            }

            logger.info(f"Embedding stats: {stats}")
            return stats
        else:
            logger.info("All sequences already processed")
            return {
                'total_sequences': processed_sequences,
                'embedding_dim': existing_data['embeddings'].shape[1] if existing_data else None,
                'file_size_mb': Path(output_file).stat().st_size / (1024 * 1024) if os.path.exists(output_file) else 0
            }


if __name__ == "__main__":
    logger.info("Starting protein embedding process...")
    mode = "fill"
    data_type = 'train'
    try:
        data_dir = f"../../data/my_data/protein_format/{mode}/{data_type}"
        result_dir = f"../../data/my_data/protbert_embedding/{mode}/{data_type}"
        for folder in os.listdir(os.path.join(data_dir)):
            if not os.path.exists(os.path.join(result_dir, folder)):
                os.makedirs(os.path.join(result_dir, folder), exist_ok=True)

            if folder == "100_400":
                batch_size = 64
                length = 256
            elif folder == "400_800":
                batch_size = 64
                length = 512
                # continue
            else:
                batch_size = 16

                length = 1024
                # continue

            for file in os.listdir(os.path.join(data_dir, folder)):
                input_fasta = os.path.join(data_dir, folder, file)
                output_h5 = os.path.join(result_dir, folder, file.replace(".fasta", ".h5"))
                if os.path.exists(output_h5):
                    logger.info(f"File {file} is embedded")
                    continue

                logger.info(f"Processing file: {file}, length: {length}")
                embedder = ProteinEmbedder(
                    model_name="prot_bert",
                    batch_size=batch_size,
                    max_length=length
                )

                # os.makedirs(os.path.dirname(output_h5), exist_ok=True)
                stats = embedder.embed_fasta(
                    input_file=input_fasta,
                    output_file=output_h5,
                    pooling_mode='mean'
                )
                torch.cuda.empty_cache()
                logger.info(f"Processing stats: {stats}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
