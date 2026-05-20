import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from embedding_sequence.abstract_embedding import AbstractEmbedding


class DNABert2Embedding(AbstractEmbedding):
    def __init__(self, data_dir, output_dir, min_size, max_size, overlap_percent, is_train, fold):
        super().__init__(
            embedding_type="dnabert_2",
            data_dir=data_dir,
            output_dir=output_dir,
            min_size=min_size,
            max_size=max_size,
            overlap_percent=overlap_percent,
            is_train=is_train,
            fold=fold)
        # config = AutoConfig.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            trust_remote_code=True,
            # config=config
        )
        self.mode = "nothing"
        self.feature_type = 'cls'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.model = self.model.to(self.device)  # Move model to GPU
        self.model.eval()

        if max_size == 1800:
            self.max_length = 512
        else:
            self.max_length = max_size // 4

        if max_size == 400:
            self.batch_size = 128
        elif max_size == 800:
            self.batch_size = 64
        elif max_size == 1200:
            self.batch_size = 32
        else:
            self.batch_size = 16

    def run(self, sequences: np.array, labels: np.array):
        if self.mode == "nothing":
            sequences = sequences.tolist()

            all_features = []
            all_labels = []

            # Calculate total number of batches for progress bar
            total_batches = (len(sequences) + self.batch_size - 1) // self.batch_size

            for i in tqdm(range(0, len(sequences), self.batch_size), desc="Processing batches", total=total_batches):
                batch_sequences = sequences[i:i + self.batch_size]
                batch_labels = labels[i:i + self.batch_size]

                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True,
                                        max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)

                    # Handle different output formats
                    if hasattr(outputs, 'last_hidden_state'):
                        last_hidden_states = outputs.last_hidden_state
                    elif isinstance(outputs, tuple):
                        last_hidden_states = outputs[0]  # Usually the first element
                    else:
                        last_hidden_states = outputs['last_hidden_state']

                    # Extract features based on feature type
                    if self.feature_type == 'cls':
                        batch_features = last_hidden_states[:, 0, :]
                    elif self.feature_type == 'mean':
                        attention_mask = inputs['attention_mask']
                        batch_features = torch.sum(
                            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
                        ) / torch.sum(attention_mask, dim=1, keepdim=True)
                    elif self.feature_type == 'max':
                        batch_features = torch.max(last_hidden_states, dim=1)[0]

                    all_features.append(batch_features.cpu())
                    all_labels.extend(batch_labels)

            features = torch.cat(all_features, dim=0).numpy()
            labels = np.array(all_labels)

            return features, labels
