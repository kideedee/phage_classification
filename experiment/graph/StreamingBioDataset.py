import numpy as np
import torch
from Bio import SeqIO
from torch_geometric.data import Dataset

from experiment.graph import overlap_graph_encoding
from experiment.graph.BipartiteData import BipartiteData


class StreamingBioDataset(Dataset):
    def __init__(self, fasta_file, k=3, transform=None):
        super().__init__(transform)
        self.fasta_file = fasta_file
        self.k = k
        self.transform = transform

        # Only store metadata, not actual sequences
        self.sequences_info = []
        self.labels = []

        # First pass: collect metadata only
        count = 1000
        for i, seq_record in enumerate(SeqIO.parse(fasta_file, "fasta")):
            if count > 1000:
                break

            self.sequences_info.append({
                'id': seq_record.id,
                'length': len(seq_record.seq),
                'index': i
            })
            if seq_record.id.split("_")[2] == 'temperate':
                self.labels.append(0)
            else:
                self.labels.append(1)

    def __len__(self):
        return len(self.sequences_info)

    def __getitem__(self, idx):
        sequences = list(SeqIO.parse(self.fasta_file, "fasta"))
        sequence = str(sequences[idx].seq)
        label = self.labels[idx]

        feature = overlap_graph_encoding.create_matrix_feature((sequence, self.k))
        edge = self._create_edge_index()
        pnode_feature = feature.reshape(-1, self.k - 1, 4 ** (self.k * 2))
        pnode_feature = np.moveaxis(pnode_feature, 1, 2)

        zero_layer = feature.reshape(-1, self.k - 1, 4 ** self.k, 4 ** self.k)[:, 0, :, :]
        fnode_feature = np.sum(zero_layer, axis=2).reshape(-1, 4 ** self.k, 1)

        edge_index = torch.tensor(edge, dtype=torch.long)
        x_p = torch.tensor(pnode_feature[0, :, :], dtype=torch.float)
        x_f = torch.tensor(fnode_feature[0, :, :], dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)

        data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, y=y, num_nodes=None)

        if self.transform:
            data = self.transform(data)

        return data

    def _create_edge_index(self):
        edge = []
        for i in range(4 ** (self.k * 2)):
            a = i // 4 ** self.k
            b = i % 4 ** self.k
            edge.append([a, i])
            edge.append([b, i])
        return np.array(edge).T
