import os

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset

from experiment.graph import overlap_graph_encoding
from experiment.graph.BipartiteData import BipartiteData


class GraphDataset:

    def __init__(self, pnode_feature, fnode_feature, other_feature, edge, graph_label):
        self.pnode_feature = pnode_feature
        self.fnode_feature = fnode_feature
        self.other_feature = other_feature
        self.edge = edge
        self.graph_label = graph_label

    def process(self):
        data_list = []  # graph classification need to define data_list for multiple graph
        for i in range(self.pnode_feature.shape[0]):
            edge_index = torch.tensor(self.edge, dtype=torch.long)  # edge_index should be long type

            x_p = torch.tensor(self.pnode_feature[i, :, :], dtype=torch.float)
            x_f = torch.tensor(self.fnode_feature[i, :, :], dtype=torch.float)
            if type(self.graph_label) == np.ndarray:
                y = torch.tensor([self.graph_label[i]], dtype=torch.long)
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, y=y, num_nodes=None)
            else:
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, num_nodes=None)

            if type(self.other_feature) == np.ndarray:
                other_feature = torch.tensor(self.other_feature[i, :], dtype=torch.float)
                data._add_other_feature(other_feature)

            data_list.append(data)

        return data_list


class GraphDatasetInMem(InMemoryDataset):

    def __init__(self, pnode_feature, fnode_feature, other_feature, edge, graph_label, root, transform=None,
                 pre_transform=None, output_name="data.dataset"):
        self.pnode_feature = pnode_feature
        self.fnode_feature = fnode_feature
        self.other_feature = other_feature
        self.edge = edge
        self.graph_label = graph_label
        self.output_name = output_name

        # Call parent constructor
        super(GraphDatasetInMem, self).__init__(root, transform, pre_transform)

        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def processed_paths(self):
        """Return the paths to processed files."""
        return [os.path.join(self.processed_dir, fname) for fname in self.processed_file_names]

    def download(self):
        pass

    def process(self):
        data_list = []  # graph classification need to define data_list for multiple graph
        for i in range(self.pnode_feature.shape[0]):
            edge_index = torch.tensor(self.edge, dtype=torch.long)  # edge_index should be long type

            x_p = torch.tensor(self.pnode_feature[i, :, :], dtype=torch.float)
            x_f = torch.tensor(self.fnode_feature[i, :, :], dtype=torch.float)
            if type(self.graph_label) == np.ndarray:
                y = torch.tensor([self.graph_label[i]], dtype=torch.long)
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, y=y, num_nodes=None)
            else:
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, num_nodes=None)

            if type(self.other_feature) == np.ndarray:
                other_feature = torch.tensor(self.other_feature[i, :], dtype=torch.float)
                data._add_other_feature(other_feature)

            data_list.append(data)

        data, slices = self.collate(data_list)  # Here used to be [data] for one graph
        torch.save((data, slices), self.processed_paths[0])

        return data_list


class Biodata:
    def __init__(self, df: pd.DataFrame, feature_file=None, k=3, d=3, seqtype="DNA", output_name="data.dataset"):
        self.dna_seq = df['sequence'].tolist()
        self.label = df['label'].values
        self.output_name = output_name

        if feature_file == None:
            self.other_feature = None
        else:
            self.other_feature = np.loadtxt(feature_file)

        self.k = k
        self.d = d
        self.seqtype = seqtype

        self.edge = []
        for i in range(4 ** (k * 2)):
            a = i // 4 ** k
            b = i % 4 ** k
            self.edge.append([a, i])
            self.edge.append([b, i])
        self.edge = np.array(self.edge).T

    def encode(self, save_dataset=True, save_path="./"):
        print("Encoding sequences...")
        seq_list = self.dna_seq

        feature = Parallel(n_jobs=-1)(
            delayed(overlap_graph_encoding.create_matrix_feature)(
                (sequence, self.k, self.d)
            ) for sequence in seq_list
        )

        feature = np.array(feature)
        self.pnode_feature = feature.reshape(-1, self.d, 4 ** (self.k * 2))
        self.pnode_feature = np.moveaxis(self.pnode_feature, 1, 2)
        zero_layer = feature.reshape(-1, self.d, 4 ** self.k, 4 ** self.k)[:, 0, :, :]
        self.fnode_feature = np.sum(zero_layer, axis=2).reshape(-1, 4 ** self.k, 1)
        del zero_layer

        if save_dataset:
            dataset = GraphDatasetInMem(self.pnode_feature, self.fnode_feature, self.other_feature, self.edge,
                                        self.label, root=save_path, output_name=self.output_name)

        else:
            graph = GraphDataset(self.pnode_feature, self.fnode_feature, self.other_feature, self.edge, self.label)
            dataset = graph.process()

        return dataset


def load_data():
    dataset = torch.load("train/processed/train.dataset", weights_only=False)
    print(dataset)


def create_dataset(fasta_file, output_dir):
    sequences = []
    labels = []

    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(seq_record.seq))
        if seq_record.id.split("_")[2] == 'temperate':
            labels.append(0)
        else:
            labels.append(1)

    df = pd.DataFrame({"sequence": sequences, "label": labels})
    train_df, valid_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    train_df, test_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["label"])

    data = Biodata(df=test_df, k=3, d=3 - 1, output_name="data.dataset")
    data.encode(save_path=output_dir, save_dataset=True)


def create_dataset_from_csv(fasta_file, k, d, output_dir):
    df = pd.read_csv(fasta_file)
    df.columns.values[1] = 'label'
    data = Biodata(df=df, k=k, d=d, output_name="data.dataset")
    data.encode(save_path=output_dir, save_dataset=True)

# if __name__ == '__main__':
#     k = 3
#     d = k - 1
#
#     for fold in range(1, 6):
#         for i in range(4):
#             if i ==0:
#                 group = "100_400"
#             elif i ==1:
#                 group = "400_800"
#             elif i ==2:
#                 group = "800_1200"
#             else:
#                 group = "1200_1800"
#
#             print("="*100)
#             print(f"Start group: {group}, fold: {fold}")
#             train_file = f"E:\master\\final_project\data\my_data\\fasta\\{group}\\{fold}\\train\data.fa"
#             test_file = f"E:\master\\final_project\data\my_data\\fasta\\{group}\\{fold}\\test\data.fa"
#
#             create_dataset(train_file, f"train")
#             create_dataset(test_file, f"test")
#
#             train_ds = GraphDatasetInMem(
#                 pnode_feature=None,  # These can be None since we're loading
#                 fnode_feature=None,
#                 other_feature=None,
#                 edge=None,
#                 graph_label=None,
#                 root="./train"  # Path where your processed data is stored
#             )
#
#             test_ds = GraphDatasetInMem(
#                 pnode_feature=None,  # These can be None since we're loading
#                 fnode_feature=None,
#                 other_feature=None,
#                 edge=None,
#                 graph_label=None,
#                 root="./test"  # Path where your processed data is stored
#             )
#
#             model = GCNModel(k=k, d=d)
#             train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, follow_batch=['x_src', 'x_dst'])
#             # valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False, follow_batch=['x_src', 'x_dst'])
#             test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, follow_batch=['x_src', 'x_dst'])
#
#             gcn_model.train(train_loader, test_loader, model)

# train_file = "E:\master\\final_project\data\my_data\start_here\\fold_1\\train\data.csv"
# test_file = "E:\master\\final_project\data\my_data\start_here\\fold_1\\test\data.csv"


# create_dataset_from_csv(train_file, k, d, "train")
# create_dataset_from_csv(test_file, k, d, "test")
