import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx


class DeBruijnGraphBuilder:
    def __init__(self, k=3):
        self.k = k

    def build_graph(self, sequence):
        """Xây dựng đồ thị de Bruijn từ sequence"""
        # Tạo các k-mer
        kmers = []
        for i in range(len(sequence) - self.k + 1):
            kmers.append(sequence[i:i + self.k])

        # Tạo mapping từ k-mer sang index
        unique_kmers = list(set(kmers))
        kmer_to_idx = {kmer: i for i, kmer in enumerate(unique_kmers)}

        # Tạo edges (node i -> node j nếu suffix của i = prefix của j)
        edges = []
        for i, kmer1 in enumerate(unique_kmers):
            for j, kmer2 in enumerate(unique_kmers):
                if kmer1[1:] == kmer2[:-1]:  # suffix của kmer1 = prefix của kmer2
                    edges.append([i, j])

        return unique_kmers, edges, kmer_to_idx

    def create_node_features(self, kmers):
        """Tạo node features từ k-mers"""
        # One-hot encoding cho mỗi nucleotide
        nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

        features = []
        for kmer in kmers:
            feature = []
            for nucleotide in kmer:
                one_hot = [0, 0, 0, 0]
                one_hot[nucleotide_map[nucleotide]] = 1
                feature.extend(one_hot)
            features.append(feature)

        return torch.tensor(features, dtype=torch.float)


class GraphEmbeddingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, dropout=0.2):
        super(GraphEmbeddingNet, self).__init__()

        # GCN layers với dropout để tránh overfitting
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        # Final embedding layer
        self.embedding_layer = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        # GCN forward pass với dropout
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)

        # Global pooling để có graph-level embedding
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)

        graph_embedding = global_mean_pool(x, batch)

        # Final embedding
        embedding = self.embedding_layer(graph_embedding)

        return embedding, x  # Return both graph and node embeddings


def visualize_debruijn_graph(kmers, edges):
    """Visualize de Bruijn graph"""
    G = nx.DiGraph()

    # Add nodes
    for i, kmer in enumerate(kmers):
        G.add_node(i, label=kmer)

    # Add edges
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                           node_size=1500, alpha=0.7)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                           arrows=True, arrowsize=20, alpha=0.6)

    # Draw labels
    labels = {i: kmers[i] for i in range(len(kmers))}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title("De Bruijn Graph Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Ví dụ cụ thể với hyperparameters được khuyến nghị
def main():
    # === HYPERPARAMETERS ĐƯỢC KHUYẾN NGHỊ ===
    # Dựa trên research từ ETH Zurich và các paper GNN genomics

    # K-mer parameters
    k = 3  # Bắt đầu với k=3, có thể tăng lên 4-8 cho sequences phức tạp hơn

    # Model architecture parameters
    hidden_dim = 64  # 32-128 cho genomic data (từ ratschlab/genomic-gnn)
    output_dim = 32  # 32-64 cho embedding dimension
    dropout = 0.2  # 0.1-0.3 để tránh overfitting

    # Training parameters
    learning_rate = 0.001  # Adam optimizer standard cho GNN
    batch_size = 32  # 16-256, tốt nhất 32 cho stability
    epochs = 100  # 50-2000 tùy thuộc vào complexity
    weight_decay = 5e-4  # L2 regularization

    print("=== RECOMMENDED HYPERPARAMETERS ===")
    print(f"K-mer size: {k}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Dropout rate: {dropout}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Weight decay: {weight_decay}")

    # Contig example
    contig = "ATCGATCG"

    print(f"\nOriginal contig: {contig}")
    print(f"k-mer size: {k}")

    # Bước 1: Xây dựng đồ thị de Bruijn
    builder = DeBruijnGraphBuilder(k=k)
    kmers, edges, kmer_to_idx = builder.build_graph(contig)

    print(f"\nK-mers found: {kmers}")
    print(f"Edges: {edges}")

    # Bước 2: Tạo node features
    node_features = builder.create_node_features(kmers)
    print(f"\nNode features shape: {node_features.shape}")
    print(f"Feature vector for first k-mer '{kmers[0]}': {node_features[0]}")

    # Bước 3: Tạo PyTorch Geometric data object
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index)

    # Bước 4: Khởi tạo model với hyperparameters được khuyến nghị
    input_dim = node_features.shape[1]  # 4 * k (one-hot cho mỗi nucleotide)
    model = GraphEmbeddingNet(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              output_dim=output_dim,
                              dropout=dropout)

    # Bước 5: Setup optimizer với recommended parameters
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    # Bước 6: Training simulation (demo một vài epochs)
    model.train()
    print(f"\n=== TRAINING SIMULATION ===")

    for epoch in range(5):  # Demo 5 epochs
        optimizer.zero_grad()

        # Forward pass
        graph_embedding, node_embeddings = model(data.x, data.edge_index)

        # Dummy loss for demonstration (trong thực tế cần loss function phù hợp)
        loss = torch.mean(graph_embedding ** 2)  # L2 loss as example

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/5, Loss: {loss.detach().item():.4f}")

    # Bước 7: Final embedding
    model.eval()
    with torch.no_grad():
        graph_embedding, node_embeddings = model(data.x, data.edge_index)

    print(f"\n=== FINAL EMBEDDING RESULTS ===")
    print(f"Graph embedding shape: {graph_embedding.shape}")
    print(f"Graph embedding vector:\n{graph_embedding}")

    print(f"\nNode embeddings shape: {node_embeddings.shape}")
    for i, kmer in enumerate(kmers):
        print(f"Node embedding for '{kmer}': {node_embeddings[i][:5]}...")  # Show first 5 dims

    # Bước 8: Visualize graph
    visualize_debruijn_graph(kmers, edges)

    # Bước 9: Demonstrate embedding properties
    print(f"\n=== EMBEDDING PROPERTIES ===")
    print(f"Embedding dimension: {graph_embedding.shape[1]}")
    print(f"Embedding norm: {torch.norm(graph_embedding).item():.4f}")

    # Test với contig khác
    print(f"\n=== COMPARISON WITH ANOTHER CONTIG ===")
    contig2 = "GCATGCAT"
    kmers2, edges2, _ = builder.build_graph(contig2)
    node_features2 = builder.create_node_features(kmers2)

    if edges2:
        edge_index2 = torch.tensor(edges2, dtype=torch.long).t().contiguous()
    else:
        edge_index2 = torch.empty((2, 0), dtype=torch.long)

    data2 = Data(x=node_features2, edge_index=edge_index2)

    with torch.no_grad():
        graph_embedding2, _ = model(data2.x, data2.edge_index)

    print(f"Contig 1: {contig}")
    print(f"Contig 2: {contig2}")
    print(f"Embedding similarity (cosine): {F.cosine_similarity(graph_embedding, graph_embedding2).item():.4f}")
    print(f"Embedding distance (L2): {torch.norm(graph_embedding - graph_embedding2).item():.4f}")

    print(f"\n=== HYPERPARAMETER TUNING TIPS ===")
    print("1. K-mer size (k):")
    print("   - k=3-4: Cho short sequences hoặc high error rate")
    print("   - k=5-7: Cho standard genomic sequences")
    print("   - k=8+: Cho long, high-quality sequences")

    print("2. Hidden dimension:")
    print("   - 32: Cho small datasets (<1000 sequences)")
    print("   - 64: Cho medium datasets (1000-10000 sequences)")
    print("   - 128+: Cho large datasets (>10000 sequences)")

    print("3. Learning rate:")
    print("   - 0.01: Cho fast convergence, risk overshoot")
    print("   - 0.001: Standard choice, balanced")
    print("   - 0.0001: Cho stable convergence, slower training")

    print("4. Batch size:")
    print("   - 16-32: Cho better generalization")
    print("   - 64-128: Cho faster training")
    print("   - 256+: Cho very large datasets")

    print("5. Epochs:")
    print("   - 50-100: Cho small/simple datasets")
    print("   - 200-500: Cho medium complexity")
    print("   - 1000+: Cho complex genomic tasks")

    # Ví dụ cụ thể
    # Contig example
    contig = "ATCGATCG"
    k = 3

    print(f"Original contig: {contig}")
    print(f"k-mer size: {k}")

    # Bước 1: Xây dựng đồ thị de Bruijn
    builder = DeBruijnGraphBuilder(k=k)
    kmers, edges, kmer_to_idx = builder.build_graph(contig)

    print(f"\nK-mers found: {kmers}")
    print(f"Edges: {edges}")

    # Bước 2: Tạo node features
    node_features = builder.create_node_features(kmers)
    print(f"\nNode features shape: {node_features.shape}")
    print(f"Feature vector for first k-mer '{kmers[0]}': {node_features[0]}")

    # Bước 3: Tạo PyTorch Geometric data object
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index)

    # Bước 4: Khởi tạo model
    input_dim = node_features.shape[1]  # 4 * k (one-hot cho mỗi nucleotide)
    model = GraphEmbeddingNet(input_dim=input_dim,
                              hidden_dim=64,
                              output_dim=32)

    # Bước 5: Forward pass để tạo embedding
    model.eval()
    with torch.no_grad():
        graph_embedding, node_embeddings = model(data.x, data.edge_index)

    print(f"\n=== EMBEDDING RESULTS ===")
    print(f"Graph embedding shape: {graph_embedding.shape}")
    print(f"Graph embedding vector:\n{graph_embedding}")

    print(f"\nNode embeddings shape: {node_embeddings.shape}")
    for i, kmer in enumerate(kmers):
        print(f"Node embedding for '{kmer}': {node_embeddings[i][:5]}...")  # Show first 5 dims

    # Bước 6: Visualize graph
    visualize_debruijn_graph(kmers, edges)

    # Bước 7: Demonstrate embedding properties
    print(f"\n=== EMBEDDING PROPERTIES ===")
    print(f"Embedding dimension: {graph_embedding.shape[1]}")
    print(f"Embedding norm: {torch.norm(graph_embedding).item():.4f}")

    # Test với contig khác
    print(f"\n=== COMPARISON WITH ANOTHER CONTIG ===")
    contig2 = "GCATGCAT"
    kmers2, edges2, _ = builder.build_graph(contig2)
    node_features2 = builder.create_node_features(kmers2)

    if edges2:
        edge_index2 = torch.tensor(edges2, dtype=torch.long).t().contiguous()
    else:
        edge_index2 = torch.empty((2, 0), dtype=torch.long)

    data2 = Data(x=node_features2, edge_index=edge_index2)

    with torch.no_grad():
        graph_embedding2, _ = model(data2.x, data2.edge_index)

    print(f"Contig 1: {contig}")
    print(f"Contig 2: {contig2}")
    print(f"Embedding similarity (cosine): {F.cosine_similarity(graph_embedding, graph_embedding2).item():.4f}")
    print(f"Embedding distance (L2): {torch.norm(graph_embedding - graph_embedding2).item():.4f}")


if __name__ == "__main__":
    main()