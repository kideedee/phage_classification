import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import json


class DeBruijnGraph:
    def __init__(self, k=3):
        """
        Khởi tạo De Bruijn Graph

        Args:
            k (int): Độ dài k-mer
        """
        self.k = k
        self.graph = defaultdict(list)
        self.node_counts = defaultdict(int)
        self.edge_counts = defaultdict(int)
        self.original_sequence = ""

    def add_sequence(self, sequence):
        """
        Thêm sequence vào De Bruijn Graph

        Args:
            sequence (str): DNA sequence
        """
        self.original_sequence = sequence
        sequence = sequence.upper()

        # Tạo k-mers từ sequence
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]

            # Tạo nodes: (k-1)-mers
            left_node = kmer[:-1]  # prefix
            right_node = kmer[1:]  # suffix

            # Thêm edge từ left_node đến right_node
            self.graph[left_node].append(right_node)

            # Đếm tần suất nodes và edges
            self.node_counts[left_node] += 1
            self.node_counts[right_node] += 1
            self.edge_counts[(left_node, right_node)] += 1

    def get_nodes(self):
        """Lấy tất cả nodes trong graph"""
        nodes = set()
        for node in self.graph.keys():
            nodes.add(node)
        for neighbors in self.graph.values():
            for neighbor in neighbors:
                nodes.add(neighbor)
        return list(nodes)

    def get_edges(self):
        """Lấy tất cả edges trong graph"""
        edges = []
        for source, targets in self.graph.items():
            for target in targets:
                edges.append((source, target))
        return edges

    def get_edge_weights(self):
        """Lấy trọng số của các edges"""
        return dict(self.edge_counts)

    def _create_node_mapping(self):
        """Tạo mapping từ node names sang indices"""
        nodes = sorted(self.get_nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        return node_to_idx, idx_to_node

    def _encode_sequence(self, sequence):
        """
        Mã hóa DNA sequence thành vector số
        A=0, T=1, G=2, C=3
        """
        mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        return [mapping.get(base, 0) for base in sequence.upper()]

    def _create_node_features(self, node_to_idx):
        """
        Tạo node features từ sequence và frequency
        """
        nodes = sorted(self.get_nodes())
        features = []

        for node in nodes:
            # One-hot encoding cho từng base trong k-mer
            sequence_encoding = []
            for base in node:
                base_vector = [0, 0, 0, 0]  # A, T, G, C
                if base == 'A':
                    base_vector[0] = 1
                elif base == 'T':
                    base_vector[1] = 1
                elif base == 'G':
                    base_vector[2] = 1
                elif base == 'C':
                    base_vector[3] = 1
                sequence_encoding.extend(base_vector)

            # Thêm frequency count
            frequency = self.node_counts[node]

            # Thêm in-degree và out-degree
            in_degree = sum(1 for source, targets in self.graph.items() if node in targets)
            out_degree = len(self.graph[node])

            # Combine features
            node_features = sequence_encoding + [frequency, in_degree, out_degree]
            features.append(node_features)

        return torch.tensor(features, dtype=torch.float)

    def _create_edge_features(self, edges):
        """
        Tạo edge features từ edge weights
        """
        edge_features = []
        for source, target in edges:
            weight = self.edge_counts[(source, target)]
            edge_features.append([weight])

        return torch.tensor(edge_features, dtype=torch.float)

    def to_pytorch_geometric(self, include_edge_features=True):
        """
        Chuyển đổi De Bruijn Graph sang PyTorch Geometric Data object

        Args:
            include_edge_features (bool): Có bao gồm edge features không

        Returns:
            Data: PyTorch Geometric Data object
        """
        # Tạo node mapping
        node_to_idx, idx_to_node = self._create_node_mapping()

        # Tạo edge_index
        edges = self.get_edges()
        edge_index = []
        for source, target in edges:
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]
            edge_index.append([source_idx, target_idx])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Tạo node features
        x = self._create_node_features(node_to_idx)

        # Tạo edge features (nếu cần)
        edge_attr = None
        if include_edge_features:
            edge_attr = self._create_edge_features(edges)

        # Tạo Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Thêm metadata
        data.num_nodes = len(node_to_idx)
        data.node_to_idx = node_to_idx
        data.idx_to_node = idx_to_node
        data.original_sequence = self.original_sequence
        data.k = self.k

        return data

    def save_pytorch_geometric(self, filename, include_edge_features=True):
        """
        Lưu graph dưới dạng PyTorch Geometric Data object

        Args:
            filename (str): Tên file để lưu
            include_edge_features (bool): Có bao gồm edge features không
        """
        data = self.to_pytorch_geometric(include_edge_features)

        # Lưu với pickle protocol để tránh lỗi weights_only
        torch.save(data, filename, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Graph saved to {filename}")
        return data

    def save_networkx(self, filename):
        """
        Lưu graph dưới dạng NetworkX format

        Args:
            filename (str): Tên file để lưu
        """
        G = nx.DiGraph()

        # Thêm nodes với attributes
        for node in self.get_nodes():
            G.add_node(node,
                       frequency=self.node_counts[node],
                       sequence=node)

        # Thêm edges với weights
        for source, targets in self.graph.items():
            for target in targets:
                weight = self.edge_counts[(source, target)]
                G.add_edge(source, target, weight=weight)

        # Lưu graph
        nx.write_gml(G, filename)
        print(f"NetworkX graph saved to {filename}")
        return G

    def save_json(self, filename):
        """
        Lưu graph dưới dạng JSON format

        Args:
            filename (str): Tên file để lưu
        """
        graph_data = {
            'k': self.k,
            'original_sequence': self.original_sequence,
            'nodes': self.get_nodes(),
            'edges': self.get_edges(),
            'node_counts': dict(self.node_counts),
            'edge_counts': {f"{edge[0]}->{edge[1]}": count
                            for edge, count in self.edge_counts.items()},
            'statistics': self.get_statistics()
        }

        with open(filename, 'w') as f:
            json.dump(graph_data, f, indent=2)

        print(f"Graph data saved to {filename}")
        return graph_data

    @classmethod
    def load_pytorch_geometric(cls, filename):
        """
        Tải graph từ PyTorch Geometric file

        Args:
            filename (str): Tên file để tải

        Returns:
            tuple: (DeBruijnGraph object, Data object)
        """
        # Import các class cần thiết để đăng ký safe globals
        from torch_geometric.data.data import DataEdgeAttr
        from torch_geometric.data import Data

        # Thêm safe globals cho PyTorch Geometric
        torch.serialization.add_safe_globals([DataEdgeAttr, Data])

        try:
            # Thử load với weights_only=True trước
            data = torch.load(filename, weights_only=True)
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=True, falling back to weights_only=False")
            print(f"Error: {e}")
            # Fallback to weights_only=False nếu cần thiết
            data = torch.load(filename, weights_only=False)

        # Tạo lại DeBruijnGraph object
        db_graph = cls(k=data.k)
        db_graph.add_sequence(data.original_sequence)

        return db_graph, data

    def print_pytorch_geometric_info(self):
        """
        In thông tin về PyTorch Geometric representation
        """
        data = self.to_pytorch_geometric()

        print("=== PyTorch Geometric Graph Information ===")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
        print(f"Node feature dimensions: {data.x.shape}")
        print(f"Edge feature dimensions: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")

        print("\nNode features structure:")
        print(f"  - One-hot encoding for each base: {4 * (data.k - 1)} dimensions")
        print(f"  - Frequency count: 1 dimension")
        print(f"  - In-degree: 1 dimension")
        print(f"  - Out-degree: 1 dimension")
        print(f"  - Total: {data.x.shape[1]} dimensions")

        print("\nFirst few nodes:")
        for i in range(min(5, data.num_nodes)):
            node_name = data.idx_to_node[i]
            features = data.x[i]
            print(f"  Node {i} ({node_name}): {features[:8].tolist()}... (showing first 8 features)")

        return data

    def save_pytorch_geometric_safe(self, filename, include_edge_features=True):
        """
        Lưu graph an toàn hơn bằng cách sử dụng pickle riêng

        Args:
            filename (str): Tên file để lưu
            include_edge_features (bool): Có bao gồm edge features không
        """
        data = self.to_pytorch_geometric(include_edge_features)

        # Tạo dictionary chứa tất cả thông tin cần thiết
        save_data = {
            'x': data.x,
            'edge_index': data.edge_index,
            'edge_attr': data.edge_attr,
            'num_nodes': data.num_nodes,
            'node_to_idx': data.node_to_idx,
            'idx_to_node': data.idx_to_node,
            'original_sequence': data.original_sequence,
            'k': data.k
        }

        # Lưu bằng pickle
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Graph safely saved to {filename}")
        return data

    @classmethod
    def load_pytorch_geometric_safe(cls, filename):
        """
        Tải graph từ file pickle an toàn

        Args:
            filename (str): Tên file để tải

        Returns:
            tuple: (DeBruijnGraph object, Data object)
        """
        # Tải data từ pickle
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)

        # Tái tạo Data object
        data = Data(
            x=save_data['x'],
            edge_index=save_data['edge_index'],
            edge_attr=save_data['edge_attr']
        )

        # Thêm metadata
        data.num_nodes = save_data['num_nodes']
        data.node_to_idx = save_data['node_to_idx']
        data.idx_to_node = save_data['idx_to_node']
        data.original_sequence = save_data['original_sequence']
        data.k = save_data['k']

        # Tạo lại DeBruijnGraph object
        db_graph = cls(k=data.k)
        db_graph.add_sequence(data.original_sequence)

        return db_graph, data
        """
        In thông tin về PyTorch Geometric representation
        """
        data = self.to_pytorch_geometric()

        print("=== PyTorch Geometric Graph Information ===")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
        print(f"Node feature dimensions: {data.x.shape}")
        print(f"Edge feature dimensions: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")

        print("\nNode features structure:")
        print(f"  - One-hot encoding for each base: {4 * (data.k - 1)} dimensions")
        print(f"  - Frequency count: 1 dimension")
        print(f"  - In-degree: 1 dimension")
        print(f"  - Out-degree: 1 dimension")
        print(f"  - Total: {data.x.shape[1]} dimensions")

        print("\nFirst few nodes:")
        for i in range(min(5, data.num_nodes)):
            node_name = data.idx_to_node[i]
            features = data.x[i]
            print(f"  Node {i} ({node_name}): {features[:8].tolist()}... (showing first 8 features)")

    def find_eulerian_path(self):
        """
        Tìm đường đi Eulerian trong graph (nếu có)
        Trả về chuỗi DNA được reconstruct
        """
        # Kiểm tra điều kiện Eulerian path
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)

        for source, targets in self.graph.items():
            out_degree[source] += len(targets)
            for target in targets:
                in_degree[target] += 1

        # Tìm start node (out_degree - in_degree = 1)
        start_node = None
        end_node = None

        for node in self.get_nodes():
            diff = out_degree[node] - in_degree[node]
            if diff == 1:
                start_node = node
            elif diff == -1:
                end_node = node

        # Nếu không có start node, chọn node bất kỳ
        if start_node is None:
            start_node = list(self.graph.keys())[0]

        # Hierholzer's algorithm để tìm Eulerian path
        path = self._find_eulerian_path_hierholzer(start_node)

        # Reconstruct sequence từ path
        if path:
            sequence = path[0]
            for i in range(1, len(path)):
                sequence += path[i][-1]
            return sequence

        return None

    def _find_eulerian_path_hierholzer(self, start_node):
        """
        Thuật toán Hierholzer để tìm Eulerian path
        """
        # Tạo bản sao graph để không thay đổi original
        graph_copy = defaultdict(list)
        for node, neighbors in self.graph.items():
            graph_copy[node] = neighbors.copy()

        path = []
        stack = [start_node]

        while stack:
            curr = stack[-1]
            if graph_copy[curr]:
                next_node = graph_copy[curr].pop(0)
                stack.append(next_node)
            else:
                path.append(stack.pop())

        return path[::-1]

    def get_statistics(self):
        """Lấy thống kê về graph"""
        nodes = self.get_nodes()
        edges = self.get_edges()

        # Tính in-degree và out-degree
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)

        for source, targets in self.graph.items():
            out_degree[source] += len(targets)
            for target in targets:
                in_degree[target] += 1

        stats = {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'avg_in_degree': np.mean([in_degree[node] for node in nodes]),
            'avg_out_degree': np.mean([out_degree[node] for node in nodes]),
            'max_in_degree': max([in_degree[node] for node in nodes]),
            'max_out_degree': max([out_degree[node] for node in nodes]),
            'density': len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
        }

        return stats

    def visualize(self, figsize=(12, 8), node_size=1000, font_size=8):
        """
        Visualize De Bruijn Graph

        Args:
            figsize (tuple): Kích thước figure
            node_size (int): Kích thước nodes
            font_size (int): Kích thước font
        """
        # Tạo NetworkX graph
        G = nx.DiGraph()

        # Thêm nodes
        nodes = self.get_nodes()
        G.add_nodes_from(nodes)

        # Thêm edges với weights
        for source, targets in self.graph.items():
            for target in targets:
                if G.has_edge(source, target):
                    G[source][target]['weight'] += 1
                else:
                    G.add_edge(source, target, weight=1)

        # Tạo figure với subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Main graph visualization
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Vẽ nodes
        node_colors = ['lightblue' if node in self.graph else 'lightcoral' for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=node_size, ax=ax1)

        # Vẽ edges
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                               width=[w * 2 for w in edge_weights],
                               arrowsize=20, ax=ax1)

        # Vẽ labels
        nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax1)

        # Vẽ edge labels (weights)
        edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=font_size - 2, ax=ax1)

        ax1.set_title(f'De Bruijn Graph (k={self.k})\nSequence: {self.original_sequence}',
                      fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 2. Node degree distribution
        in_degrees = [G.in_degree(node) for node in G.nodes()]
        out_degrees = [G.out_degree(node) for node in G.nodes()]

        ax2.hist(in_degrees, bins=max(1, max(in_degrees)), alpha=0.7, label='In-degree', color='blue')
        ax2.hist(out_degrees, bins=max(1, max(out_degrees)), alpha=0.7, label='Out-degree', color='red')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Degree Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Node frequency heatmap
        node_freq = [self.node_counts[node] for node in sorted(nodes)]
        node_names = sorted(nodes)

        ax3.barh(range(len(node_names)), node_freq, color='skyblue')
        ax3.set_yticks(range(len(node_names)))
        ax3.set_yticklabels(node_names, fontsize=font_size)
        ax3.set_xlabel('Frequency')
        ax3.set_title('Node Frequency')
        ax3.grid(True, alpha=0.3)

        # 4. Graph statistics
        stats = self.get_statistics()
        stats_text = f"""Graph Statistics:

Nodes: {stats['num_nodes']}
Edges: {stats['num_edges']}
Density: {stats['density']:.3f}

Average In-degree: {stats['avg_in_degree']:.2f}
Average Out-degree: {stats['avg_out_degree']:.2f}
Max In-degree: {stats['max_in_degree']}
Max Out-degree: {stats['max_out_degree']}

Original sequence length: {len(self.original_sequence)}
K-mer size: {self.k}
"""

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Statistics')

        plt.tight_layout()
        plt.show()

        return G


# Ví dụ sử dụng với PyTorch Geometric
def pytorch_geometric_example():
    """Ví dụ về cách sử dụng với PyTorch Geometric"""

    # Tạo De Bruijn Graph
    db_graph = DeBruijnGraph(k=3)
    sequence = "ATGCGTGCA"
    db_graph.add_sequence(sequence)

    print(f"Original sequence: {sequence}")
    print(f"K-mer size: {db_graph.k}")

    # Chuyển đổi sang PyTorch Geometric
    data = db_graph.to_pytorch_geometric()

    # In thông tin về PyTorch Geometric representation
    db_graph.print_pytorch_geometric_info()

    # Lưu graph
    print("\n=== Saving graphs ===")

    # Sử dụng phương thức an toàn hơn
    db_graph.save_pytorch_geometric_safe("debruijn_graph_safe.pkl")
    db_graph.save_networkx("debruijn_graph.gml")
    db_graph.save_json("debruijn_graph.json")

    # Thử lưu bằng torch.save (có thể gặp lỗi khi load)
    try:
        db_graph.save_pytorch_geometric("debruijn_graph.pt")
    except Exception as e:
        print(f"Warning: torch.save failed: {e}")

    # Tải lại graph
    print("\n=== Loading graph ===")
    try:
        # Thử load bằng torch.load trước
        loaded_graph, loaded_data = DeBruijnGraph.load_pytorch_geometric("debruijn_graph.pt")
        print(f"Successfully loaded with torch.load: {loaded_data.original_sequence}")
    except Exception as e:
        print(f"torch.load failed: {e}")
        print("Using safe pickle method instead...")
        # Fallback sang pickle method
        loaded_graph, loaded_data = DeBruijnGraph.load_pytorch_geometric_safe("debruijn_graph_safe.pkl")
        print(f"Successfully loaded with pickle: {loaded_data.original_sequence}")

    return db_graph, data


def batch_processing_example():
    """Ví dụ về batch processing nhiều graphs"""
    from torch_geometric.data import DataLoader

    sequences = [
        "ATGCGTGCA",
        "GCATGCATGC",
        "TTAAGCCGGT",
        "CGATCGATCG"
    ]

    # Tạo danh sách Data objects
    data_list = []
    for i, seq in enumerate(sequences):
        db_graph = DeBruijnGraph(k=3)
        db_graph.add_sequence(seq)
        data = db_graph.to_pytorch_geometric()
        data.y = torch.tensor([i], dtype=torch.long)  # Graph-level label
        data_list.append(data)

    # Tạo DataLoader
    loader = DataLoader(data_list, batch_size=2, shuffle=True)

    print("=== Batch Processing Example ===")
    for batch_idx, batch in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Batch size: {batch.num_graphs}")
        print(f"  Total nodes: {batch.num_nodes}")
        print(f"  Total edges: {batch.num_edges}")
        print(f"  Node features shape: {batch.x.shape}")
        print(f"  Edge features shape: {batch.edge_attr.shape if batch.edge_attr is not None else 'None'}")
        print(f"  Labels: {batch.y}")
        print()


if __name__ == "__main__":
    # Chạy ví dụ PyTorch Geometric
    print("=== PyTorch Geometric Example ===")
    db_graph, data = pytorch_geometric_example()

    # print("\n=== Batch Processing Example ===")
    # batch_processing_example()

    # Ví dụ với sequence phức tạp hơn
    print("\n=== Complex sequence example ===")
    complex_graph = DeBruijnGraph(k=4)
    complex_sequence = "ATGCGTGCATGCATGCGTGCA"
    complex_graph.add_sequence(complex_sequence)

    # Chuyển đổi và in thông tin
    complex_data = complex_graph.to_pytorch_geometric()
    complex_graph.print_pytorch_geometric_info()

    # Lưu graph
    complex_graph.save_pytorch_geometric("complex_debruijn_graph.pt")

    print("\n=== Visualization ===")
    complex_graph.visualize()