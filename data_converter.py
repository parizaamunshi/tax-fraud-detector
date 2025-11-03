import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
import re


class TransactionToGraphConverter:
    """
    Converts your transaction dataset into a graph structure
    for GNN-based fraud detection
    """

    def __init__(self):
        self.entity_mapping = {}  # Maps entity names to node IDs
        self.node_id_counter = 0

    def convert_dataset(self, df: pd.DataFrame) -> Tuple[nx.DiGraph, Dict]:
        """
        Convert transaction dataframe to graph

        Args:
            df: DataFrame with columns: step, type, amount, nameOrig,
                oldbalanceOrig, newbalanceOrig, nameDest,
                oldbalanceDest, newbalanceDest, isFraud

        Returns:
            graph: NetworkX directed graph
            node_features: Dictionary of node features
        """
        G = nx.DiGraph()

        # Handle different possible column name variations
        column_mapping = {
            "oldbalanceOrg": "oldbalanceOrig",
            "newbalanceOrg": "newbalanceOrig",
        }

        # Rename columns if needed
        df = df.rename(columns=column_mapping)

        # Verify required columns exist
        required_cols = ["nameOrig", "nameDest", "amount", "type", "isFraud", "step"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"Converting {len(df)} transactions to graph...")
        print(f"Available columns: {list(df.columns)}")

        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"Processed {idx} transactions...")

            # Extract entity information
            orig_entity = row["nameOrig"]
            dest_entity = row["nameDest"]

            # Determine entity types from name prefix
            orig_type = self._get_entity_type(orig_entity)
            dest_type = self._get_entity_type(dest_entity)

            # Get balance values with fallback
            orig_old_balance = row.get("oldbalanceOrig", 0)
            dest_old_balance = row.get("oldbalanceDest", 0)

            # Add or update origin node
            if orig_entity not in self.entity_mapping:
                self.entity_mapping[orig_entity] = self.node_id_counter
                self.node_id_counter += 1

                G.add_node(
                    self.entity_mapping[orig_entity],
                    name=orig_entity,
                    entity_type=orig_type,
                    total_sent=0,
                    total_received=0,
                    transaction_count=0,
                    avg_balance=orig_old_balance,
                    is_fraudster=0,
                )

            # Add or update destination node
            if dest_entity not in self.entity_mapping:
                self.entity_mapping[dest_entity] = self.node_id_counter
                self.node_id_counter += 1

                G.add_node(
                    self.entity_mapping[dest_entity],
                    name=dest_entity,
                    entity_type=dest_type,
                    total_sent=0,
                    total_received=0,
                    transaction_count=0,
                    avg_balance=dest_old_balance,
                    is_fraudster=0,
                )

            orig_id = self.entity_mapping[orig_entity]
            dest_id = self.entity_mapping[dest_entity]

            # Update node statistics
            G.nodes[orig_id]["total_sent"] += row["amount"]
            G.nodes[orig_id]["transaction_count"] += 1
            G.nodes[dest_id]["total_received"] += row["amount"]
            G.nodes[dest_id]["transaction_count"] += 1

            # Mark nodes involved in fraud
            if row["isFraud"] == 1:
                G.nodes[orig_id]["is_fraudster"] = 1
                G.nodes[dest_id]["is_fraudster"] = 1

            # Add edge (transaction)
            if G.has_edge(orig_id, dest_id):
                # Update existing edge
                G[orig_id][dest_id]["total_amount"] += row["amount"]
                G[orig_id][dest_id]["transaction_count"] += 1
                G[orig_id][dest_id]["transactions"].append(
                    {
                        "step": row["step"],
                        "type": row["type"],
                        "amount": row["amount"],
                        "is_fraud": row["isFraud"],
                    }
                )
            else:
                # Create new edge
                G.add_edge(
                    orig_id,
                    dest_id,
                    transaction_type=row["type"],
                    total_amount=row["amount"],
                    transaction_count=1,
                    is_fraudulent=row["isFraud"],
                    transactions=[
                        {
                            "step": row["step"],
                            "type": row["type"],
                            "amount": row["amount"],
                            "is_fraud": row["isFraud"],
                        }
                    ],
                )

        print(
            f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        # Compute additional graph features
        self._compute_graph_features(G)

        return G, self.entity_mapping

    def _get_entity_type(self, entity_name: str) -> str:
        """Determine entity type from name prefix"""
        if entity_name.startswith("C"):
            return "customer"
        elif entity_name.startswith("M"):
            return "merchant"
        else:
            return "unknown"

    def _compute_graph_features(self, G: nx.DiGraph):
        """Compute additional graph-based features"""
        print("Computing graph features...")

        num_nodes = G.number_of_nodes()

        # Degree centrality (fast)
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())

        # For large graphs (>1M nodes), skip expensive computations
        if num_nodes > 1000000:
            print(f"  Large graph ({num_nodes:,} nodes), using fast approximations...")
            pagerank = {node: 0 for node in G.nodes()}
            clustering = {node: 0 for node in G.nodes()}
        else:
            # PageRank (expensive for large graphs)
            print("  Computing PageRank...")
            try:
                pagerank = nx.pagerank(G, max_iter=50, tol=1e-3)
            except:
                pagerank = {node: 0 for node in G.nodes()}

            # Clustering coefficient (very expensive)
            print("  Computing clustering coefficients...")
            clustering = nx.clustering(G.to_undirected())

        # Update node attributes
        for node in G.nodes():
            G.nodes[node]["in_degree"] = in_degree.get(node, 0)
            G.nodes[node]["out_degree"] = out_degree.get(node, 0)
            G.nodes[node]["pagerank"] = pagerank.get(node, 0)
            G.nodes[node]["clustering"] = clustering.get(node, 0)

            # Compute transaction velocity (avg transactions per step)
            total_trans = G.nodes[node]["transaction_count"]
            if total_trans > 0:
                G.nodes[node]["velocity"] = total_trans / 100  # Normalized
            else:
                G.nodes[node]["velocity"] = 0

    def extract_node_features(
        self, G: nx.DiGraph, node_list: List[int] = None
    ) -> np.ndarray:
        """
        Extract feature matrix for GNN

        Returns:
            Feature matrix [num_nodes, num_features]
        """
        if node_list is None:
            node_list = list(G.nodes())

        features = []
        feature_names = [
            "total_sent",
            "total_received",
            "transaction_count",
            "avg_balance",
            "in_degree",
            "out_degree",
            "pagerank",
            "clustering",
            "velocity",
        ]

        for node in node_list:
            node_data = G.nodes[node]
            feature_vector = [node_data.get(feat, 0) for feat in feature_names]
            features.append(feature_vector)

        # Normalize features
        features = np.array(features, dtype=np.float32)

        # Log transform for large values
        features[:, 0] = np.log1p(features[:, 0])  # total_sent
        features[:, 1] = np.log1p(features[:, 1])  # total_received
        features[:, 3] = np.log1p(features[:, 3])  # avg_balance

        # Standard scaling
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features = (features - mean) / std

        return features

    def get_labels(self, G: nx.DiGraph, node_list: List[int] = None) -> np.ndarray:
        """Extract fraud labels for nodes"""
        if node_list is None:
            node_list = list(G.nodes())

        labels = [G.nodes[node]["is_fraudster"] for node in node_list]
        return np.array(labels, dtype=np.int64)

    def save_graph(
        self, G: nx.DiGraph, filepath: str = "graph_data/transaction_graph.gpickle"
    ):
        """Save graph to file"""
        import os
        import pickle

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Use pickle directly (works with all NetworkX versions)
        with open(filepath, "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        print(f"Graph saved to {filepath}")

    def load_graph(
        self, filepath: str = "graph_data/transaction_graph.gpickle"
    ) -> nx.DiGraph:
        """Load graph from file"""
        import pickle

        with open(filepath, "rb") as f:
            G = pickle.load(f)
        print(f"Graph loaded from {filepath}")
        return G


def create_pytorch_geometric_data(
    G: nx.DiGraph, converter: TransactionToGraphConverter
):
    """
    Convert NetworkX graph to PyTorch Geometric Data object
    """
    import torch
    from torch_geometric.data import Data

    # Get node features and labels
    node_list = list(G.nodes())
    x = torch.tensor(converter.extract_node_features(G, node_list), dtype=torch.float)
    y = torch.tensor(converter.get_labels(G, node_list), dtype=torch.long)

    # Create edge index
    edge_list = list(G.edges())
    edge_index = (
        torch.tensor(
            [[node_list.index(u), node_list.index(v)] for u, v in edge_list],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )

    # Edge features (transaction amounts, counts)
    edge_attr = []
    for u, v in edge_list:
        edge_data = G[u][v]
        edge_attr.append(
            [
                np.log1p(edge_data["total_amount"]),
                edge_data["transaction_count"],
                edge_data["is_fraudulent"],
            ]
        )
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create train/val/test masks
    num_nodes = len(node_list)
    indices = torch.randperm(num_nodes)

    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size : train_size + val_size]] = True
    test_mask[indices[train_size + val_size :]] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data


# Example usage
if __name__ == "__main__":
    # Load your transaction data
    df = pd.read_csv("/path/to/fraud_detection.csv")

    # Convert to graph
    converter = TransactionToGraphConverter()
    G, entity_mapping = converter.convert_dataset(df)

    # Save graph
    converter.save_graph(G)

    # Create PyTorch Geometric data
    data = create_pytorch_geometric_data(G, converter)

    print(f"\nPyTorch Geometric Data:")
    print(f"Nodes: {data.x.shape[0]}")
    print(f"Features: {data.x.shape[1]}")
    print(f"Edges: {data.edge_index.shape[1]}")
    print(f"Train nodes: {data.train_mask.sum()}")
    print(f"Val nodes: {data.val_mask.sum()}")
    print(f"Test nodes: {data.test_mask.sum()}")
