import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import Data
import numpy as np
from typing import Tuple


class GraphAttentionFraudDetector(nn.Module):
    """
    Graph Attention Network (GAT) for tax fraud detection
    Uses attention mechanisms to focus on suspicious relationships
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        # First GAT layer with multi-head attention
        self.conv1 = GATConv(
            num_node_features, hidden_channels, heads=num_heads, dropout=dropout
        )

        # Second GAT layer
        self.conv2 = GATConv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
        )

        # Third GAT layer (single head for final representation)
        self.conv3 = GATConv(
            hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout
        )

        # Classification layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels * num_heads)

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        """
        Forward pass through the network

        Args:
            x: Node feature matrix [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features (optional)
            return_attention: Whether to return attention weights

        Returns:
            Node-level fraud predictions and optionally attention weights
        """
        # First GAT layer
        x, attn_weights_1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Second GAT layer
        x, attn_weights_2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Third GAT layer
        x, attn_weights_3 = self.conv3(x, edge_index, return_attention_weights=True)
        x = F.elu(x)

        # Classification head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if return_attention:
            return x, (attn_weights_1, attn_weights_2, attn_weights_3)

        return x

    def predict_fraud(self, x, edge_index):
        """
        Predict fraud probability for each node

        Returns:
            fraud_probs: Probability of fraud for each node
            fraud_labels: Binary fraud prediction
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs = F.softmax(logits, dim=1)
            fraud_probs = probs[:, 1]  # Probability of fraud class
            fraud_labels = (fraud_probs > 0.5).long()

        return fraud_probs, fraud_labels

    def get_attention_weights(self, x, edge_index):
        """
        Extract attention weights to understand which connections are suspicious
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(x, edge_index, return_attention=True)

        return attention_weights


class HeterogeneousGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for multi-type entities
    Handles different entity types (individuals, companies, trusts)
    """

    def __init__(
        self,
        entity_type_features: dict,
        hidden_channels: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()

        self.entity_types = list(entity_type_features.keys())
        self.hidden_channels = hidden_channels

        # Separate encoders for each entity type
        self.type_encoders = nn.ModuleDict(
            {
                entity_type: nn.Linear(num_features, hidden_channels)
                for entity_type, num_features in entity_type_features.items()
            }
        )

        # Graph convolution layers
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x_dict, edge_index):
        """
        Forward pass for heterogeneous graph

        Args:
            x_dict: Dictionary of node features by entity type
            edge_index: Graph connectivity
        """
        # Encode each entity type separately
        encoded_features = []
        for entity_type, features in x_dict.items():
            encoded = self.type_encoders[entity_type](features)
            encoded_features.append(encoded)

        # Combine all entity types
        x = torch.cat(encoded_features, dim=0)

        # Apply graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Classify
        out = self.classifier(x)

        return out


class AnomalyScorer:
    """
    Computes anomaly scores for entities based on GNN embeddings
    and graph structure
    """

    def __init__(self, model: GraphAttentionFraudDetector):
        self.model = model

    def compute_anomaly_scores(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> np.ndarray:
        """
        Compute anomaly scores combining:
        - GNN fraud predictions
        - Graph structural features
        - Temporal patterns
        """
        # Get fraud probabilities from GNN
        fraud_probs, _ = self.model.predict_fraud(x, edge_index)

        # Get attention weights to understand suspicious connections
        # Get attention weights for later analysis
        _, _ = self.model.get_attention_weights(x, edge_index)

        # Compute structural anomaly score
        structural_scores = self._compute_structural_scores(x, edge_index)

        # Combine scores
        fraud_score = fraud_probs.cpu().numpy() * 0.6
        anomaly_scores = fraud_score + structural_scores * 0.4

        return anomaly_scores

    def _compute_structural_scores(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> np.ndarray:
        """
        Compute structural anomaly indicators:
        - Unusual degree distribution
        - Bridge nodes (high betweenness)
        - Isolated clusters
        """
        num_nodes = x.shape[0]

        # Convert to networkx for structural analysis
        import networkx as nx

        G = nx.Graph()
        edges = edge_index.cpu().numpy().T
        G.add_edges_from(edges)

        # Compute structural metrics
        degrees = dict(G.degree())
        betweenness = nx.betweenness_centrality(G)

        # Normalize scores
        max_degree = max(degrees.values()) if degrees else 1
        degree_scores = np.array(
            [degrees.get(i, 0) / max_degree for i in range(num_nodes)]
        )
        betweenness_scores = np.array([betweenness.get(i, 0) for i in range(num_nodes)])

        # Combine structural features
        structural_scores = (degree_scores + betweenness_scores) / 2

        return structural_scores


class FraudDetectionTrainer:
    """
    Training pipeline for GNN fraud detection model
    """

    def __init__(
        self, model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 5e-4
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, data: Data) -> float:
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(data.x, data.edge_index)

        # Compute loss only on training nodes
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, data: Data, mask) -> Tuple[float, float]:
        """Evaluate model performance"""
        self.model.eval()

        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            # Calculate accuracy
            correct = (pred[mask] == data.y[mask]).sum().item()
            total = mask.sum().item()
            accuracy = correct / total if total > 0 else 0

            # Calculate loss
            loss = self.criterion(out[mask], data.y[mask]).item()

        return accuracy, loss

    def train(
        self, data: Data, num_epochs: int = 200, early_stopping_patience: int = 20
    ) -> dict:
        """
        Full training loop with early stopping
        """
        best_val_acc = 0
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(data)

            # Validate
            val_acc, val_loss = self.evaluate(data, data.val_mask)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_gnn_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                    f"Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}"
                )

        # Load best model
        self.model.load_state_dict(torch.load("best_gnn_model.pt"))

        return history


# Example usage
if __name__ == "__main__":
    # Create dummy data
    num_nodes = 100
    num_features = 20
    num_edges = 500

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))  # Binary labels

    # Create masks for train/val/test split
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:60] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[60:80] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[80:] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Initialize model
    model = GraphAttentionFraudDetector(num_node_features=num_features)

    # Train
    trainer = FraudDetectionTrainer(model)
    history = trainer.train(data, num_epochs=50)

    # Predict
    fraud_probs, fraud_labels = model.predict_fraud(data.x, data.edge_index)
    print(f"Detected {fraud_labels.sum().item()} fraudulent entities")
