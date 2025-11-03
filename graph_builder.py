import networkx as nx
import numpy as np
from typing import Dict, List
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from gnn_fraud_detector import GraphAttentionFraudDetector


class TaxEntityGraph:
    def __init__(
        self, gnn_model: GraphAttentionFraudDetector = None, gnn_data: Data = None
    ):
        self.graph = nx.DiGraph()
        self.entity_features = {}
        self.transaction_history = []
        self.gnn_model = gnn_model
        self.gnn_data = gnn_data
        self.risk_scores_cache = None

    def add_entity(self, entity_id: str, entity_type: str, features: Dict):
        """
        Add a tax entity (individual, company, shell company) as a node

        Args:
            entity_id: Unique identifier
            entity_type: 'individual', 'company', 'shell_company', 'trust'
            features: Dict of entity attributes (income, deductions, etc.)
        """
        self.graph.add_node(entity_id, entity_type=entity_type, **features)
        self.entity_features[entity_id] = features

    def add_transaction(
        self,
        from_id: str,
        to_id: str,
        transaction_type: str,
        amount: float,
        timestamp: str,
        metadata: Dict = None,
    ):
        """
        Add a transaction as a directed edge between entities

        Args:
            from_id: Source entity
            to_id: Destination entity
            transaction_type: 'payment', 'invoice', 'transfer', 'dividend'
            amount: Transaction amount
            timestamp: ISO format timestamp
            metadata: Additional transaction details
        """
        edge_data = {
            "transaction_type": transaction_type,
            "amount": amount,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }

        if self.graph.has_edge(from_id, to_id):
            # Update existing edge with new transaction
            existing = self.graph[from_id][to_id]
            existing["total_amount"] = existing.get("total_amount", 0) + amount
            curr_count = existing.get("transaction_count", 0)
            existing["transaction_count"] = curr_count + 1
            curr_trans = existing.get("transactions", [])
            existing["transactions"] = curr_trans + [edge_data]
        else:
            self.graph.add_edge(
                from_id,
                to_id,
                transaction_type=transaction_type,
                amount=amount,
                total_amount=amount,
                transaction_count=1,
                timestamp=timestamp,
                transactions=[edge_data],
                metadata=metadata or {},
            )

        transaction = {"from": from_id, "to": to_id, **edge_data}
        self.transaction_history.append(transaction)

    def detect_circular_transactions(self, max_cycle_length=10) -> List[Dict]:
        """
        Detect potential circular fraud rings using Strongly Connected Components (SCCs).
        An SCC is a group of nodes where every node is reachable from every other node.
        """
        fraud_rings = []

        # Use networkx to find all SCCs
        # We only care about components with 2 or more nodes (a "cycle")
        scc_generator = (
            comp
            for comp in nx.strongly_connected_components(self.graph)
            if len(comp) > 1
        )

        for component_nodes in scc_generator:
            # We only care about small, tight-knit rings
            if len(component_nodes) > max_cycle_length:
                continue

            # Create a subgraph of just this component
            component_graph = self.graph.subgraph(component_nodes)

            total_amount = 0
            edge_count = 0

            # Calculate total transaction amount *within* the ring
            for u, v, data in component_graph.edges(data=True):
                total_amount += data.get("amount", 0)
                edge_count += 1

            # We don't want "cliques" where every node just links to every other
            # We want "cycles" where nodes = edges
            if edge_count == 0:
                continue

            # Score the ring
            # A good score = high total amount, and a "pure" cycle (nodes ~= edges)
            length = len(component_nodes)
            purity_score = abs(length - edge_count) + 1  # +1 to avoid div by zero
            score = (total_amount / 1000) / purity_score

            # Get the entity names
            cycle_names = [
                self.graph.nodes[node].get("name", str(node))
                for node in component_nodes
            ]

            fraud_rings.append(
                {
                    "cycle": cycle_names,
                    "score": score,
                    "total_amount": total_amount,
                    "length": length,
                    "edges": edge_count,
                }
            )

        # Return the highest-scoring rings first
        return sorted(fraud_rings, key=lambda x: x["score"], reverse=True)

    def _calculate_cycle_amount(self, cycle: List[str]) -> float:
        """Calculate total transaction amount in a cycle"""
        total = 0
        for i in range(len(cycle)):
            from_node = cycle[i]
            to_node = cycle[(i + 1) % len(cycle)]
            if self.graph.has_edge(from_node, to_node):
                total += self.graph[from_node][to_node].get("total_amount", 0)
        return total

    def detect_suspicious_merchants(self, threshold: float = 0.2) -> List[Dict]:
        """
        Identify suspicious merchants based on graph features.
        """
        suspicious_merchants = []

        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]

            if node_data.get("entity_type") == "merchant":
                # This helper function *guarantees* all keys are present
                indicators = self._calculate_merchant_suspicion(node)

                # Use the score from the indicators
                score = indicators["suspicion_score"]

                if score > threshold:
                    suspicious_merchants.append(
                        {
                            "entity_id": node_data.get("name", str(node)),
                            "indicators": indicators,  # Put the whole dict here
                        }
                    )

        # Sort by the score *inside* the indicators dict
        return sorted(
            suspicious_merchants,
            key=lambda x: x["indicators"]["suspicion_score"],
            reverse=True,
        )

    def _calculate_merchant_suspicion(self, entity_id: str) -> Dict:
        """
        Calculate merchant suspicion score based on available graph features.
        This function *always* returns a dict with all keys.
        """
        node_data = self.graph.nodes[entity_id]

        # Get features, default to 0
        total_sent = node_data.get("total_sent", 0)
        total_received = node_data.get("total_received", 0)
        in_degree = node_data.get("in_degree", 0)
        out_degree = node_data.get("out_degree", 0)

        total_flow = total_sent + total_received

        # Use 1e-6 for safe division
        safe_total_flow = total_flow + 1e-6
        imbalance = abs(total_sent - total_received) / safe_total_flow

        suspicion_score = 0

        if imbalance > 0.90:
            suspicion_score += 0.4
        if (in_degree + out_degree) > 15:
            suspicion_score += 0.2
        # Only penalize if flow is small but not zero (avoids inactive merchants)
        if total_flow < 1000 and total_flow > 0:
            suspicion_score += 0.1
        if in_degree > 5 and out_degree < 2:
            suspicion_score += 0.3

        transaction_ratio = total_sent / (total_received + 1e-6)
        return {
            "suspicion_score": min(suspicion_score, 1.0),
            "flow_imbalance": imbalance,
            "connection_count": in_degree + out_degree,
            "total_flow": total_flow,
            "transaction_ratio": transaction_ratio,
        }

    def compute_entity_risk_scores(self) -> Dict[str, float]:
        """
        Compute risk scores using the GNN model if available.
        Falls back to graph metrics if GNN is not present.
        """
        # Check cache first
        if self.risk_scores_cache:
            return self.risk_scores_cache

        if self.gnn_model and self.gnn_data:
            print("Computing risk scores using GNN model...")
            try:
                # Use the GNN model to predict fraud probabilities
                self.gnn_model.eval()
                with torch.no_grad():
                    logits = self.gnn_model(
                        self.gnn_data.x.to("cpu"), self.gnn_data.edge_index.to("cpu")
                    )
                    probs = F.softmax(logits, dim=1)
                    fraud_probs = probs[:, 1]  # Probability of fraud (class 1)

                risk_scores = {}
                # Use the graph's node list, which matches the order of gnn_data.x
                node_list = list(self.graph.nodes())

                for i, node_id in enumerate(node_list):
                    # Map the integer node ID back to the real entity name
                    entity_name = self.graph.nodes[node_id].get("name", str(node_id))
                    risk_scores[entity_name] = float(fraud_probs[i].item())

                self.risk_scores_cache = risk_scores  # Save to cache
                return risk_scores

            except Exception as e:
                print(f"⚠️ GNN risk score computation failed: {e}")
                # Fallback to non-GNN method if GNN fails
                return self._compute_graph_metric_scores()

        else:
            print("Computing risk scores using graph metrics (GNN not loaded)...")
            return self._compute_graph_metric_scores()

    def _compute_graph_metric_scores(self) -> Dict[str, float]:
        """
        Original fallback method using only graph metrics.
        """
        risk_scores = {}
        # Betweenness centrality (entities that bridge different groups)
        betweenness = nx.betweenness_centrality(self.graph)

        # PageRank (entities with many important connections)
        pagerank = nx.pagerank(self.graph)

        # Clustering coefficient (how connected an entity's neighbors are)
        clustering = nx.clustering(self.graph.to_undirected())

        for node in self.graph.nodes():
            # Combine metrics into risk score
            risk_score = (
                betweenness.get(node, 0) * 0.3
                + pagerank.get(node, 0) * 100 * 0.4
                + (1 - clustering.get(node, 0)) * 0.3
            )
            # Get entity name from node data
            entity_name = self.graph.nodes[node].get("name", str(node))
            risk_scores[entity_name] = risk_score

        return risk_scores

    def export_graph_for_gnn(self, output_path: str = "graph_data.json"):
        """
        Export graph in format suitable for GNN processing
        """
        graph_data = {"nodes": [], "edges": []}

        # Export nodes with features
        for node in self.graph.nodes():
            node_data = dict(self.graph.nodes[node])
            node_data["id"] = node
            graph_data["nodes"].append(node_data)

        # Export edges with features
        for u, v, edge_data in self.graph.edges(data=True):
            edge_export = {
                "source": u,
                "target": v,
                **{k: v for k, v in edge_data.items() if k != "transactions"},
            }
            graph_data["edges"].append(edge_export)

        with open(output_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        return graph_data

    def visualize_subgraph(
        self, entity_ids: List[str], output_path: str = "fraud_network.html"
    ):
        """
        Create interactive visualization of a subgraph (fraud network)
        """
        subgraph = self.graph.subgraph(entity_ids)

        # This would integrate with visualization libraries
        # For now, return the subgraph structure
        return subgraph


# Example usage
if __name__ == "__main__":
    # Create graph
    graph = TaxEntityGraph()

    # Add entities
    graph.add_entity(
        "COMPANY_001",
        "company",
        {
            "reported_income": 1000000,
            "deductions": 900000,
            "employee_count": 2,
            "years_active": 1,
        },
    )

    graph.add_entity(
        "COMPANY_002",
        "shell_company",
        {
            "reported_income": 500000,
            "deductions": 450000,
            "employee_count": 0,
            "years_active": 1,
        },
    )

    graph.add_entity(
        "INDIVIDUAL_001", "individual", {"reported_income": 50000, "deductions": 45000}
    )

    # Add transactions (creating a circular pattern)
    graph.add_transaction("COMPANY_001", "COMPANY_002", "payment", 800000, "2024-01-15")
    graph.add_transaction(
        "COMPANY_002", "INDIVIDUAL_001", "dividend", 400000, "2024-02-01"
    )
    graph.add_transaction(
        "INDIVIDUAL_001", "COMPANY_001", "investment", 350000, "2024-03-01"
    )

    # Detect fraud patterns
    cycles = graph.detect_circular_transactions()
    print("Circular transactions detected:", len(cycles))

    shells = graph.detect_shell_company_networks()
    print("Shell company networks:", len(shells))

    risks = graph.compute_entity_risk_scores()
    print(
        "Top 5 risky entities:",
        sorted(risks.items(), key=lambda x: x[1], reverse=True)[:5],
    )
