from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import torch
import networkx as nx
from gnn_fraud_detector import GraphAttentionFraudDetector
from graph_builder import TaxEntityGraph
from data_converter import TransactionToGraphConverter, create_pytorch_geometric_data

app = Flask(__name__)

# Load traditional ML model
model_path = os.path.join("model", "fraud_pipeline.pkl")
ml_model = joblib.load(model_path)

# Load GNN model
gnn_model = None
gnn_data = None
entity_graph = None

try:
    print("Loading GNN model, graph, and data...")
    # This checkpoint was created by re-running train_gnn_fast.py
    checkpoint = torch.load("model/gnn_fraud_model.pt", map_location="cpu")

    gnn_model = GraphAttentionFraudDetector(
        num_node_features=checkpoint["num_features"],
        hidden_channels=checkpoint["hidden_channels"],
    )
    gnn_model.load_state_dict(checkpoint["model_state_dict"])
    gnn_model.eval()

    # Load graph (requires networkx 2.x)
    G = nx.read_gpickle("model/transaction_graph.gpickle")

    # --- THIS IS THE FAST PART ---
    # Load GNN data object directly from checkpoint
    gnn_data = checkpoint["gnn_data"]
    # -----------------------------

    # Pass the model and data to the graph object
    entity_graph = TaxEntityGraph(gnn_model=gnn_model, gnn_data=gnn_data)
    entity_graph.graph = G  # Assign the loaded graph

    print("✅ GNN model, graph, and data loaded successfully")

except Exception as e:
    print(f"⚠️ Could not load GNN model or data: {e}")
    print("    This may be because the model wasn't retrained.")
    print("    Or, you may have a package version mismatch.")
    print("    Falling back to traditional ML and graph metrics only.")

    # Fallback: Load graph but without GNN
    try:
        G = nx.read_gpickle("model/transaction_graph.gpickle")
        entity_graph = TaxEntityGraph(gnn_model=None, gnn_data=None)
        entity_graph.graph = G
        print("✅ Graph loaded for metrics, but GNN is disabled.")
    except Exception as e2:
        print(f"⛔ CRITICAL: Could not load graph at all: {e2}")
        entity_graph = TaxEntityGraph()  # Create empty graph


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Traditional ML prediction"""
    try:
        # Get data from form
        data = {
            "step": float(request.form["step"]),
            "type": request.form["type"],
            "amount": float(request.form["amount"]),
            "oldbalanceOrig": float(request.form["oldbalanceOrg"]),
            "newbalanceOrig": float(request.form["newbalanceOrig"]),
            "oldbalanceDest": float(request.form["oldbalanceDest"]),
            "newbalanceDest": float(request.form["newbalanceDest"]),
        }

        input_df = pd.DataFrame([data])

        # ✅ Apply SAME preprocessing as in training
        type_mapping = {
            "CASH_IN": 1,
            "CASH_OUT": 2,
            "DEBIT": 3,
            "PAYMENT": 4,
            "TRANSFER": 5,
        }
        input_df["type"] = input_df["type"].map(type_mapping)

        # Derived columns
        input_df["errorBalanceOrig"] = (
            input_df["newbalanceOrig"] + input_df["amount"] - input_df["oldbalanceOrig"]
        )
        input_df["errorBalanceDest"] = (
            input_df["oldbalanceDest"] + input_df["amount"] - input_df["newbalanceDest"]
        )

        # ✅ Match feature order from training
        features = [
            "step",
            "type",
            "amount",
            "oldbalanceOrig",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "errorBalanceOrig",
            "errorBalanceDest",
        ]
        input_df = input_df[features]

        # Predict
        prediction = ml_model.predict(input_df)[0]
        probability = ml_model.predict_proba(input_df)[0][1]

        result = {
            "prediction": (
                "🚨 Fraudulent Transaction"
                if prediction == 1
                else "✅ Legitimate Transaction"
            ),
            "confidence": f"{probability * 100:.2f}%",
            "model": "Traditional ML (Logistic Regression)",
        }

        return render_template("results.html", result=result, data=data)

    except Exception as e:
        return render_template(
            "results.html",
            result={"prediction": f"⚠️ Error: {e}"},
            data={},
        )


@app.route("/predict_gnn", methods=["POST"])
def predict_gnn():
    """GNN-based prediction with graph analysis"""
    if gnn_model is None:
        return (
            jsonify(
                {
                    "error": "GNN model not loaded",
                    "message": "Please train the GNN model first",
                }
            ),
            400,
        )

    try:
        data = request.get_json()

        # Extract entity IDs
        orig_entity = data.get("nameOrig", "NEW_CUSTOMER")

        # Check if entities exist in graph
        analysis = {
            "traditional_ml": None,
            "gnn_prediction": None,
            "graph_analysis": {},
            "risk_factors": [],
        }

        # Traditional ML prediction
        ml_data = {
            "step": float(data["step"]),
            "type": data["type"],
            "amount": float(data["amount"]),
            "oldbalanceOrig": float(data["oldbalanceOrg"]),
            "newbalanceOrig": float(data["newbalanceOrig"]),
            "oldbalanceDest": float(data["oldbalanceDest"]),
            "newbalanceDest": float(data["newbalanceDest"]),
        }
        input_df = pd.DataFrame([ml_data])
        ml_pred = ml_model.predict_proba(input_df)[0][1]
        analysis["traditional_ml"] = {
            "fraud_probability": float(ml_pred),
            "prediction": "Fraud" if ml_pred > 0.5 else "Legitimate",
        }

        # Graph-based analysis
        if entity_graph and entity_graph.graph.has_node(orig_entity):
            # Origin entity analysis
            orig_node = entity_graph.graph.nodes[orig_entity]
            analysis["graph_analysis"]["origin"] = {
                "entity_type": orig_node.get("entity_type", "unknown"),
                "total_transactions": orig_node.get("transaction_count", 0),
                "total_sent": orig_node.get("total_sent", 0),
                "connections": entity_graph.graph.degree(orig_entity),
            }

            # Risk factors
            if orig_node.get("transaction_count", 0) > 100:
                analysis["risk_factors"].append("High transaction volume")

            if entity_graph.graph.degree(orig_entity) > 50:
                analysis["risk_factors"].append("Unusually many connections")

        # Detect circular transactions
        if entity_graph:
            cycles = entity_graph.detect_circular_transactions(max_cycle_length=5)
            if len(cycles) > 0:
                analysis["graph_analysis"]["circular_patterns"] = len(cycles)
                analysis["risk_factors"].append(
                    f"Part of {len(cycles)} circular transaction patterns"
                )

        # Combined risk score
        risk_score = ml_pred * 0.6  # Weight traditional ML
        if len(analysis["risk_factors"]) > 0:
            risk_score += 0.2 * len(analysis["risk_factors"])

        analysis["combined_risk_score"] = min(risk_score, 1.0)
        analysis["final_prediction"] = (
            "Fraudulent" if risk_score > 0.5 else "Legitimate"
        )

        return jsonify(analysis)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/graph/circular_transactions", methods=["GET"])
def get_circular_transactions():
    """Detect circular transaction patterns"""
    if entity_graph is None:
        return jsonify({"error": "Graph not loaded"}), 400

    try:
        cycles = entity_graph.detect_circular_transactions(max_cycle_length=5)
        print(f"DEBUG: Found {len(cycles)} circular transactions.")
        # Format for response
        result = {
            "total_cycles": len(cycles),
            "suspicious_cycles": [
                {
                    "length": cycle["length"],
                    "total_amount": cycle["total_amount"],
                    "suspicion_score": cycle["score"],
                    "entities": cycle["cycle"],  # renamed key
                }
                for cycle in cycles[:20]
            ],
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/graph/shell_companies", methods=["GET"])
def get_shell_companies():
    """Identify potential suspicious merchants (acting as shell companies)"""
    if entity_graph is None:
        return jsonify({"error": "Graph not loaded"}), 400

    try:
        # 1. Call the function from graph_builder.py
        merchants = entity_graph.detect_suspicious_merchants(threshold=0.2)

        # DEBUGGING: Print the first item to confirm structure
        print(f"DEBUG: Found {len(merchants)} merchants with score > 0.2")
        if len(merchants) > 0:
            print(f"DEBUG: Top merchant details: {merchants[0]}")

        # 2. Format the data to match what the dashboard.html expects
        result = {
            "total_suspects": len(merchants),
            "shell_networks": [  # dashboard.html expects this key
                {
                    "entity_id": merchant["entity_id"],
                    # Read the score from *inside* the indicators
                    "shell_score": merchant["indicators"]["suspicion_score"],
                    "indicators": merchant["indicators"],  # Pass the whole dict
                }
                for merchant in merchants[:20]  # Get top 20
            ],
        }
        return jsonify(result)

    except Exception as e:
        print(f"Error in /graph/shell_companies: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/graph/risk_scores", methods=["GET"])
def get_risk_scores():
    """Get risk scores for all entities"""
    if entity_graph is None:
        return jsonify({"error": "Graph not loaded"}), 400

    try:
        risk_scores = entity_graph.compute_entity_risk_scores()

        # Get high risk entities (top 50)
        sorted_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[
            :50
        ]

        result = {
            "high_risk_entities": [
                {"entity_id": entity, "risk_score": score}
                for entity, score in sorted_risks
            ]
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard"""
    return render_template("dashboard.html")


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get overall system statistics"""
    stats = {
        "models_loaded": {
            "traditional_ml": ml_model is not None,
            "gnn": gnn_model is not None,
            "graph": entity_graph is not None,
        }
    }

    if entity_graph and entity_graph.graph:
        G = entity_graph.graph
        stats["graph_stats"] = {
            "total_entities": G.number_of_nodes(),
            "total_transactions": G.number_of_edges(),
            "fraud_cases": sum(
                1 for n in G.nodes() if G.nodes[n].get("is_fraudster", 0) == 1
            ),
        }

    return jsonify(stats)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DYNAMIC TAX FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  - /                          : Main prediction interface")
    print("  - /dashboard                 : Analytics dashboard")
    print("  - /predict                   : Traditional ML prediction")
    print("  - /predict_gnn               : GNN-based prediction (POST JSON)")
    print("  - /graph/circular_transactions : Detect circular patterns")
    print("  - /graph/shell_companies     : Identify shell companies")
    print("  - /graph/risk_scores         : Entity risk scores")
    print("  - /api/stats                 : System statistics")
    print("\n" + "=" * 60 + "\n")

    app.run(debug=True, port=5002)
