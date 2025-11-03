import torch
import pandas as pd
import numpy as np
from data_converter import TransactionToGraphConverter, create_pytorch_geometric_data
from gnn_fraud_detector import GraphAttentionFraudDetector, FraudDetectionTrainer
import matplotlib.pyplot as plt
import os


def train_complete_pipeline():
    """
    Complete training pipeline:
    1. Load transaction data
    2. Convert to graph
    3. Train GNN model
    4. Save model and graph
    """

    print("=" * 60)
    print("GRAPH-BASED TAX FRAUD DETECTION - TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Load transaction data
    print("\n[1/5] Loading transaction data...")
    # UPDATE THIS PATH TO YOUR DATA FILE
    df = pd.read_csv("/Users/Parizaa_1/Downloads/fraud_detection.csv")

    # Use a subset for faster training (optional)
    # Uncomment next line to use full dataset
    # df = df.sample(n=100000, random_state=42)

    print(f"Loaded {len(df)} transactions")
    fraud_percent = df["isFraud"].mean() * 100
    print(f"Fraud cases: {df['isFraud'].sum()} ({fraud_percent:.2f}%)")

    # Step 2: Convert to graph
    print("\n[2/5] Converting transactions to graph structure...")
    converter = TransactionToGraphConverter()
    G, entity_mapping = converter.convert_dataset(df)

    # Save graph
    os.makedirs("model", exist_ok=True)
    converter.save_graph(G, "model/transaction_graph.gpickle")

    # Step 3: Create PyTorch Geometric data
    print("\n[3/5] Creating PyTorch Geometric dataset...")
    data = create_pytorch_geometric_data(G, converter)

    print("\nDataset Statistics:")
    print(f"  Total nodes: {data.x.shape[0]}")
    print(f"  Node features: {data.x.shape[1]}")
    print(f"  Total edges: {data.edge_index.shape[1]}")
    print(f"  Training nodes: {data.train_mask.sum()}")
    print(f"  Validation nodes: {data.val_mask.sum()}")
    print(f"  Test nodes: {data.test_mask.sum()}")
    print(f"  Fraud nodes (total): {data.y.sum()}")

    # Handle class imbalance
    fraud_ratio = data.y[data.train_mask].float().mean()
    print(f"  Fraud ratio in training: {fraud_ratio:.4f}")

    # Step 4: Initialize and train model
    print("\n[4/5] Training Graph Neural Network...")

    # Model parameters
    num_features = data.x.shape[1]
    hidden_channels = 64
    num_classes = 2

    model = GraphAttentionFraudDetector(
        num_node_features=num_features,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        num_heads=4,
        dropout=0.3,
    )

    # Use weighted loss for imbalanced data
    fraud_weight = (1 - fraud_ratio) / fraud_ratio
    class_weights = torch.tensor([1.0, fraud_weight])

    trainer = FraudDetectionTrainer(model=model, learning_rate=0.001, weight_decay=5e-4)

    # Update criterion to use class weights
    trainer.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Train model
    print("\nStarting training...")
    history = trainer.train(data=data, num_epochs=200, early_stopping_patience=20)

    # Step 5: Evaluate and save
    print("\n[5/5] Evaluating model...")

    # Test set evaluation
    test_acc, test_loss = trainer.evaluate(data, data.test_mask)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Detailed metrics
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        # Test set metrics
        test_pred = pred[data.test_mask]
        test_true = data.y[data.test_mask]

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(test_true.cpu(), test_pred.cpu())

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(
            classification_report(
                test_true.cpu(), test_pred.cpu(), target_names=["Legitimate", "Fraud"]
            )
        )

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_features": num_features,
            "hidden_channels": hidden_channels,
            "entity_mapping": entity_mapping,
            "history": history,
        },
        "model/gnn_fraud_model.pt",
    )

    print("\n✅ Model saved to: model/gnn_fraud_model.pt")

    # Plot training history
    plot_training_history(history)

    return model, data, history


def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history["val_accuracy"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("model/training_history.png", dpi=150, bbox_inches="tight")
    print("Training history plot saved to: model/training_history.png")
    plt.show()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Train the model
    model, data, history = train_complete_pipeline()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python app.py' to start the web interface")
    print("2. The system now uses both traditional ML and GNN models")
    print("3. Graph analysis features are available via API")
