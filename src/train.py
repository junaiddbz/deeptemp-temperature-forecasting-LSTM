"""
DeepTemp: Training Pipeline (Phase 3)
Module: src/train.py

Handles:
1. Training loop for StackedLSTM using Adam optimizer and MSE/Huber loss.
2. Learning rate scheduling (ReduceLROnPlateau) and early stopping.
3. Checkpoint persistence (weights, hyperparameter config, and scaler parameters).
4. Training and evaluation metrics tracking (MSE, RMSE, MAE).
5. Loss curve visualization export.
"""

import os
import argparse
import time
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import StackedLSTM
from data_loader import DataPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepTemp Stacked LSTM for Climate Forecasting")
    parser.add_argument("--city", type=str, default="islamabad", choices=["islamabad", "lahore", "karachi"],
                        help="Target city for climate modeling")
    parser.add_argument("--features", nargs="+", default=["tavg"],
                        help="Features to use (default: ['tavg'])")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size (default: 32)")
    parser.add_argument("--window_size", type=int, default=30,
                        help="Sliding window size in days (default: 30)")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Hidden dimensions in LSTM units (default: 64)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of stacked LSTM layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability (default: 0.2)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial Adam learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="L2 weight decay regularization (default: 1e-5)")
    parser.add_argument("--loss_fn", type=str, default="mse", choices=["mse", "huber"],
                        help="Loss function: 'mse' or 'huber' (default: 'mse')")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience in epochs (default: 10)")
    parser.add_argument("--split_date", type=str, default="2019-01-01",
                        help="Date boundary between train and test (default: 2019-01-01)")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints (default: 'checkpoints')")
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Trains model for one full epoch."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in recurrent networks
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluates model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item() * len(y_batch)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.concatenate(all_preds), np.concatenate(all_targets)


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    config: Dict[str, Any],
    scaler: Any,
    metrics: Dict[str, float]
):
    """Saves checkpoint weights, architecture config, and scaler parameters."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "scaler_min_": scaler.min_,
        "scaler_scale_": scaler.scale_,
        "scaler_data_min_": scaler.data_min_,
        "scaler_data_max_": scaler.data_max_,
        "metrics": metrics
    }
    torch.save(checkpoint, save_path)
    print(f"[*] Checkpoint saved to: {save_path}")


def train_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting DeepTemp Training Pipeline ===")
    print(f"Target City:       {args.city.upper()}")
    print(f"Features:          {args.features}")
    print(f"Compute Device:    {device}")
    print(f"Window Size:       {args.window_size} days")
    print(f"Batch Size:        {args.batch_size}")
    print(f"LSTM Config:       {args.num_layers} layers x {args.hidden_size} units (Dropout: {args.dropout})")
    print(f"Optimizer:         Adam (lr={args.lr})")
    print(f"Loss Function:     {args.loss_fn.upper()}")

    # 1. Prepare Data
    preprocessor = DataPreprocessor()
    data_bundle = preprocessor.prepare_dataset_for_city(
        city=args.city,
        features=args.features,
        window_size=args.window_size,
        forecast_horizon=1,
        split_date=args.split_date,
        batch_size=args.batch_size
    )

    train_loader = data_bundle["train_loader"]
    test_loader = data_bundle["test_loader"]
    scaler = data_bundle["scaler"]

    print(f"\n[+] Data ready: {len(train_loader.dataset)} train samples, {len(test_loader.dataset)} test samples.")

    # 2. Initialize Model
    model = StackedLSTM(
        input_size=len(args.features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=1,
        dropout=args.dropout
    ).to(device)

    print(f"[+] Model instantiated with {model.get_parameter_count():,} trainable parameters.")

    # 3. Loss & Optimizer
    if args.loss_fn == "huber":
        criterion = nn.HuberLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # 4. Training Loop
    best_test_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "test_loss": []}
    checkpoint_file = os.path.join(args.save_dir, f"{args.city}_lstm_best.pt")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, preds_scaled, targets_scaled = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        scheduler.step(test_loss)

        # Calculate unscaled metric on test set (in degrees Celsius)
        # scaler.data_min_[0] and scaler.data_max_[0]
        temp_range = scaler.data_max_[0] - scaler.data_min_[0]
        unscaled_rmse = np.sqrt(np.mean(((preds_scaled - targets_scaled) * temp_range) ** 2))
        unscaled_mae = np.mean(np.abs((preds_scaled - targets_scaled) * temp_range))

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f} | "
            f"Test RMSE: {unscaled_rmse:.2f}°C | Test MAE: {unscaled_mae:.2f}°C"
        )

        # Checkpoint if best test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            config = vars(args)
            metrics = {
                "best_test_loss": best_test_loss,
                "best_rmse_celsius": float(unscaled_rmse),
                "best_mae_celsius": float(unscaled_mae),
                "epoch": epoch
            }
            save_checkpoint(checkpoint_file, model, config, scaler, metrics)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[*] Early stopping triggered after {epoch} epochs.")
                break

    elapsed = time.time() - start_time
    print(f"\n[+] Training completed in {elapsed:.1f}s.")
    print(f"    Best Test Loss: {best_test_loss:.5f}")

    # 5. Plot Loss Curves
    os.makedirs("reports", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
    ax.plot(range(1, len(history["test_loss"]) + 1), history["test_loss"], label="Test Loss", color="#dc2626", linewidth=2)
    ax.set_title(f"DeepTemp LSTM Training & Validation Loss ({args.city.title()})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(f"Loss ({args.loss_fn.upper()})", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()

    loss_plot_path = os.path.join("reports", f"loss_curve_{args.city}.png")
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)
    print(f"[*] Loss curve exported to: {loss_plot_path}")

    return checkpoint_file


if __name__ == "__main__":
    cli_args = parse_args()
    train_pipeline(cli_args)
