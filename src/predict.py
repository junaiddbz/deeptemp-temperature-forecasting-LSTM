"""
DeepTemp: Inference and Prediction Pipeline (Phase 3)
Module: src/predict.py

Handles:
1. Loading trained PyTorch checkpoints and model architectures.
2. Restoring scaler parameters to inversely transform predictions back to °C.
3. Generating forecasts on unseen test sequences or arbitrary input CSVs.
4. Exporting Actual vs. Predicted comparison plots and computing RMSE, MAE, R2.
"""

import os
import argparse
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch

from model import StackedLSTM
from data_loader import DataPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepTemp LSTM Inference")
    parser.add_argument("--city", type=str, default="islamabad", choices=["islamabad", "lahore", "karachi"],
                        help="Target city to predict")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint file (defaults to checkpoints/{city}_lstm_best.pt)")
    parser.add_argument("--input", type=str, default=None,
                        help="Optional custom CSV input file for inference")
    parser.add_argument("--output_plot", type=str, default=None,
                        help="Destination path for actual vs predicted plot")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[StackedLSTM, MinMaxScaler, Dict[str, Any]]:
    """Loads model weights, hyperparameter configuration, and scaler state."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Reconstruct model
    model = StackedLSTM(
        input_size=len(config["features"]),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=1,
        dropout=config["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Reconstruct fitted scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.min_ = checkpoint["scaler_min_"]
    scaler.scale_ = checkpoint["scaler_scale_"]
    scaler.data_min_ = checkpoint["scaler_data_min_"]
    scaler.data_max_ = checkpoint["scaler_data_max_"]
    scaler.n_features_in_ = len(config["features"])

    return model, scaler, config


def predict_test_set(
    city: str,
    checkpoint_path: Optional[str] = None,
    output_plot: Optional[str] = None
):
    """
    Evaluates the model on the full unseen test set, computes metrics,
    and produces a publication-grade Actual vs. Predicted graph.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is None:
        checkpoint_path = os.path.join("checkpoints", f"{city}_lstm_best.pt")

    print(f"=== Running Inference on Test Set ({city.upper()}) ===")
    print(f"Loading checkpoint: {checkpoint_path}")
    model, scaler, config = load_model_from_checkpoint(checkpoint_path, device)

    # Prepare test data with exact window size from training config
    preprocessor = DataPreprocessor()
    data_bundle = preprocessor.prepare_dataset_for_city(
        city=city,
        features=config["features"],
        window_size=config["window_size"],
        forecast_horizon=1,
        split_date=config.get("split_date", "2019-01-01")
    )

    X_test = torch.tensor(data_bundle["X_test"], dtype=torch.float32).to(device)
    y_test_scaled = data_bundle["y_test"]
    test_df = data_bundle["test_df"]
    window_size = config["window_size"]

    # Corresponding dates for predictions (aligned after the sliding window)
    prediction_dates = test_df.index[window_size:]

    # Model inference
    with torch.no_grad():
        preds_scaled = model(X_test).cpu().numpy()

    # Inverse transform to actual degrees Celsius (°C)
    # y = (scaled - min) / scale
    preds_celsius = (preds_scaled - scaler.min_[0]) / scaler.scale_[0]
    actual_celsius = (y_test_scaled - scaler.min_[0]) / scaler.scale_[0]

    # Metrics
    mse = mean_squared_error(actual_celsius, preds_celsius)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_celsius, preds_celsius)
    r2 = r2_score(actual_celsius, preds_celsius)

    print("\n=== Model Performance Evaluation (Unseen Test Data) ===")
    print(f"  Test Period:      {prediction_dates.min().strftime('%Y-%m-%d')} to {prediction_dates.max().strftime('%Y-%m-%d')}")
    print(f"  Total Predictions: {len(preds_celsius):,}")
    print(f"  Mean Squared Error (MSE):       {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}°C")
    print(f"  Mean Absolute Error (MAE):      {mae:.2f}°C")
    print(f"  R-squared (R² Score):           {r2:.4f}")

    # Plot Actual vs Predicted
    os.makedirs("reports", exist_ok=True)
    if output_plot is None:
        output_plot = os.path.join("reports", f"{city}_forecast_vs_actual.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top plot: 5-year timeline
    ax1.plot(prediction_dates, actual_celsius, label="Actual Temperature", color="#0284c7", alpha=0.75, linewidth=1.2)
    ax1.plot(prediction_dates, preds_celsius, label="LSTM Forecast", color="#dc2626", alpha=0.85, linewidth=1.2, linestyle="--")
    ax1.set_title(f"DeepTemp LSTM Forecast vs. Actual Temperature ({city.title()}) [2019-2023]\nRMSE: {rmse:.2f}°C | MAE: {mae:.2f}°C | R²: {r2:.4f}",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Temperature (°C)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Bottom plot: Residual error (Actual - Predicted)
    residuals = actual_celsius - preds_celsius
    ax2.plot(prediction_dates, residuals, color="#475569", linewidth=0.8, alpha=0.7)
    ax2.axhline(0, color="#ef4444", linestyle="--", linewidth=1.2)
    ax2.fill_between(prediction_dates, -rmse, rmse, color="#fde047", alpha=0.25, label=f"±1 RMSE ({rmse:.2f}°C)")
    ax2.set_ylabel("Error (°C)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)
    print(f"[*] Evaluation plot saved to: {output_plot}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "plot_path": output_plot
    }


if __name__ == "__main__":
    cli_args = parse_args()
    predict_test_set(city=cli_args.city, checkpoint_path=cli_args.checkpoint, output_plot=cli_args.output_plot)
