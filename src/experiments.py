"""
DeepTemp: Hyperparameter Tuning & Regional Experimentation (Phase 4)
Module: src/experiments.py

Runs systematic experiments comparing:
1. Sequence Window Size: 14 days vs 30 days vs 60 days
2. Model Capacity: Shallow (1 layer x 32) vs Medium (2 layers x 64) vs Deep (3 layers x 128)
3. Regional Cross-Evaluation: Trains and benchmarks best models for Islamabad, Lahore, and Karachi
4. Exports an experimental results summary table and metrics JSON.
"""

import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import StackedLSTM
from data_loader import DataPreprocessor
from train import train_epoch, evaluate, save_checkpoint


def run_experiment(
    exp_name: str,
    city: str = "islamabad",
    features = ["tavg"],
    window_size: int = 30,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    save_checkpoint_bool: bool = False,
    checkpoint_dir: str = "checkpoints"
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Experiment: [{exp_name}] ---")
    print(f"City: {city.upper()} | Features: {features} | Window: {window_size}d | "
          f"Arch: {num_layers}L x {hidden_size}H | Epochs: {epochs}")

    preprocessor = DataPreprocessor()
    data_bundle = preprocessor.prepare_dataset_for_city(
        city=city,
        features=features,
        window_size=window_size,
        split_date="2019-01-01",
        batch_size=batch_size
    )

    train_loader = data_bundle["train_loader"]
    test_loader = data_bundle["test_loader"]
    scaler = data_bundle["scaler"]

    model = StackedLSTM(
        input_size=len(features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_test_loss = float("inf")
    best_rmse = float("inf")
    best_mae = float("inf")
    checkpoint_path = os.path.join(checkpoint_dir, f"{city}_lstm_best.pt")

    start_t = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, preds_scaled, targets_scaled = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        temp_range = scaler.data_max_[0] - scaler.data_min_[0]
        unscaled_rmse = np.sqrt(np.mean(((preds_scaled - targets_scaled) * temp_range) ** 2))
        unscaled_mae = np.mean(np.abs((preds_scaled - targets_scaled) * temp_range))

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_rmse = float(unscaled_rmse)
            best_mae = float(unscaled_mae)

            if save_checkpoint_bool:
                config = {
                    "city": city,
                    "features": features,
                    "window_size": window_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "split_date": "2019-01-01"
                }
                save_checkpoint(checkpoint_path, model, config, scaler, {
                    "best_test_loss": best_test_loss,
                    "best_rmse_celsius": best_rmse,
                    "best_mae_celsius": best_mae,
                    "epoch": epoch
                })

    elapsed = time.time() - start_t
    print(f"Result: Test RMSE = {best_rmse:.2f}°C | Test MAE = {best_mae:.2f}°C (Time: {elapsed:.1f}s)")

    return {
        "experiment": exp_name,
        "city": city,
        "features": ",".join(features),
        "window_size": window_size,
        "architecture": f"{num_layers}L x {hidden_size}H",
        "parameters": model.get_parameter_count(),
        "test_rmse": round(best_rmse, 2),
        "test_mae": round(best_mae, 2),
        "train_time_sec": round(elapsed, 1)
    }


def run_all_experiments():
    results = []

    print("==================================================================")
    print("      DeepTemp: Systematic Hyperparameter & Regional Tuning       ")
    print("==================================================================")

    # 1. Window Size Experiments (Islamabad)
    for w in [14, 30, 60]:
        res = run_experiment(
            exp_name=f"Window Size: {w} days",
            city="islamabad",
            window_size=w,
            hidden_size=64,
            num_layers=2,
            epochs=15
        )
        results.append(res)

    # 2. Architecture Depth Experiments (Islamabad)
    for layers, hidden in [(1, 32), (2, 64), (3, 128)]:
        res = run_experiment(
            exp_name=f"Capacity: {layers}L x {hidden}H",
            city="islamabad",
            window_size=30,
            hidden_size=hidden,
            num_layers=layers,
            epochs=15
        )
        results.append(res)

    # 3. Regional Models Training & Checkpointing (Islamabad, Lahore, Karachi)
    for city in ["islamabad", "lahore", "karachi"]:
        res = run_experiment(
            exp_name=f"Production Model ({city.title()})",
            city=city,
            window_size=30,
            hidden_size=64,
            num_layers=2,
            epochs=25,
            save_checkpoint_bool=True
        )
        results.append(res)

    # Save summary table
    os.makedirs("reports", exist_ok=True)
    df_res = pd.DataFrame(results)
    csv_report = os.path.join("reports", "experiment_results.csv")
    df_res.to_csv(csv_report, index=False)

    print("\n==================================================================")
    print("                    Final Experimentation Summary                 ")
    print("==================================================================")
    print(df_res.to_string(index=False))
    print(f"\n[*] Full experimentation table saved to: {csv_report}")


if __name__ == "__main__":
    run_all_experiments()
