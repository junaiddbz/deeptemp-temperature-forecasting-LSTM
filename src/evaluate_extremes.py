"""
DeepTemp: Extreme Weather Evaluation & Business Insights (Phase 5)
Module: src/evaluate_extremes.py

Analyzes model resilience during high-impact climatic extremes:
1. Extreme Summer Heatwaves (T >= 35°C in Punjab/Islamabad, T >= 33°C in Karachi)
2. Severe Winter Cold Drops (T <= 8°C in Islamabad/Northern Punjab)
3. Computes extreme-regime RMSE/MAE and generates focused comparative plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import StackedLSTM
from data_loader import DataPreprocessor
from predict import load_model_from_checkpoint


def evaluate_city_extremes(city: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, config = load_model_from_checkpoint(checkpoint_path, device)

    preprocessor = DataPreprocessor()
    bundle = preprocessor.prepare_dataset_for_city(
        city=city,
        features=config["features"],
        window_size=config["window_size"],
        split_date=config.get("split_date", "2019-01-01")
    )

    X_test = torch.tensor(bundle["X_test"], dtype=torch.float32).to(device)
    y_test_scaled = bundle["y_test"]
    test_df = bundle["test_df"]
    window_size = config["window_size"]
    dates = test_df.index[window_size:]

    with torch.no_grad():
        preds_scaled = model(X_test).cpu().numpy()

    # Inverse transform
    actual = (y_test_scaled - scaler.min_[0]) / scaler.scale_[0]
    preds = (preds_scaled - scaler.min_[0]) / scaler.scale_[0]

    df_eval = pd.DataFrame({
        "time": dates,
        "actual": actual,
        "pred": preds,
        "error": actual - preds,
        "abs_error": np.abs(actual - preds)
    }).set_index("time")

    overall_rmse = np.sqrt(mean_squared_error(df_eval["actual"], df_eval["pred"]))
    overall_mae = mean_absolute_error(df_eval["actual"], df_eval["pred"])

    # Define thresholds
    heat_thresh = 33.0 if city == "karachi" else 35.0
    cold_thresh = 8.0

    heatwave_df = df_eval[df_eval["actual"] >= heat_thresh]
    cold_snap_df = df_eval[df_eval["actual"] <= cold_thresh]

    heat_rmse = np.sqrt(mean_squared_error(heatwave_df["actual"], heatwave_df["pred"])) if len(heatwave_df) > 0 else np.nan
    heat_mae = mean_absolute_error(heatwave_df["actual"], heatwave_df["pred"]) if len(heatwave_df) > 0 else np.nan

    cold_rmse = np.sqrt(mean_squared_error(cold_snap_df["actual"], cold_snap_df["pred"])) if len(cold_snap_df) > 0 else np.nan
    cold_mae = mean_absolute_error(cold_snap_df["actual"], cold_snap_df["pred"]) if len(cold_snap_df) > 0 else np.nan

    return {
        "df_eval": df_eval,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "heat_count": len(heatwave_df),
        "heat_rmse": heat_rmse,
        "heat_mae": heat_mae,
        "cold_count": len(cold_snap_df),
        "cold_rmse": cold_rmse,
        "cold_mae": cold_mae
    }


def run_extremes_analysis():
    print("=== Analyzing DeepTemp Performance During Extreme Weather Events ===")
    cities = ["islamabad", "lahore", "karachi"]
    results = {}

    for city in cities:
        ckpt = os.path.join("checkpoints", f"{city}_lstm_best.pt")
        res = evaluate_city_extremes(city, ckpt)
        results[city] = res
        print(f"\n[{city.upper()}] Evaluation on Unseen Test Data (2019-2023):")
        print(f"  Overall:  RMSE = {res['overall_rmse']:.2f}°C | MAE = {res['overall_mae']:.2f}°C")
        if res["heat_count"] > 0:
            print(f"  Heatwaves (T >= 35°C/33°C, {res['heat_count']} days): RMSE = {res['heat_rmse']:.2f}°C | MAE = {res['heat_mae']:.2f}°C")
        if res["cold_count"] > 0:
            print(f"  Cold Snaps (T <= 8°C, {res['cold_count']} days):        RMSE = {res['cold_rmse']:.2f}°C | MAE = {res['cold_mae']:.2f}°C")

    # Generate Publication-Quality Visuals of Extreme Episodes
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1. Islamabad Winter Cold Wave (Dec 2019 - Jan 2020)
    isb_df = results["islamabad"]["df_eval"]
    sub_isb = isb_df.loc["2019-12-01":"2020-02-28"]
    axes[0].plot(sub_isb.index, sub_isb["actual"], label="Actual Temperature", color="#0284c7", linewidth=1.8, marker="o", markersize=3)
    axes[0].plot(sub_isb.index, sub_isb["pred"], label="LSTM Prediction", color="#dc2626", linestyle="--", linewidth=1.8)
    axes[0].axhline(8.0, color="#f59e0b", linestyle=":", label="Cold Snap Threshold (8°C)")
    axes[0].set_title("Islamabad: Winter Cold Snap Tracking (Dec 2019 - Feb 2020)", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Temp (°C)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True)

    # 2. Lahore Extreme Summer Heatwave (May - June 2022)
    lhr_df = results["lahore"]["df_eval"]
    sub_lhr = lhr_df.loc["2022-05-01":"2022-06-30"]
    axes[1].plot(sub_lhr.index, sub_lhr["actual"], label="Actual Temperature", color="#ea580c", linewidth=1.8, marker="o", markersize=3)
    axes[1].plot(sub_lhr.index, sub_lhr["pred"], label="LSTM Prediction", color="#991b1b", linestyle="--", linewidth=1.8)
    axes[1].axhline(35.0, color="#b91c1c", linestyle=":", label="Heatwave Threshold (35°C)")
    axes[1].set_title("Lahore: Peak Summer Heatwave Tracking (May - June 2022) [LESCO Demand Peak]", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Temp (°C)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", frameon=True)

    # 3. Karachi Maritime Pre-Monsoon Peak (May - June 2021)
    khi_df = results["karachi"]["df_eval"]
    sub_khi = khi_df.loc["2021-05-01":"2021-06-30"]
    axes[2].plot(sub_khi.index, sub_khi["actual"], label="Actual Temperature", color="#16a34a", linewidth=1.8, marker="o", markersize=3)
    axes[2].plot(sub_khi.index, sub_khi["pred"], label="LSTM Prediction", color="#1e3a8a", linestyle="--", linewidth=1.8)
    axes[2].axhline(33.0, color="#ea580c", linestyle=":", label="Coastal Heat Alert (33°C)")
    axes[2].set_title("Karachi: Coastal Maritime Heat Episode (May - June 2021) [K-Electric Grid Peak]", fontweight="bold", fontsize=12)
    axes[2].set_ylabel("Temp (°C)")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plot_path = os.path.join("reports", "extreme_weather_analysis.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"\n[*] Extreme weather evaluation figure exported to: {plot_path}")


if __name__ == "__main__":
    run_extremes_analysis()
