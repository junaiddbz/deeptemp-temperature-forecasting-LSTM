"""
DeepTemp: Data Preprocessing & Sequence Engineering (Phase 2)
Module: src/data_loader.py

This module handles:
1. Cleaning and imputation of raw climate data (daily continuity, linear/time interpolation).
2. Outlier detection and sanity checks for meteorological data.
3. Sequential train/test splitting (avoiding temporal data leakage).
4. Feature normalization (MinMaxScaler fitted strictly on training data).
5. Sliding-window sequence generator (past W days -> future H days).
6. PyTorch Dataset and DataLoader abstractions for training LSTM networks.
"""

import os
from typing import Tuple, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


class TemperatureDataset(Dataset):
    """
    PyTorch Dataset wrapping sliding-window time series sequences (X, y).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DataPreprocessor:
    """
    Comprehensive pipeline for time-series cleaning, normalization,
    and sequence engineering for climate forecasting.
    """
    def __init__(
        self,
        raw_data_dir: Optional[str] = None,
        processed_data_dir: Optional[str] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_data_dir = raw_data_dir or os.path.join(base_dir, "data")
        self.processed_data_dir = processed_data_dir or os.path.join(base_dir, "data", "processed")
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_col_scaler = MinMaxScaler(feature_range=(0, 1))

    def clean_city_data(
        self,
        input_csv: str,
        output_csv: Optional[str] = None,
        start_date: str = "2000-01-01",
        end_date: str = "2023-12-31"
    ) -> pd.DataFrame:
        """
        Step 1: Cleaning & Imputation
        - Parses dates and sorts chronologically.
        - Reindexes to a continuous daily frequency (D) to fix missing date steps.
        - Imputes missing temperature values using time-weighted interpolation.
        - Caps outliers based on physical limits for Pakistan (-10°C to 55°C).
        """
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file not found: {input_csv}")

        df = pd.read_csv(input_csv)
        if "time" not in df.columns:
            raise ValueError(f"CSV must contain a 'time' column. Found: {df.columns.tolist()}")

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"])
        df = df.set_index("time")

        # Create full continuous daily index
        full_idx = pd.date_range(start=start_date, end=end_date, freq="D")
        reindexed_df = df.reindex(full_idx)
        reindexed_df.index.name = "time"

        # Outlier bounds check for temperatures (Pakistan historical extremes)
        for temp_col in ["tavg", "tmin", "tmax"]:
            if temp_col in reindexed_df.columns:
                reindexed_df[temp_col] = reindexed_df[temp_col].clip(lower=-10.0, upper=55.0)

        # Time-based interpolation for tavg, followed by backfill/forward-fill for edges
        if "tavg" in reindexed_df.columns:
            reindexed_df["tavg"] = (
                reindexed_df["tavg"]
                .interpolate(method="time")
                .bfill()
                .ffill()
            )

        # Handle secondary features if present
        for col in ["tmin", "tmax", "wspd", "pres"]:
            if col in reindexed_df.columns:
                reindexed_df[col] = (
                    reindexed_df[col]
                    .interpolate(method="time")
                    .bfill()
                    .ffill()
                )

        if "prcp" in reindexed_df.columns:
            # Precipitation missing values are generally treated as 0.0 mm (no rain recorded)
            reindexed_df["prcp"] = reindexed_df["prcp"].fillna(0.0)

        # Drop columns that are completely empty (e.g. snow, tsun, wpgt in Meteostat)
        cols_to_drop = [c for c in ["snow", "wdir", "wpgt", "tsun"] if c in reindexed_df.columns and reindexed_df[c].isnull().all()]
        if cols_to_drop:
            reindexed_df = reindexed_df.drop(columns=cols_to_drop)

        # Save cleaned file if destination is provided
        if output_csv:
            reindexed_df.to_csv(output_csv)
            print(f"[*] Cleaned data saved: {output_csv} ({len(reindexed_df)} days)")

        return reindexed_df

    def clean_all_cities(self, cities: List[str] = ["islamabad", "lahore", "karachi"]):
        """
        Cleans and saves continuous preprocessed datasets for all target cities.
        """
        results = {}
        for city in cities:
            raw_path = os.path.join(self.raw_data_dir, f"{city}_daily_temps.csv")
            out_path = os.path.join(self.processed_data_dir, f"{city}_cleaned.csv")
            print(f"[+] Processing {city.upper()}...")
            cleaned_df = self.clean_city_data(raw_path, out_path)
            results[city] = cleaned_df
        return results

    @staticmethod
    def train_test_split(
        df: pd.DataFrame,
        split_date: str = "2019-01-01",
        val_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, ...]:
        """
        Step 4: Sequential Train / Test Split (Strictly avoiding data leakage)
        Default split:
        - Train: 2000-01-01 to 2018-12-31 (19 years ~ 79%)
        - Test:  2019-01-01 to 2023-12-31 (5 years  ~ 21%)
        """
        train_df = df.loc[df.index < split_date].copy()
        test_df = df.loc[df.index >= split_date].copy()

        if val_date is not None:
            val_df = train_df.loc[train_df.index >= val_date].copy()
            train_df = train_df.loc[train_df.index < val_date].copy()
            return train_df, val_df, test_df

        return train_df, test_df

    def normalize(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        val_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Step 2: Normalization (MinMaxScaler 0 to 1)
        Important: Scaler is fitted ONLY on the training split to avoid leakage.
        """
        # Ensure 2D shape for scaler
        is_1d = (train_data.ndim == 1)
        if is_1d:
            train_data = train_data.reshape(-1, 1)
            test_data = test_data.reshape(-1, 1)
            if val_data is not None:
                val_data = val_data.reshape(-1, 1)

        # Fit strictly on train
        scaled_train = self.scaler.fit_transform(train_data)
        scaled_test = self.scaler.transform(test_data)

        if val_data is not None:
            scaled_val = self.scaler.transform(val_data)
            return scaled_train, scaled_val, scaled_test

        return scaled_train, scaled_test

    @staticmethod
    def create_sliding_windows(
        data: np.ndarray,
        window_size: int = 30,
        forecast_horizon: int = 1,
        target_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 3: Sliding Windows (Past W days -> Predict Day W + H)
        
        Args:
            data: 2D array of shape (num_samples, num_features).
            window_size: Length of past historical window (e.g., 30 days).
            forecast_horizon: Number of days ahead to forecast (default 1).
            target_idx: Feature index representing the forecast target (default 0 for tavg).
            
        Returns:
            X: 3D array of shape (N, window_size, num_features)
            y: 1D array of shape (N,) for single-step, or (N, horizon) for multi-step.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        total_steps = len(data)
        X, y = [], []

        for i in range(total_steps - window_size - forecast_horizon + 1):
            window = data[i : i + window_size]
            if forecast_horizon == 1:
                target = data[i + window_size, target_idx]
            else:
                target = data[i + window_size : i + window_size + forecast_horizon, target_idx]

            X.append(window)
            y.append(target)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def prepare_dataset_for_city(
        self,
        city: str = "islamabad",
        features: List[str] = ["tavg"],
        window_size: int = 30,
        forecast_horizon: int = 1,
        split_date: str = "2019-01-01",
        batch_size: int = 32
    ) -> dict:
        """
        End-to-End Pipeline for a single city:
        1. Loads cleaned data.
        2. Splits train / test sequentially.
        3. Normalizes features (fit on train, transform test).
        4. Generates sliding windows (X, y).
        5. Wraps in PyTorch DataLoaders.
        """
        cleaned_path = os.path.join(self.processed_data_dir, f"{city}_cleaned.csv")
        if not os.path.exists(cleaned_path):
            raw_path = os.path.join(self.raw_data_dir, f"{city}_daily_temps.csv")
            self.clean_city_data(raw_path, cleaned_path)

        df = pd.read_csv(cleaned_path, index_col="time", parse_dates=True)
        feature_data = df[features].values

        train_df, test_df = self.train_test_split(df, split_date=split_date)
        train_raw = train_df[features].values
        test_raw = test_df[features].values

        # Normalize
        train_scaled, test_scaled = self.normalize(train_raw, test_raw)

        # Sliding windows
        X_train, y_train = self.create_sliding_windows(
            train_scaled, window_size=window_size, forecast_horizon=forecast_horizon
        )
        X_test, y_test = self.create_sliding_windows(
            test_scaled, window_size=window_size, forecast_horizon=forecast_horizon
        )

        # PyTorch DataLoaders
        train_dataset = TemperatureDataset(X_train, y_train)
        test_dataset = TemperatureDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return {
            "city": city,
            "features": features,
            "window_size": window_size,
            "scaler": self.scaler,
            "train_df": train_df,
            "test_df": test_df,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "train_loader": train_loader,
            "test_loader": test_loader,
        }


if __name__ == "__main__":
    print("=== Executing Phase 2: Data Preprocessing & Sequence Engineering ===")
    preprocessor = DataPreprocessor()

    # Clean all 3 regional datasets
    cleaned_datasets = preprocessor.clean_all_cities(["islamabad", "lahore", "karachi"])

    # Prepare sequences and PyTorch loader for Islamabad prototype
    print("\n=== Building Sequences for Islamabad (Window: 30 days) ===")
    pipeline = preprocessor.prepare_dataset_for_city(
        city="islamabad",
        features=["tavg"],
        window_size=30,
        forecast_horizon=1,
        split_date="2019-01-01",
        batch_size=32
    )

    print(f"Training sequences X_train shape: {pipeline['X_train'].shape}")
    print(f"Training targets y_train shape:   {pipeline['y_train'].shape}")
    print(f"Testing sequences X_test shape:   {pipeline['X_test'].shape}")
    print(f"Testing targets y_test shape:     {pipeline['y_test'].shape}")
    print(f"Number of training batches:       {len(pipeline['train_loader'])}")
    print(f"Number of testing batches:        {len(pipeline['test_loader'])}")
    print("\nPhase 2 pipeline executed successfully!")
