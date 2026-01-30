# Load libraries
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from typing import Optional, Tuple
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


def sine_dataset(length: int, EASY_DATASET: bool = False, **kwargs):
    """
    Generates synthetic sine wave data. 
    **kwargs captures extra arguments like freq_type passed from other functions.
    """
    x = np.linspace(1,length,length)
    np.random.seed(42) # Fixed seed for reproducibility

    if EASY_DATASET:
        period = 90
        delay = np.random.uniform(0, period)
        # Sine wave with a linear trend
        return np.sin(2 * np.pi * (x - delay) / period)
    else:
        # Multi-seasonal sine wave
        period_1 = 300
        sine_wave_1 = 0.8 * np.sin(2 * np.pi * x / period_1)

        period_2 = 30
        sine_wave_2 = 0.4 * np.sin(2 * np.pi * x / period_2)

        period_3 = 5
        sine_wave_3 = 0.1 * np.sin(2 * np.pi * x / period_3)

        noise = np.random.normal(0, 0.1, length)
        return (sine_wave_1 + sine_wave_2 + sine_wave_3 + noise).astype(np.float32)
    


class TimeSeriesDataset(Dataset):
    """Dataset for time series data compatible with TimesFM."""

    def __init__(self,
                 series: np.ndarray,
                 context_length: int,
                 horizon_length: int,
                 freq_type: int = 0):
        if freq_type not in [0, 1, 2]:
            raise ValueError("freq_type must be 0, 1, or 2")

        self.series = series
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.freq_type = freq_type
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        """Prepare sliding window samples from the time series."""
        self.samples = []
        total_length = self.context_length + self.horizon_length

        for start_idx in range(0, len(self.series) - total_length + 1):
            end_idx = start_idx + self.context_length
            x_context = self.series[start_idx:end_idx]
            x_future = self.series[end_idx:end_idx + self.horizon_length]
            self.samples.append((x_context, x_future))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_context, x_future = self.samples[index]

        x_context = torch.tensor(x_context, dtype=torch.float32)
        x_future = torch.tensor(x_future, dtype=torch.float32)

        input_padding = torch.zeros_like(x_context)
        freq = torch.tensor([self.freq_type], dtype=torch.long)

        return x_context, input_padding, freq, x_future
    

def prepare_datasets(series: np.ndarray,
                     context_length: int,
                     horizon_length: int,
                     freq_type: int = 0,
                     train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
    """Prepare training and test datasets from time series data."""

    train_size = int(len(series) * train_split)
    # Adjust split to respect the minimum data needed for one sample
    min_len_needed = context_length + horizon_length
    
    if len(series) - train_size < min_len_needed:
        # Ensure test set has at least one full sample window
        train_size = len(series) - min_len_needed
        if train_size < 0:
            raise ValueError("Series length is too short for the given context and horizon lengths.")

    # Split the raw data
    train_raw = series[:train_size].reshape(-1, 1)
    test_raw = series[train_size:].reshape(-1, 1)

    # Fit scaler on training data, then transform both
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw).flatten()
    test_scaled = scaler.transform(test_raw).flatten()

    # Create datasets with specified frequency type
    train_dataset = TimeSeriesDataset(train_scaled,
                                      context_length=context_length,
                                      horizon_length=horizon_length,
                                      freq_type=freq_type)

    test_dataset = TimeSeriesDataset(test_scaled,
                                    context_length=context_length,
                                    horizon_length=horizon_length,
                                    freq_type=freq_type)

    return train_dataset, test_dataset, scaler


def get_data_real(route_path: str,
                  value_col: str,
                  context_len: int,
                  horizon_len: int,
                  freq_type: int = 0) -> Tuple[Dataset, Dataset, StandardScaler]:
    """
    Retrieves a real-world time series (e.g., Hourly Energy) and prepares datasets.

    Here, we simulate loading a real dataset like the 'Hourly Energy Consumption'
    which is high frequency (freq_type=0).
    """
    try:
        df = pd.read_csv(route_path)
        series = df[value_col].values.astype(np.float32)
        print(f"Successfully loaded {len(series)} data points.")

    except FileNotFoundError:
        return get_data_synthetic(context_len, horizon_len, freq_type)
    
    # Ensure a minimum length
    if len(series) < context_len + horizon_len:
         raise ValueError(f"Real series length ({len(series)}) is too short for C={context_len} and H={horizon_len}.")

    # Use the existing data preparation logic (which is now corrected)
    train_dataset, test_dataset, scaler = prepare_datasets(
        series=series,
        context_length=context_len,
        horizon_length=horizon_len,
        freq_type=freq_type,
        train_split=0.8,
    )

    print(f"Created datasets (Real Data):")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Using frequency type: {freq_type}")
    return train_dataset, test_dataset, scaler


def get_real_data(context_len: int,
                  horizon_len: int,
                  freq_type: int = 0) -> Tuple[Dataset, Dataset]:
    df = yf.download("AAPL", start="2010-01-01", end="2019-01-01")
    df_clean = df["Close"].dropna()

    time_series = df_clean.values
    time_series = time_series[np.isfinite(time_series)]
        
    if len(time_series) == 0:
        raise ValueError("Time Series is empty after NaN clean up")

    train_dataset, test_dataset, scaler = prepare_datasets(
        series=time_series,
        context_length=context_len,
        horizon_length=horizon_len,
        freq_type=freq_type,
        train_split=0.8,
    )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Using frequency type: {freq_type}")
    return train_dataset, test_dataset, scaler


def get_data_synthetic(context_len: int,
                       horizon_len: int,
                       freq_type: int = 0) -> Tuple[Dataset, Dataset, StandardScaler]:
    """Generates synthetic data and returns datasets + scaler."""
    # Fixed the call to match sine_dataset's new parameters
    time_series = sine_dataset(length=2264, freq_type=freq_type)
    
    train_dataset, test_dataset, scaler = prepare_datasets(
        series=time_series,
        context_length=context_len,
        horizon_length=horizon_len,
        freq_type=freq_type,
        train_split=0.8,
    )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Using frequency type: {freq_type}")
    return train_dataset, test_dataset, scaler