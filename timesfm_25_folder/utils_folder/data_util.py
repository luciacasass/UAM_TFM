# Load libraries
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from pathlib import Path
from typing import Optional, Tuple, Union
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings("ignore")


def estimate_period_from_acf(y, max_lag=500, min_lag=2):
    """Estimates the dominant period of a time series using Autocorrelation Function (ACF).

    Args:
        y: The input time series array.
        max_lag: The maximum number of lags to calculate for ACF.
        min_lag: The minimum lag to consider for a valid period.

    Returns:
        The lag corresponding to the strongest ACF peak, or None if no peaks are found.
    """

    y_detrended = y - np.mean(y)
    acf_vals = acf(y_detrended, nlags=max_lag, fft=True)

    # Ignore lag 0
    lags = np.arange(len(acf_vals))
    acf_vals = acf_vals[1:]
    lags = lags[1:]

    # Find peaks
    peaks, _ = find_peaks(acf_vals)

    # Keep only peaks above min_lag
    peaks = peaks[lags[peaks] >= min_lag]

    if len(peaks) == 0:
        return None

    # Strongest peak
    best_peak = peaks[np.argmax(acf_vals[peaks])]

    return lags[best_peak]



def sine_dataset(length: int, EASY_DATASET: bool = False, **kwargs):
    """Generates synthetic sine wave data with single or multiple seasonalities.

    Args:
        length: The number of data points to generate.
        EASY_DATASET: If True, generates a simple sine wave with a random delay. 
            If False, generates a complex multi-seasonal wave with noise.
        **kwargs: Additional arguments (e.g., freq_type) passed from callers.

    Returns:
        A 1D numpy array containing the generated synthetic time series.
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
    """Dataset for time series data compatible with TimesFM using a sliding window.

    Attributes:
        series: The full time series array.
        context_length: The number of historical points used for input.
        horizon_length: The number of future points to predict.
        freq_type: Integer flag representing the frequency of the data (0, 1, or 2).
        samples: List of tuples containing (context_window, future_window).
    """

    def __init__(self,
                 series: np.ndarray,
                 context_length: int,
                 horizon_length: int,
                 freq_type: int = 0):
        """Initializes the dataset and prepares sliding window samples.

        Args:
            series: The input time series.
            context_length: Look-back window size.
            horizon_length: Prediction window size.
            freq_type: Frequency category (must be 0, 1, or 2).

        Raises:
            ValueError: If freq_type is not within the allowed range.
        """

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
        """Returns the total number of windowed samples available."""
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves a single sample window for training/inference.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            A tuple containing:
                - x_context: Historical window tensor.
                - input_padding: Zero-filled tensor of context shape.
                - freq: Frequency type tensor.
                - x_future: Target future window tensor.
        """
        x_context, x_future = self.samples[index]

        x_context = torch.tensor(x_context, dtype=torch.float32)
        x_future = torch.tensor(x_future, dtype=torch.float32)

        input_padding = torch.zeros_like(x_context)
        freq = torch.tensor([self.freq_type], dtype=torch.long)

        return x_context, input_padding, freq, x_future
    


def prepare_datasets(series: np.ndarray,
                     context_length: int,
                     horizon_length: int,
                     max_horizon_length: int,
                     freq_type: int = 0,
                     train_split: float = 0.8,
                     decompose: Optional[str] = None,
                     period: Union[int, str] = "auto") -> Tuple[Dataset, Dataset]:
    """Splits, decomposes, and scales time series data into Train/Test datasets.

    Args:
        series: The raw 1D time series data.
        context_length: Input window size.
        horizon_length: Output window size.
        freq_type: Frequency identifier for the model.
        train_split: Fraction of data to use for training (0.0 to 1.0).
        decompose: Type of decomposition ("trend_diff", "seasonal_diff", "seasonal", "both").
        period: Seasonal period. If "auto", it's estimated via ACF.

    Returns:
        A tuple containing:
            - train_dataset: TimeSeriesDataset for training.
            - test_dataset: TimeSeriesDataset for testing.
            - scaler: The StandardScaler instance fitted on training data.
            - removed_component: Array of the trend/seasonal components removed.
            - train_size: The index where the training set ends.

    Raises:
        ValueError: If the series is too short for the requested windows.
    """

    series = np.asarray(series).flatten()
    train_size = int(len(series) * train_split)
    # Adjust split to respect the minimum data needed for one sample
    min_len_needed = context_length + horizon_length
    
    if len(series) - train_size < min_len_needed:
        # Ensure test set has at least one full sample window
        train_size = len(series) - min_len_needed
        if train_size < 0:
            raise ValueError("Series length is too short for the given context and horizon lengths.")

    # Split the raw data
    train_raw = series[:train_size]
    test_raw = series[train_size:]

    # test too?

    train_size = len(train_raw)
    test_size = len(test_raw)
    
    removed_component = np.zeros(train_size + test_size)
    if decompose in ["seasonal_diff", "seasonal", "both"] and period == "auto":
            period = estimate_period_from_acf(train_raw)
            if period is None: 
                print("Warning: No period detected, falling back to no decomposition.")
                decompose = None
            else:
                period = int(period)
                print(f"Detected Period: {period}")

    if decompose == "trend_diff":
        train_input = pd.Series(train_raw).diff().fillna(0).values
        removed_component[:train_size] = pd.Series(train_raw).shift(1).fillna(train_raw[0]).values

        test_combined = np.concatenate([[train_raw[-1]], test_raw])
        test_input = pd.Series(test_combined).diff().dropna().values
        removed_component[train_size:] = pd.Series(test_combined).shift(1).dropna().values

    elif decompose == "seasonal_diff":
        # Train Diff
        # train_input = pd.Series(train_raw).diff(periods=period).fillna(0).values
        # removed_component[:train_size] = pd.Series(train_raw).shift(period).fillna(train_raw[0]).values

        # test_combined = np.concatenate([train_raw[-period:], test_raw])
        # test_input = pd.Series(test_combined).diff(periods=period).dropna().values
        # removed_component[train_size:] = pd.Series(test_combined).shift(period).dropna().values
        train_series = pd.Series(train_raw)
        train_diffs = train_series.diff(periods=period)

        mean_diff = train_diffs.mean()
        train_input = train_diffs.fillna(mean_diff).values
        
        # For the removed component, use bfill 
        removed_component[:train_size] = train_series.shift(period).bfill().values

        test_combined = np.concatenate([train_raw[-period:], test_raw])
        test_combined_series = pd.Series(test_combined)
        
        test_input = test_combined_series.diff(periods=period).dropna().values
        removed_component[train_size:] = test_combined_series.shift(period).dropna().values
    
    elif decompose in ["seasonal", "both"]:
    
        # Decompose training data
        stl = STL(train_raw, period=period, robust=True).fit()
        seasonal_train = stl.seasonal
        
        # Project seasonality to test set
        last_cycle = seasonal_train[-period:]
        reps = int(np.ceil(len(test_raw) / period))
        seasonal_test = np.tile(last_cycle, reps)[:len(test_raw)]

        if decompose == "seasonal":
            # Keep Trend + Residual (Remove only Season)
            train_input = train_raw - seasonal_train
            test_input = test_raw - seasonal_test

            removed_component[:train_size] = seasonal_train
            removed_component[train_size:] = seasonal_test
        elif decompose == "both":
            # Keep only Residual (Remove Season + Trend)
            indices = np.arange(len(stl.trend)).reshape(-1, 1)
            lr = LinearRegression().fit(indices, stl.trend)
            
            # Project the trend line into the test indices
            test_indices = np.arange(len(stl.trend), \
                                     len(stl.trend) + len(test_raw)).reshape(-1, 1)
            projected_trend_test = lr.predict(test_indices)
            
            train_input = stl.resid
            test_input = test_raw - seasonal_test - projected_trend_test
            
            removed_component[:train_size] = seasonal_train + stl.trend
            removed_component[train_size:] = seasonal_test + projected_trend_test
    else:
        train_input = train_raw
        test_input = test_raw

    # Fit scaler on training data, then transform both
    scaler = StandardScaler()
    scaler.fit(train_input.reshape(-1, 1))

    h_diff = max_horizon_length - horizon_length
    train_input = train_input[h_diff:]
    test_input = test_input[h_diff:]

    train_scaled = scaler.transform(train_input.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_input.reshape(-1, 1)).flatten()

    # Create datasets with specified frequency type
    train_dataset = TimeSeriesDataset(train_scaled,
                                      context_length=context_length,
                                      horizon_length=horizon_length,
                                      freq_type=freq_type)

    test_dataset = TimeSeriesDataset(test_scaled,
                                    context_length=context_length,
                                    horizon_length=horizon_length,
                                    freq_type=freq_type)

    return train_dataset, test_dataset, scaler, removed_component, train_size



### DATA RETRIEVAL FUNCTIONS ###

def get_real_data(context_len: int,
                  horizon_len: int,
                  max_horizon_len: int,
                  freq_type: int = 0,
                  decompose: Optional[str] = None,
                  period: Union[int, str] = "auto") -> Tuple[Dataset, Dataset, StandardScaler]:
    """Downloads AAPL stock data from yfinance and prepares it for modeling.

    Args:
        context_len: Historical window size.
        horizon_len: Prediction window size.
        freq_type: Frequency category.
        decompose: Optional decomposition method.
        period: Seasonal period.

    Returns:
        Prepared train/test datasets, scaler, removed components, and split index.

    Raises:
        ValueError: If the downloaded series is empty after cleaning.
    """

    df = yf.download("AAPL", start="2010-01-01", end="2019-01-01")
    df_clean = df["Close"].dropna()

    time_series = df_clean.values
    time_series = time_series[np.isfinite(time_series)]
        
    if len(time_series) == 0:
        raise ValueError("Time Series is empty after NaN clean up")

    train_dataset, test_dataset, scaler, seas, train_size = prepare_datasets(
        series=time_series,
        context_length=context_len,
        horizon_length=horizon_len,
        max_horizon_length=max_horizon_len,
        freq_type=freq_type,
        train_split=0.8,
        decompose=decompose,
        period=period
    )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Using frequency type: {freq_type}")
    return train_dataset, test_dataset, scaler, seas, train_size



def get_data_synthetic(context_len: int,
                       horizon_len: int,
                       max_horizon_len: int,
                       freq_type: int = 0,
                       decompose: bool = False,
                       period: Union[int, str] = "auto") -> Tuple[Dataset, Dataset, StandardScaler]:
    """Generates a synthetic sine wave dataset and prepares it for modeling.

    Args:
        context_len: Historical window size.
        horizon_len: Prediction window size.
        freq_type: Frequency category.
        decompose: Whether to apply decomposition.
        period: Seasonal period.

    Returns:
        Prepared train/test datasets, scaler, removed components, and split index.
    """

    # Fixed the call to match sine_dataset's new parameters
    time_series = sine_dataset(length=2264, freq_type=freq_type)
    
    train_dataset, test_dataset, scaler, seas, train_size = prepare_datasets(
        series=time_series,
        context_length=context_len,
        horizon_length=horizon_len,
        max_horizon_length=max_horizon_len,
        freq_type=freq_type,
        train_split=0.8,
        decompose=decompose,
        period=period
    )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Using frequency type: {freq_type}")
    return train_dataset, test_dataset, scaler, seas, train_size



def get_demand_data(file_path: str,
                  context_len: int,
                  horizon_len: int,
                  max_horizon_len: int,
                  freq_type: int = 0,
                  decompose: Optional[str] = None,
                  period: Union[int, str] = "auto") -> Tuple[Dataset, Dataset, StandardScaler]:
    """Loads demand data from a parquet file and prepares it for modeling.

    Args:
        file_path: Path to the .parquet file.
        context_len: Historical window size.
        horizon_len: Prediction window size.
        freq_type: Frequency category.
        decompose: Optional decomposition method.
        period: Seasonal period.

    Returns:
        Prepared train/test datasets, scaler, removed components, and split index.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the demand column is empty after cleaning.
    """

    if "diario" in file_path:
        pp = 7
        freq_str = 'D'
    else:
        pp = 24
        freq_str = 'h'

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    df = pd.read_parquet(path, engine='fastparquet')

    if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq_str)
    df = df.reindex(full_range)

    max_repeats = 2 if freq_str == "D" else 7*2

    for i in range(max_repeats):
        df['demand'] = df['demand'].fillna(df['demand'].shift(pp))
        # if no more NaNs exist, stop the loop early
        if not df['demand'].isnull().any():
            break

    # If there are still NaNs
    df['demand'] = df['demand'].interpolate()

    time_series = df["demand"].values
    if "horario" in file_path:
        time_series = time_series[1680:5040]
        
    if len(time_series) == 0:
        raise ValueError("Time Series is empty after NaN clean up")

    train_dataset, test_dataset, scaler, seas, train_size = prepare_datasets(
        series=time_series,
        context_length=context_len,
        horizon_length=horizon_len,
        max_horizon_length=max_horizon_len,
        freq_type=freq_type,
        train_split=0.8,
        decompose=decompose,
        period=period
    )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Using frequency type: {freq_type}")
    return train_dataset, test_dataset, scaler, seas, train_size