import pandas as pd
import numpy as np
import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from utils_folder.data_util import *


# # ARIMA Prediction
# def predict_arima_sktime(context_vals, horizon_len):
#     forecaster = AutoARIMA(
#         sp=1, 
#         suppress_warnings=True, 
#         error_action='ignore', # Ignores models that fail to fit
#         stationary=False,      # Allows it to search for non-stationary models
#         information_criterion='aic',
#         maxiter=50
#     )
    
#     y_train = pd.Series(context_vals)
    
#     try:
#         forecaster.fit(y_train)

#         print("ARIMA model selected:")
#         print(f"Order (p,d,q): {forecaster.order}")
#         print(f"Seasonal order (P,D,Q,s): {forecaster.seasonal_order}")
#         print(f"AIC del modelo: {forecaster.aic():.2f}")

#         fh = np.arange(1, horizon_len + 1)
#         forecast = forecaster.predict(fh)   # generates all n future steps at once
#         return forecast.values
#     except Exception as e:
#         # Fallback: if ARIMA fails completely on a window, 
#         # return the last value (Persistence) so the loop doesn't crash.
#         return np.full(horizon_len, context_vals[-1])


def statsmodels_auto_arima_(context_vals, horizon_len, params=None, max_p=4, max_q=2):
    y_train = pd.Series(context_vals)

    if params is None:
        # Find optimal d value
        d = 0
        y_diff = y_train.copy()
        while d <= 2:
            if adfuller(y_diff.dropna())[1] < 0.05:
                break
            y_diff = y_diff.diff()
            d += 1
        
        # Find (p, q) now using stationary data
        stat_data = y_train.diff(d).dropna() if d > 0 else y_train
        res = sm.tsa.arma_order_select_ic(stat_data, max_ar=max_p, max_ma=max_q, ic='aic')
        p, q = res.aic_min_order
    else:
        p, d, q = params
    
    # Create ARIMA model
    model = ARIMA(y_train, order=(p, d, q)).fit()
    
    # Prediction for future horizon
    forecast = model.forecast(steps=horizon_len)
    return forecast.to_numpy(), (p, d, q)


# Persistence Model Prediction
def predict_persistence(context_vals, horizon_len):
    return np.full(horizon_len, context_vals[-1])


def split_train_val(dataset):
    train_size = int(0.8 * len(dataset))
    val_internal_size = len(dataset) - train_size
    
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_internal_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=True)

    return train_loader, val_loader


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=128):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])
    
class AutoregressiveLSTM(nn.Module):
    def __init__(self, horizon_len, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.horizon_len = horizon_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        outputs = []
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        _, h = self.lstm(x)
        cur = x[:, -1:, :]

        for _ in range(self.horizon_len):
            out, h = self.lstm(cur, h)
            pred = self.fc(out)
            outputs.append(pred)
            cur = pred

        return torch.cat(outputs, dim=1).squeeze(-1)
    

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, target_idx=None):
    model.eval()
    total = 0.0
    for x, _, _, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        target = y[:, target_idx] if target_idx is not None else y

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        total += loss_fn(model(x), target).item()
    return total / len(loader)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    max_epochs=100,
    patience=5,
    target_idx=None
):
    best_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for _ in range(max_epochs):
        model.train()
        for x, _, _, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            target = y[:, target_idx] if target_idx is not None else y

            if x.dim() == 2:
                x = x.unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), target)
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader, loss_fn, device, target_idx=target_idx)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    

def train_multi_output(train_dataset, horizon_len, device):
    model = LSTMModel(1, 64, 2, horizon_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_loader, val_loader = split_train_val(train_dataset)
    
    start = time.time()
    train_with_early_stopping(
        model, train_loader, val_loader,
        optimizer, criterion, device
    )
    return model, time.time() - start


def train_multi_model(train_dataset, horizon_len, device):
    criterion = nn.MSELoss()
    train_loader, val_loader = split_train_val(train_dataset)

    models = []
    start = time.time()

    for i in range(horizon_len):
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1).to(device)
        opt = torch.optim.Adam(model.parameters(), 1e-3)

        train_with_early_stopping(
            model, train_loader, val_loader,
            opt, criterion, device, target_idx=i
        )

        models.append(model)

    return models, time.time() - start



def train_autoregressive(train_dataset, horizon_len, device):

    model = AutoregressiveLSTM(horizon_len=horizon_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_loader, val_loader = split_train_val(train_dataset)
    
    start = time.time()
    train_with_early_stopping(
        model, train_loader, val_loader,
        optimizer, criterion, device
    )
    return model, time.time() - start
    
