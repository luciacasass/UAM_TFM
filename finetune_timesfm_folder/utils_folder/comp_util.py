import pandas as pd
import numpy as np
import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

from utils_folder.data_util import *

import os
import random


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    
    # Critical for CuDNN reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)


def select_d(train_data, max_d=2, alpha=0.05, min_obs=30):
    d = 0
    y = train_data.copy()
    y = pd.Series(y)

    while d <= max_d:
        y_clean = y.dropna()
        # Avoid ADF with few data points
        if len(y_clean) < min_obs:
            break

        pvalue = adfuller(y_clean, autolag='AIC')[1]
        if pvalue < alpha:
            return d

        y = y.diff()
        d += 1

    return d

def statsmodels_auto_arima_(context_vals, horizon_len=None, params=None, max_p=8, max_q=8):

    if params is None:
        train_size = int(0.8 * len(context_vals))
        train_data = context_vals[:train_size]
        val_data = context_vals[train_size:]

        # Find optimal d value
        d = select_d(train_data)
        
        best_rmse = float('inf')
        p, q = 0, 0
        
        # Using grid search once param d is set
        for cur_p in range(max_p + 1):
            for cur_q in range(max_q + 1):
                try:
                    # Fit on train
                    model = ARIMA(train_data, order=(cur_p, d, cur_q)).fit()
                    # Validate on validation length
                    preds = model.forecast(steps=len(val_data))
                    rmse = np.sqrt(mean_squared_error(val_data, preds))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        p, q = cur_p, cur_q
                except:
                    continue
                
        if (p, d, q) == (0, 0, 0):
            warnings.warn(
                "Best ARIMA model is (0,0,0). This may indicate the series is white noise or no structure was found.",
                UserWarning
            )
    else:
        p, d, q = params
    
    # Create ARIMA model
    if horizon_len is not None:
        y_test = pd.Series(context_vals)
        model = ARIMA(y_test, order=(p, d, q)).fit()
        
        # Prediction for future horizon
        forecast = model.forecast(steps=horizon_len)
        return forecast.to_numpy(), (p, d, q)
    else:
        return None, (p, d, q)
    


# Persistence Model Prediction
def predict_persistence(context_vals, horizon_len):
    return np.full(horizon_len, context_vals[-1])


# Dummy Regressor Model Prediction
def predict_dummy(model, context_vals):
    x_input = context_vals.reshape(1, -1)
    preds = model.predict(x_input)

    return preds.flatten()


def split_train_val(dataset):
    train_size = int(0.8 * len(dataset))

    # Slice sequentially, no shuffle
    train_ds = torch.utils.data.Subset(dataset, range(0, train_size))
    val_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    return train_loader, val_loader


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=128):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # , dropout=0.2??
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])
    
class AutoregressiveLSTM(nn.Module):
    def __init__(self, horizon_len, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.horizon_len = horizon_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # , dropout=0.2??
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

        if target_idx is not None:
            target = y[:, target_idx].unsqueeze(-1) # Ensures shape [Batch, 1]
        else:
            target = y

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

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for _ in range(max_epochs):
        model.train()
        for x, _, _, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if target_idx is not None:
                target = y[:, target_idx].unsqueeze(-1) # Ensures shape [Batch, 1]
            else:
                target = y

            if x.dim() == 2:
                x = x.unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)

            output = model(x)
            if output.shape != target.shape:
                output = output.view(target.shape)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader, loss_fn, device, target_idx=target_idx)
        scheduler.step(val_loss)

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
    

def train_multi_output(train_dataset, horizon_len, device, seed=42):
    
    seed_everything(seed)

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


def train_multi_model(train_dataset, horizon_len, num_pred, device, seed=42):
    criterion = nn.MSELoss()
    train_loader, val_loader = split_train_val(train_dataset)

    models = []
    start = time.time()

    start_idx = horizon_len - num_pred

    for i in range(start_idx, horizon_len):
        seed_everything(seed + i)
        
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1).to(device)
        opt = torch.optim.Adam(model.parameters(), 1e-3)

        train_with_early_stopping(
            model, train_loader, val_loader,
            opt, criterion, device, target_idx=i
        )

        models.append(model)

    return models, time.time() - start



def train_autoregressive(train_dataset, horizon_len, device, seed=42):

    seed_everything(seed)

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
    
