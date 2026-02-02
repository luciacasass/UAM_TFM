import numpy as np
import torch
from os import path
import time
import os
import logging

from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from utils_folder.data_util import *
from utils_folder.comp_util import *

import warnings
warnings.filterwarnings("ignore")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ID = "google/timesfm-2.0-500m-pytorch"


# Model creation
def get_model(horizon_len: int, load_weights: bool = False):
    hf_path = snapshot_download(REPO_ID)
    hparams = TimesFmHparams(
        backend=str(DEVICE),
        per_core_batch_size=32,
        horizon_len=horizon_len,
        num_layers=50,
        use_positional_embedding=True,
        context_len=192,
    )
    tfm = TimesFm(hparams=hparams,
                  checkpoint=TimesFmCheckpoint(huggingface_repo_id=REPO_ID))
    model = PatchedTimeSeriesDecoder(tfm._model_config)

    ckpt = torch.load(f"{hf_path}/torch_model.ckpt", weights_only=True, map_location=DEVICE)
    model.load_state_dict(ckpt)
    return model.to(DEVICE).eval()



# Plot predictions
def plot_predictions(model, dataset, scaler, title, save_path):
    """Cleaned up plotting logic."""
    model.eval()
    # Get a single batch and take first item
    loader = DataLoader(dataset, batch_size=1)
    x_ctx, x_pad, freq, x_future = next(iter(loader))
    
    x_ctx, x_pad, freq = [t.to(DEVICE) for t in [x_ctx, x_pad, freq]]
    
    with torch.no_grad():
        preds = model(x_ctx, x_pad.float(), freq)[..., 0]
        # Get predictions for the specific horizon length
        pred_vals = preds[0, -1, :x_future.shape[1]].cpu().numpy()

    # Batch inverse transform is faster
    context_vals = scaler.inverse_transform(x_ctx[0].cpu().numpy().reshape(-1, 1)).flatten()
    future_vals = scaler.inverse_transform(x_future[0].cpu().numpy().reshape(-1, 1)).flatten()
    pred_vals = scaler.inverse_transform(pred_vals.reshape(-1, 1)).flatten()

    plt.figure(figsize=(10, 5))
    plt.plot(context_vals, label="History")
    plt.plot(range(len(context_vals), len(context_vals) + len(future_vals)), future_vals, label="Actual")
    plt.plot(range(len(context_vals), len(context_vals) + len(future_vals)), pred_vals, label="Predicted")
    plt.title(title)
    plt.legend()
    if save_path: plt.savefig(save_path)
    plt.close()




def get_rmse(model: PatchedTimeSeriesDecoder, test_dataset: Dataset, scaler: StandardScaler, num_pred: int) -> float:
    """Computes RMSE evaluating just the last step in horizon H"""
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []
    all_targets = []
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for x_context, x_padding, freq, x_future in test_dataloader:
            x_context, x_padding, freq = x_context.to(device), x_padding.to(device), freq.to(device)
            
            # TimesFM Preds: [Batch, Num_Patches, Prediction_Length]
            predictions = model(x_context, x_padding.float(), freq)
            predictions_mean = predictions[..., 0] 
            last_patch_pred = predictions_mean[:, -1, :] 

            # Take only last step in horizon
            # x_future shape: [Batch, Horizon_Len] -> slice [:, -1]
            final_step_pred = last_patch_pred[:, -num_pred:] 
            final_step_target = x_future[:, -num_pred:]

            # Invertir escala (reshape para sklearn)
            preds_unscaled = scaler.inverse_transform(final_step_pred.cpu().numpy().reshape(-1, 1)).flatten()
            targets_unscaled = scaler.inverse_transform(final_step_target.cpu().numpy().reshape(-1, 1)).flatten()
            
            all_preds.extend(preds_unscaled)
            all_targets.extend(targets_unscaled)

    return np.sqrt(mean_squared_error(all_targets, all_preds))


def compare_performance(context_len: int, 
                        horizon_len: int, 
                        num_pred: int = 1,
                        freq_type: int = 0, 
                        real_data: bool = False,
                        more_models: bool = False,
                        plot_graph: bool = False):
    """Loads and compares zero-shot vs fine-tuned TimesFM performance."""

    # Update horizon_len (number of predictions to be made in order to extract
    # num_pred points with horizon H)
    horizon_len = horizon_len + num_pred - 1

    # Load Data
    if real_data:
        print(f"Loading Real Dataset...", flush=True)
        train_dataset, test_dataset, scaler = get_real_data(
            context_len=context_len,
            horizon_len=horizon_len,
            freq_type=freq_type
            )
    else:
        print("Generating Synthetic Sine Dataset...", flush=True)
        train_dataset, test_dataset, scaler = get_data_synthetic(
            context_len=context_len,
            horizon_len=horizon_len,
            freq_type=freq_type
        )


    results = {}

    # Zero-Shot (No Fine-Tuning) Performance
    print("\nZero-Shot (Pre-trained) Evaluation", flush=True)
    model = get_model(horizon_len)

    t0 = time.time()
    rmse = get_rmse(model, test_dataset, scaler, num_pred)
    infer_t = time.time() - t0

    results["TimesFM ZeroShot"] = {
        "RMSE": rmse,
        "Inference_Time": infer_t
    }
    print(f"Zero-Shot RMSE: {rmse:.4f}", flush=True)
    
    # Plot Zero-Shot
    if real_data:
        path_ = f"_{horizon_len}_real"
    else:
        path_ = f"_{horizon_len}_synth"

    if plot_graph:
        plot_predictions(
            model=model,
            test_dataset=test_dataset,
            scaler=scaler,
            title=f"TimesFM Zero-Shot Prediction on {path_[:-4]} Data",
            save_path="zero_shot_predictions" + path_ + ".png",
        )
    
    del model # Remove from memory before fine-tuning the new model

    
    # Fine-Tuning
    print("\nFine-Tuning Process", flush=True)
    finetune_model = get_model(horizon_len)

    config = FinetuningConfig(
        batch_size=32,
        num_epochs=5,
        learning_rate=1e-4,
        use_wandb=False, # Set to True to enable Weights & Biases logging
        freq_type=freq_type,
        log_every_n_steps=10,
        val_check_interval=0.5,
        use_quantile_loss=False # Recommended True loss for TimesFM
    )

    def sliced_mse_loss(preds, targets):
        # Force targets to be 2D: [Batch, Horizon]
        # If it was [Batch], it becomes [Batch, 1]
        t = targets.view(targets.shape[0], -1)
        
        # Slice predictions to match the target horizon width
        # preds is [Batch, Max_Horizon]
        p = preds[:, :t.shape[1]]
        
        return torch.mean((p - t)**2)

    finetuner = TimesFMFinetuner(finetune_model, config, loss_fn=sliced_mse_loss)

    print("Training...")
    start_train = time.time()
    finetuner.finetune(train_dataset, test_dataset)
    train_time = time.time() - start_train

    t0 = time.time()
    rmse_ft = get_rmse(finetune_model, test_dataset, scaler, num_pred)
    infer_t = time.time() - t0

    results["TimesFM Finetuned"] = {
        "RMSE": rmse_ft,
        "Training_Time": train_time,
        "Inference_Time": infer_t
    }

    print("\nFine-Tuned Evaluation", flush=True)
    print(f"Fine-Tuned RMSE: {rmse_ft:.4f}", flush=True)

    # Plot Fine-Tuned
    if plot_graph:
        plot_predictions(
            model=finetune_model,
            test_dataset=test_dataset,
            scaler=scaler,
            title=f"TimesFM Fine-Tuned Prediction on {path_[:-4]} Data",
            save_path="finetuned_predictions" + path_ + ".png",
        )
    
    # Comparison
    print("\nPerformance Comparison (RMSE)")
    print(f"Zero-Shot (No Finetuning) RMSE: {rmse:.4f}")
    print(f"Fine-Tuned RMSE: {rmse_ft:.4f}")
    
    if rmse_ft < rmse:
        print(f"Fine-tuning improved performance by: {((rmse - rmse_ft) / rmse) * 100:.2f}%")
    else:
        print("Fine-tuning did not improve performance or made it worse (may need more epochs/data).")


    if more_models:
        print("\nEvaluating additional models (ARIMA, LSTM, Persistence)...", flush=True)
        device = next(finetune_model.parameters()).device
        all_preds = {"ARIMA": [], "Persistence": [], "LSTM Multi Output": [], 
                     "LSTM Multi Model": [], "LSTM AR": []}
        all_times = {"ARIMA": [0.0], "Persistence": [0.0], "LSTM Multi Output": [0.0, 0.0], 
                     "LSTM Multi Model": [0.0, 0.0], "LSTM AR": [0.0, 0.0]}
        all_targets = []

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize empty parameters for ARIMA optimal model
        params = None

        # Load trained LSTM models
        model_mo, all_times["LSTM Multi Output"][1] = train_multi_output(train_dataset, horizon_len, device)
        models_list, all_times["LSTM Multi Model"][1] = train_multi_model(train_dataset, horizon_len, device)
        model_ar, all_times["LSTM AR"][1] = train_autoregressive(train_dataset, horizon_len, device)

        with torch.no_grad():
            for x_context, x_padding, freq, x_future in test_loader:
                x_context = x_context.to(device)
                x_padding = x_padding.to(device)
                freq = freq.to(device)
                x_future = x_future.to(device)

                # Inverse-transform targets
                target_last = x_future[:, -num_pred:].cpu().numpy().reshape(-1, 1)
                all_targets.extend(scaler.inverse_transform(target_last).flatten())

                # ARIMA
                s_arima = time.time()
                try:
                    for i in range(x_context.shape[0]):
                        context_np = x_context[i].cpu().numpy().flatten()

                        if params is None:
                            arima_pred, params = statsmodels_auto_arima_(context_np, horizon_len)
                        else:
                            arima_pred, _ = statsmodels_auto_arima_(context_np, horizon_len, params=params)
                        arima_unscaled = scaler.inverse_transform(arima_pred[-num_pred:].reshape(-1, 1)).flatten()
                        all_preds["ARIMA"].extend(arima_unscaled)
                    
                except Exception as e:
                    print("ARIMA failed for a batch:", e)
                all_times["ARIMA"][0] += time.time() - s_arima

                # Persistence
                s_pers = time.time()
                for i in range(x_context.shape[0]):
                    pers_pred = predict_persistence(x_context[i].cpu().numpy(), horizon_len)
                    pers_unscaled = scaler.inverse_transform(pers_pred[-num_pred:].reshape(-1,1)).flatten()
                    all_preds["Persistence"].extend(pers_unscaled)
                all_times["Persistence"][0] += time.time() - s_pers

                # LSTM Multi Output
                x_lstm_input = x_context.unsqueeze(-1)

                s_inf = time.time()
                model_mo.eval()
                out_mo = model_mo(x_lstm_input)[:, -num_pred:]
                all_preds["LSTM Multi Output"].extend(
                    scaler.inverse_transform(out_mo.cpu().numpy().reshape(-1, 1)).flatten()
                )
                all_times["LSTM Multi Output"][0] += time.time() - s_inf

                # LSTM Multi Model
                s_inf = time.time()
                preds_multi = []

                for i in range(horizon_len):
                    models_list[i].eval()
                    step_pred = models_list[i](x_lstm_input).detach() # [Batch, 1]
                    preds_multi.append(step_pred)

                out_mm = torch.cat(preds_multi, dim=1) 

                last_step_mm = out_mm[:, -num_pred:].cpu().numpy().reshape(-1, 1)
                all_preds["LSTM Multi Model"].extend(
                    scaler.inverse_transform(last_step_mm).flatten()
                )
                all_times["LSTM Multi Model"][0] += time.time() - s_inf

                # LSTM AutoRegressive
                s_inf = time.time()
                out_ar = model_ar(x_lstm_input)[:, -num_pred:]
                all_preds["LSTM AR"].extend(
                    scaler.inverse_transform(out_ar.cpu().numpy().reshape(-1, 1)).flatten()
                )
                all_times["LSTM AR"][0] += time.time() - s_inf


        p, d, q = params
        print("ARIMA model selected:")
        print(f"Order (p,d,q): ({p}, {d}, {q})")

        for model_name, preds in all_preds.items():
            preds_arr = np.array(preds)
            targets_arr = np.array(all_targets)
            rmse = np.sqrt(np.mean((targets_arr - preds_arr) ** 2))
            if len(all_times[model_name]) > 1:
                results[model_name] = {
                    "RMSE": float(rmse),
                    "Training_Time": float(all_times[model_name][1]),
                    "Inference_Time": float(all_times[model_name][0])
                }
            else:
                results[model_name] = {"RMSE": float(rmse), 
                                       "Inference_Time": float(all_times[model_name][0])}


    print("\nGlobal Metrics & Time Comparison:")
    for model, metrics in results.items():
        print(f"{model}: {metrics}")
        
    return results

