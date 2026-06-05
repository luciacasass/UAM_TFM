import numpy as np
import torch
import time
import os
import logging

from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader, Subset

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

from utils_folder.data_util import *
from utils_folder.comp_util import *

seed_everything(42)

import warnings
warnings.filterwarnings("ignore")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ID = "google/timesfm-2.0-500m-pytorch"


# Model creation
def get_model(context_len: int, horizon_len: int, load_weights: bool = False):
    hf_path = snapshot_download(REPO_ID)
    hparams = TimesFmHparams(
        backend=str(DEVICE),
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=50,
        use_positional_embedding=True,
        context_len=1024
    )
    tfm = TimesFm(hparams=hparams,
                  checkpoint=TimesFmCheckpoint(huggingface_repo_id=REPO_ID))
    model = PatchedTimeSeriesDecoder(tfm._model_config)

    ckpt = torch.load(f"{hf_path}/torch_model.ckpt", weights_only=True, map_location=DEVICE)
    model.load_state_dict(ckpt)
    return model.to(DEVICE).eval()




def get_rmse(model: PatchedTimeSeriesDecoder, 
             test_dataset: Dataset, 
             scaler: StandardScaler, 
             test_raw_aligned: np.ndarray,  # New parameter
             horizon: int,
             num_pred: int, 
             seasonal_comp: Optional[np.ndarray] = None, 
             train_size: int = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    """Computes RMSE evaluating just the last step in horizon H"""
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_targets = [], []
    batch_size_val = 32
    loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False)

    with torch.no_grad():
        for b_idx, (x_ctx, x_pad, freq, x_future) in enumerate(loader):
            x_ctx, x_pad, freq = x_ctx.to(device), x_pad.to(device), freq.to(device)
            
            outputs = model(x_ctx, x_pad.float(), freq)
            predictions_mean = outputs[..., 0]
            last_patch_pred = predictions_mean[:, -1, :]

            # last_patch_pred has shape [Num Batchs, 128]
            step_preds = last_patch_pred[:, horizon-1:horizon+num_pred-1].cpu().numpy()
            orig_shape = step_preds.shape
            p_unscaled = scaler.inverse_transform(step_preds.reshape(-1, 1)).reshape(orig_shape)

            # Add Seasonality based on batch position
            for i in range(x_ctx.shape[0]):
                idx = (b_idx * batch_size_val) + i
                target_offset = test_dataset.context_length + (horizon - 1)
                if seasonal_comp is not None and np.any(seasonal_comp):
                    # Calculate the exact window in the original series
                    # sample_idx = i * batch_size + b
                    start_in_series = train_size + idx + target_offset
                    season_slice = seasonal_comp[start_in_series : start_in_series + num_pred]

                    p_unscaled[i, :] += season_slice

                target_start = idx + target_offset
                target_raw = test_raw_aligned[target_start : target_start + num_pred]

                all_preds.append(p_unscaled[i].flatten())
                all_targets.append(target_raw.flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return (
        np.sqrt(mean_squared_error(all_targets, all_preds)),
        np.array(all_preds),
        np.array(all_targets),
    )



def evaluate_lstm_models(model_mo, models_list, model_ar,
                         test_loader, scaler, test_raw_aligned, 
                         horizon, num_pred, device, seasonal_comp=None, train_size=0):

    preds = {
        "LSTM Multi Output": [],
        "LSTM Multi Model": [],
        "LSTM AR": []
    }
    times = {
            "LSTM Multi Output": 0.0,
            "LSTM Multi Model": 0.0,
            "LSTM AR": 0.0
        }
    
    targets = []
    context_length = test_loader.dataset.context_length
    batch_size_loader = test_loader.batch_size

    with torch.no_grad():
        for b_idx, batch_data in enumerate(test_loader):
            # Unpack based on TimeSeriesDataset __getitem__
            x_context, _, _, _ = batch_data 
            
            x_context = x_context.to(device)
            x_lstm_input = x_context.unsqueeze(-1)
            batch_size = x_context.shape[0]

            # Model Inferences
            start_mo = time.time()
            mo_pred = model_mo(x_lstm_input)[:, -num_pred:]
            times["LSTM Multi Output"] += (time.time() - start_mo)

            start_ar = time.time()
            ar_pred = model_ar(x_lstm_input)[:, -num_pred:]
            times["LSTM AR"] += (time.time() - start_ar)
            
            preds_multi = []
            start_mm = time.time()
            for m in models_list:
                preds_multi.append(m(x_lstm_input).detach())
            mm_pred = torch.cat(preds_multi, dim=1)[:, -num_pred:]
            times["LSTM Multi Model"] += (time.time() - start_mm)

            # Process each sample in the batch for Reconstruction and Truth
            for i in range(batch_size):
                idx = (b_idx * batch_size_loader) + i

                target_offset = context_length + (horizon - 1)
                
                # Truth window starts after context
                target_start = idx + target_offset
                t_raw = test_raw_aligned[target_start : target_start + num_pred].flatten()
                targets.extend(t_raw.tolist())

                # Handle Seasonal Reconstruction for Predictions
                s_slice = 0
                if seasonal_comp is not None and np.any(seasonal_comp):
                    # Align with the specific window in the global seasonal array
                    seas_start = train_size + idx + target_offset
                    s_slice = seasonal_comp[seas_start : seas_start + num_pred].flatten()

                # Inverse Scale + Add Seasonality
                def reconstruct(p_tensor):
                    p_np = p_tensor[i].cpu().numpy().reshape(-1, 1)
                    unscaled = scaler.inverse_transform(p_np).flatten()
                    return (unscaled + s_slice).tolist()

                preds["LSTM Multi Output"].extend(reconstruct(mo_pred))
                preds["LSTM Multi Model"].extend(reconstruct(mm_pred))
                preds["LSTM AR"].extend(reconstruct(ar_pred))

    # Calculate RMSEs
    rmses = {}
    targets_np = np.array(targets)
    for key in preds:
            preds_np = np.array(preds[key])
            rmses[key] = {
                "RMSE": np.sqrt(mean_squared_error(targets_np, preds_np)),
                "Inference_Time": times[key],
                "Targets": targets_np,
                "Predictions": preds_np
            }

    return rmses




def compare_performance(context_len: int, 
                        horizon_len: int,
                        max_horizon_len: int = 0,
                        num_pred: int = 1,
                        freq_type: int = 0, 
                        finetune_model = None,
                        n_epoch = 0,
                        pre_train_ft = 0,
                        real_data: bool = False,
                        file_path:str = None,
                        more_models: bool = False,
                        decompose: bool = False, 
                        period: int = 7,
                        arima_params=None,
                        lstm_variance: bool = False,
                        lstm_runs: int = 10):
    """Loads and compares zero-shot vs fine-tuned TimesFM performance."""

    seed_everything(42)
    
    # Update horizon_len (number of predictions to be made in order to extract
    # num_pred points with horizon H)
    horizon_len = horizon_len + num_pred - 1
    if max_horizon_len == 0:
        max_horizon_len = horizon_len
    max_horizon_len = max_horizon_len + num_pred - 1

    # Load Data
    if real_data:
        if file_path:
            print(f"Loading Energy Demand Dataset...", flush=True)
            train_dataset, test_dataset, scaler, seasonal_comp, train_size, test_raw_aligned = get_demand_data(
                file_path=file_path,
                context_len=context_len,
                horizon_len=horizon_len,
                max_horizon_len=max_horizon_len,
                freq_type=freq_type,
                decompose=decompose, period=period
                )
        else:
            print(f"Loading Real Dataset...", flush=True)
            train_dataset, test_dataset, scaler, seasonal_comp, train_size, test_raw_aligned = get_real_data(
                context_len=context_len,
                horizon_len=horizon_len,
                max_horizon_len=max_horizon_len,
                freq_type=freq_type,
                decompose=decompose, period=period
                )
    else:
        print("Generating Synthetic Sine Dataset...", flush=True)
        train_dataset, test_dataset, scaler, seasonal_comp, train_size, test_raw_aligned = get_data_synthetic(
            context_len=context_len,
            horizon_len=horizon_len,
            max_horizon_len=max_horizon_len,
            freq_type=freq_type,
            decompose=decompose, period=period
        )

    # Get validation dataset
    train_sub, val_sub = split_train_val(train_dataset)

    results = {}

    # Zero-Shot (No Fine-Tuning) Performance
    print("\nZero-Shot (Pre-trained) Evaluation", flush=True)
    model = get_model(context_len, horizon_len)

    t0 = time.time()
    rmse, preds_zs, targets_zs = get_rmse(model, test_dataset, scaler, 
                                          test_raw_aligned=test_raw_aligned, 
                                          horizon=horizon_len, num_pred=num_pred,
                                          seasonal_comp=seasonal_comp, train_size=train_size)
    infer_t = time.time() - t0

    results["TimesFM ZeroShot"] = {
        "RMSE": rmse,
        "Inference_Time": infer_t,
        "Predictions": preds_zs.tolist(),
        "Targets": targets_zs.tolist()
    }
    print(f"Zero-Shot RMSE: {rmse:.4f}", flush=True)
    del model


    def sliced_mse_loss(preds, targets):
        t = targets.view(targets.shape[0], -1)
        p = preds[:, :t.shape[1]]
        
        return torch.mean((p - t)**2)
    
    # Fine-Tuning
    if finetune_model is None:
        # Complete process
        print("\nFine-Tuning Process", flush=True)
        finetune_model = get_model(context_len, horizon_len)

        config = FinetuningConfig(
            batch_size=64,
            num_epochs=4,
            learning_rate=1e-4,
            use_wandb=False,
            freq_type=freq_type,
            log_every_n_steps=10,
            val_check_interval=0.5,
            use_quantile_loss=False
        )

        finetuner = TimesFMFinetuner(finetune_model, config, loss_fn=sliced_mse_loss)

        print("Training...")
        start_train = time.time()
        finetuner.finetune(train_sub, val_sub)
        train_ft = time.time() - start_train + pre_train_ft

    elif n_epoch > 0:
        # Partial
        print("\nFine-Tuning Process 1 epoch", flush=True)

        print("Training...")
        start_train = time.time()

        config = FinetuningConfig(
            batch_size=64,
            num_epochs=n_epoch,
            learning_rate=1e-4,
            use_wandb=False, # Set to True to enable Weights & Biases logging
            freq_type=freq_type,
            log_every_n_steps=10,
            val_check_interval=0.5,
            use_quantile_loss=False # Recommended True loss for TimesFM
        )

        def sliced_mse_loss(preds, targets):
            t = targets.view(targets.shape[0], -1)
            p = preds[:, :horizon_len]
            return torch.mean((p - t)**2)
        
        finetuner = TimesFMFinetuner(finetune_model, config, loss_fn=sliced_mse_loss)
        finetuner.finetune(train_sub, val_sub)

        train_ft = time.time() - start_train + pre_train_ft

    else:
        # predict over previously finetuned model
        train_ft = pre_train_ft

    t0 = time.time()
    rmse_ft, preds_ft, targets_ft = get_rmse(finetune_model, test_dataset, scaler, 
                                             test_raw_aligned=test_raw_aligned, 
                                             horizon=horizon_len, num_pred=num_pred,
                                             seasonal_comp=seasonal_comp, train_size=train_size)
    infer_t = time.time() - t0

    results["TimesFM Finetuned"] = {
        "RMSE": rmse_ft,
        "Training_Time": train_ft,
        "Inference_Time": infer_t,
        "Predictions": preds_ft.tolist(),
        "Targets": targets_ft.tolist()
    }

    print("\nFine-Tuned Evaluation", flush=True)
    print(f"Fine-Tuned RMSE: {rmse_ft:.4f}", flush=True)

    
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
        all_preds = {"ARIMA": [], "Persistence": [], "Dummy Regressor": [], "LSTM Multi Output": [], 
                     "LSTM Multi Model": [], "LSTM AR": []}
        all_times = {"ARIMA": [0.0, 0.0], "Persistence": [0.0], "Dummy Regressor": [0.0], "LSTM Multi Output": [0.0, 0.0], 
                     "LSTM Multi Model": [0.0, 0.0], "LSTM AR": [0.0, 0.0]}
        all_targets = {
            "ARIMA": [],
            "Persistence": [],
            "Dummy Regressor": [],
            "LSTM Multi Output": [],
            "LSTM Multi Model": [],
            "LSTM AR": []
        }
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


        # Load trained LSTM models
        if not lstm_variance:
            model_mo, all_times["LSTM Multi Output"][1] = train_multi_output(train_dataset, horizon_len, device)
            models_list, all_times["LSTM Multi Model"][1] = train_multi_model(train_dataset, horizon_len, num_pred, device)
            model_ar, all_times["LSTM AR"][1] = train_autoregressive(train_dataset, horizon_len, device)

        # Find ARIMA params
        if arima_params is None:
            print("Finding optimal ARIMA parameters on training data...")
            train_series = pd.Series(train_dataset.series)

            _, params = statsmodels_auto_arima_(train_series)
            print(f"Optimal Parameters Found: {params}")
        else:
            params = arima_params


        # Train Dummy Regressor
        train_x = np.array([s[0] for s in train_dataset])
        train_y = np.array([s[3] for s in train_dataset])[:, -num_pred:]
        dummy_regr = DummyRegressor(strategy="mean")
        dummy_regr.fit(train_x, train_y)


        with torch.no_grad():
            for b_idx, (x_context, x_padding, freq, x_future) in enumerate(test_loader):
                x_context = x_context.to(device)
                x_future = x_future.to(device)
                batch_size = x_context.shape[0]

                # Compute seasonal slices per batch
                batch_season_slices = []
                current_batch_targets = []
                target_offset = test_dataset.context_length + (horizon_len - 1)
                
                for i in range(batch_size):
                    # Calculate the absolute position in the test series
                    sample_idx = (b_idx * test_loader.batch_size) + i
                    # Prediction/Target window starts after the context
                    target_start = sample_idx + target_offset
                    
                    # Get the Actual Ground Truth (Raw values)
                    t_raw = test_raw_aligned[target_start : target_start + num_pred].flatten()
                    current_batch_targets.append(t_raw)

                    # Get the Seasonal Slice (for reconstructing predictions only)
                    if decompose and seasonal_comp is not None:
                        start_in_seas = train_size + sample_idx + target_offset
                        batch_season_slices.append(seasonal_comp[start_in_seas : start_in_seas + num_pred])
                    else:
                        batch_season_slices.append(np.zeros(num_pred))

                for i in range(batch_size):
                    target_vals = current_batch_targets[i].tolist()
                    for k in all_targets.keys():
                        if k != "ARIMA":
                            all_targets[k].extend(target_vals)


                # ARIMA
                s_arima = time.time()
                for i in range(batch_size):
                    try:
                        context_np = x_context[i].cpu().numpy().flatten()

                        arima_pred, _ = statsmodels_auto_arima_(context_np, horizon_len, params=params)
                        arima_unscaled = scaler.inverse_transform(arima_pred[-num_pred:].reshape(-1, 1)).flatten()

                        all_preds["ARIMA"].extend(arima_unscaled + batch_season_slices[i])
                        all_targets["ARIMA"].extend(current_batch_targets[i].tolist())
                    
                    except Exception as e:
                        print(f"ARIMA failed at batch {b_idx}: {e}")

                all_times["ARIMA"][0] += time.time() - s_arima

                # Persistence
                s_pers = time.time()
                for i in range(batch_size):
                    pers_pred = predict_persistence(x_context[i].cpu().numpy(), horizon_len)
                    pers_unscaled = scaler.inverse_transform(pers_pred[-num_pred:].reshape(-1,1)).flatten()
                    all_preds["Persistence"].extend(pers_unscaled + batch_season_slices[i])
                all_times["Persistence"][0] += time.time() - s_pers


                # Dummy Regressor
                s_dummy = time.time()
                for i in range(batch_size):
                    dummy_pred = predict_dummy(dummy_regr, x_context[i].cpu().numpy())
                    dummy_unscaled = scaler.inverse_transform(dummy_pred[-num_pred:].reshape(-1,1)).flatten()
                    all_preds["Dummy Regressor"].extend(dummy_unscaled + batch_season_slices[i])
                all_times["Dummy Regressor"][0] += time.time() - s_dummy

                if not lstm_variance:
                    # LSTM Multi Output
                    x_lstm_input = x_context.unsqueeze(-1)
                    def add_seasonal_to_lstm(preds_tensor, model_key, timer_start):
                        # preds_tensor shape: [Batch, num_pred]
                        preds_np = preds_tensor.cpu().numpy()
                        for i in range(batch_size):
                            unscaled = scaler.inverse_transform(preds_np[i].reshape(-1, 1)).flatten()
                            all_preds[model_key].extend(unscaled + batch_season_slices[i])
                        all_times[model_key][0] += time.time() - timer_start

                    s_inf = time.time()
                    model_mo.eval()
                    out_mo = model_mo(x_lstm_input)[:, -num_pred:]
                    add_seasonal_to_lstm(out_mo, "LSTM Multi Output", s_inf)

                    # LSTM Multi Model
                    s_inf = time.time()
                    preds_multi = []

                    for m in models_list:
                        m.eval()
                        step_pred = m(x_lstm_input).detach() # [Batch, 1]
                        preds_multi.append(step_pred)

                    out_mm = torch.cat(preds_multi, dim=1)[:, -num_pred:]
                    add_seasonal_to_lstm(out_mm, "LSTM Multi Model", s_inf)

                    # LSTM AutoRegressive
                    s_inf = time.time()
                    out_ar = model_ar(x_lstm_input)[:, -num_pred:]
                    add_seasonal_to_lstm(out_ar, "LSTM AR", s_inf)


        if lstm_variance:
            history = {
                "LSTM Multi Output": {"rmse": [], "time": [], "preds": []},
                "LSTM Multi Model": {"rmse": [], "time": [], "preds": []},
                "LSTM AR": {"rmse": [], "time": [], "preds": []}
            }
            
            train_times = {
                "LSTM Multi Output": [],
                "LSTM Multi Model": [],
                "LSTM AR": []
            }

            for run in range(lstm_runs):
                print(f"Run {run+1}/{lstm_runs}...", end="\r")
                m_mo, t_mo = train_multi_output(train_dataset, horizon_len, device, seed=42+run)
                m_mm, t_mm = train_multi_model(train_dataset, horizon_len, num_pred, device, seed=42+run)
                m_ar, t_ar = train_autoregressive(train_dataset, horizon_len, device, seed=42+run)

                train_times["LSTM Multi Output"].append(t_mo)
                train_times["LSTM Multi Model"].append(t_mm)
                train_times["LSTM AR"].append(t_ar)

                rmses_dict = evaluate_lstm_models(
                    m_mo, m_mm, m_ar,
                    test_loader, scaler, test_raw_aligned,
                    horizon_len, num_pred, device,
                    seasonal_comp=seasonal_comp, train_size=train_size
                )

                if run == 0:
                    targets_ground_truth = np.array(rmses_dict["LSTM Multi Output"]["Targets"])

                for key in history:
                    history[key]["rmse"].append(rmses_dict[key]["RMSE"])
                    history[key]["time"].append(rmses_dict[key]["Inference_Time"])
                    history[key]["preds"].append(np.array(rmses_dict[key]["Predictions"]))

                del m_mo, m_mm, m_ar


            # Procesar estadísticas finales
            lstm_results = {}
            for key in history:
                rmse_arr = np.array(history[key]["rmse"])
                time_arr = np.array(history[key]["time"])

                pred_stack = np.stack(history[key]["preds"])
                predictions_mean_per_point = pred_stack.mean(axis=0)

                ensemble_rmse = np.sqrt(mean_squared_error(targets_ground_truth,\
                                                           predictions_mean_per_point))

                lstm_results[key] = {
                    "Mean_RMSE_Individual": float(rmse_arr.mean()),
                    "Ensemble_RMSE": float(ensemble_rmse),
                    "Variance": float(rmse_arr.var()),
                    "Std": float(rmse_arr.std()),
                    "Min_RMSE": float(rmse_arr.min()),
                    "Max_RMSE": float(rmse_arr.max()),
                    "Inference_Time": time_arr,
                    "Training_Time": train_times[key],
                    "Predictions_Mean": predictions_mean_per_point.tolist(),
                    "All_RMSE_Runs": rmse_arr.tolist()
                }

                all_preds[key] = lstm_results[key]["Predictions_Mean"]
                all_times[key][0] = np.array(lstm_results[key]["Inference_Time"]).mean()
                all_times[key][1] = np.array(lstm_results[key]["Training_Time"]).mean()


        p, d, q = params
        print("ARIMA model selected:")
        print(f"Order (p,d,q): ({p}, {d}, {q})")

        for model_name, preds in all_preds.items():
            preds_arr = np.array(preds)
            targets_arr = np.array(all_targets[model_name])
            rmse = np.sqrt(np.mean((targets_arr - preds_arr) ** 2))
            results[model_name] = {
                "RMSE": float(rmse),
                "Inference_Time": float(all_times[model_name][0]),
                "Predictions": preds_arr.tolist(),
                "Targets": targets_arr.tolist(),
            }
            if len(all_times[model_name]) > 1:
                results[model_name]["Training_Time"] = float(all_times[model_name][1])
            if lstm_variance and model_name in lstm_results:
                results[model_name]["Other Metrics"] = lstm_results[model_name]


    print("\nGlobal Metrics & Time Comparison:")
    summary = {
        model: {
            k: v for k, v in metrics.items()
            if k in ["RMSE", "Training_Time", "Inference_Time"]
        }
        for model, metrics in results.items()
    }
    print(summary)
        
    return results
