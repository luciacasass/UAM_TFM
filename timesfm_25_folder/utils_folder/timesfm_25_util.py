import numpy as np
import time
from typing import Any, Dict, List, Optional, Union

from sklearn.metrics import mean_squared_error

def get_timesfm_metrics(model_tsfm: Any,
                        test_dataset: Any,
                        scaler: Any,
                        seasonal_comp: Optional[np.ndarray] = None,
                        train_size:int = 0
                        ) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """
    Computes RMSE and inference time for TimesFM 2.5 model over the test dataset.
    """
    all_preds = []
    all_targets = []


    inputs_list = [test_dataset[i][0].numpy() for i in range(len(test_dataset))]
    targets_list = [test_dataset[i][3].numpy() for i in range(len(test_dataset))]
    
    start_time = time.time()
    point_forecast, _ = model_tsfm.forecast(
        inputs=inputs_list,
        horizon=test_dataset.horizon_length,
        # freq=[test_dataset.freq_type] * len(inputs_list)
    )
    
    end_time = time.time()
    execution_time = end_time - start_time

    # Process each sample to descale and store
    for i in range(len(point_forecast)):

        pred_unscaled = scaler.inverse_transform(point_forecast[i].reshape(-1, 1)).flatten()
        target_unscaled = scaler.inverse_transform(targets_list[i].reshape(-1, 1)).flatten()

        if seasonal_comp is not None:
            # Calculate where this specific forecast window ends in the global series
            end_in_series = train_size + i + test_dataset.context_length + test_dataset.horizon_length
            
            # Slice the seasonal component for the horizon duration
            season_slice = seasonal_comp[end_in_series - test_dataset.horizon_length : end_in_series]
            
            pred_unscaled += season_slice
            target_unscaled += season_slice
        
        all_preds.append(pred_unscaled[-1])
        all_targets.append(target_unscaled[-1])
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)

    print(f"Validation Metrics")
    print(f"RMSE: {rmse:.4f}")
    print(f"Inference time: {execution_time:.4f} seconds\n")
    
    return {"TimesFM 2.5": {
        "RMSE": float(rmse),  
        "Inference_Time": float(execution_time),
        "Predictions": all_preds.tolist(),
        "Targets": all_targets.tolist()}
    }