# Load libraries
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from typing import List, Dict, Optional



def extract_experiment_data(data: Dict) -> pd.DataFrame:
    """Flattens experiment JSON data into a pandas DataFrame for detailed analysis.

    Args:
        data: The dictionary loaded from an experiment JSON file.

    Returns:
        A DataFrame where each row represents a single time-step prediction, 
        including columns for Model, Horizon, Step, Prediction, Target, and Error.
    """
    rows = []
    for exp in data['experiments']:
        horizon = exp['horizon']
        context = exp['context_length']

        # Iterate over each model
        for model_name, results in exp['metrics'].items():
            preds = results['Predictions']
            targets = results["Targets"]
            rmse = results['RMSE']
            train_time = results.get('Training_Time', 0)
            inf_time = results.get('Inference_Time', 0)
            
            # Store each point
            for i, (p, t) in enumerate(zip(preds, targets)):
                rows.append({
                    "Model": model_name,
                    "Context": context,
                    "Horizon": horizon,
                    "Step": i,
                    "RMSE": rmse,
                    "Prediction": p,
                    "Target": t,
                    "Error": p - t,
                    "Training_Time": train_time,
                    "Inference_Time": inf_time
                })
    return pd.DataFrame(rows)


def plot_metric_evolution(
        df: pd.DataFrame,
        models: List[str] = None, 
        valid_horizon_keys: List[int] = None, 
        metric_name: str = "RMSE", 
        log_scale: bool = True
        ) -> None:
    """Plots the evolution of a specific metric across different forecast horizons.

    Args:
        session_results: Dictionary containing the metric results.
        models: List of model names to plot.
        valid_horizon_keys: List of horizon keys to include on the X-axis.
        metric_name: The name of the metric to visualize (e.g., 'RMSE').
        log_scale: Whether to use a logarithmic scale for the Y-axis.
    """
    df_filtered = df.copy()
    if models:
        df_filtered = df_filtered[df_filtered['Model'].isin(models)]
    else:
        models = df_filtered["Model"].unique()
    if valid_horizon_keys:
        df_filtered = df_filtered[df_filtered['Horizon'].isin(valid_horizon_keys)]
    else:
        valid_horizon_keys = df_filtered["Horizon"].unique()

    plt.figure(figsize=(9, 5))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for i, model in enumerate(models):
        vals = []

        for h in valid_horizon_keys:
            subset = df[
                (df['Model'] == model) &
                (df['Horizon'] == h)]

            if not subset.empty:
                vals.append(subset[metric_name].values[0])
            else:
                vals.append(np.nan)

        plt.plot(
            valid_horizon_keys,
            vals,
            marker=markers[i % len(markers)],
            label=model,
            linewidth=1.6,
            markersize=5
        )

    plt.title(f'{metric_name} Evolution Across Forecast Horizons')
    plt.xlabel('Forecast Horizon (steps)')
    plt.ylabel(metric_name)

    if log_scale:
        plt.yscale('log')

    plt.xticks(valid_horizon_keys)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Model Name', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()



def plot_time_metrics(
    df: pd.DataFrame, 
    models: List[str] = None, 
    valid_horizon_keys: List[int] = None, 
    title_suffix: str = 'Seconds'
) -> None:
    """Generates bar charts comparing Training, Inference, and Total time per model.

    Args:
        session_results: Dictionary containing the time metric results.
        models: List of model names to compare.
        valid_horizon_keys: Horizon keys to display on the X-axis.
        title_suffix: Unit label for the Y-axis (e.g., 'Seconds', 'ms').
    """

    # Aggregate per Model + Horizon
    df_filtered = df.copy()
    if models:
        df_filtered = df_filtered[df_filtered['Model'].isin(models)]
    else:
        models = df_filtered["Model"].unique()
    if valid_horizon_keys:
        df_filtered = df_filtered[df_filtered['Horizon'].isin(valid_horizon_keys)]
    else:
        valid_horizon_keys = df_filtered["Horizon"].unique()

    agg = df_filtered.groupby(['Model', 'Horizon']).agg({
        'Training_Time': 'first',
        'Inference_Time': 'first'
    }).reset_index()

    agg['Total_Time'] = agg['Training_Time'] + agg['Inference_Time']

    metrics = ['Training_Time', 'Inference_Time', 'Total_Time']
    horizons = sorted(agg['Horizon'].unique())
    models = agg['Model'].unique()

    x = np.arange(len(horizons))
    width = 0.15

    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(models):
            vals = [
                agg[(agg['Model'] == model) & (agg['Horizon'] == h)][metric].values[0]
                for h in horizons
            ]
            pos = x + (i - len(models)/2) * width + width/2
            ax.bar(pos, vals, width, label=model)

        ax.set_title(metric.replace('_', ' '))
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in horizons])
        ax.set_ylabel(title_suffix)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(models))

    plt.tight_layout()
    plt.show()
    


def plot_rmse_vs_time(
        df: pd.DataFrame,
        models: List[str] = None, 
        valid_horizon_keys: List[int] = None
        ) -> None:
    """Creates a scatter plot showing the trade-off between Mean RMSE and Total Time.

    Args:
        session_results: Dictionary containing metrics and timing data.
        models: List of model names.
        valid_horizon_keys: Horizon keys to aggregate data from.
    """

    # Time per Model + Horizon
    df_filtered = df.copy()
    if models:
        df_filtered = df_filtered[df_filtered['Model'].isin(models)]
    else:
        models = df_filtered["Model"].unique()
    if valid_horizon_keys:
        df_filtered = df_filtered[df_filtered['Horizon'].isin(valid_horizon_keys)]
    else:
        valid_horizon_keys = df_filtered["Horizon"].unique()
        
    time_df = df_filtered.groupby(['Model', 'Horizon']).agg({
        "RMSE": 'first',
        'Training_Time': 'first',
        'Inference_Time': 'first'
    }).reset_index()

    time_df['Total_Time'] = time_df['Training_Time'] + time_df['Inference_Time']

    # Aggregate per model
    summary = time_df.groupby('Model').agg({
        'RMSE': 'mean',
        'Total_Time': 'sum'
    }).reset_index()

    # Plot
    plt.figure(figsize=(9, 5))

    for _, row in summary.iterrows():
        plt.scatter(row['RMSE'], row['Total_Time'], s=80, label=row['Model'])

    plt.xlabel('Mean RMSE Across Horizons')
    plt.ylabel('Total Time (sec)')
    plt.title('Mean RMSE vs Total Time per Model')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Model')

    plt.tight_layout()
    plt.show()



def plot_seasonal_comparison(
    normal_paths: list[str],
    seasonal_paths: list[str],
    first_label: str = "Normal",
    second_label: str = "Seasonal"
) -> None:
    """Compares RMSE between normal and seasonal experiments combining multiple result files."""

    def load_and_merge(paths):
        experiments = []
        for path in paths:
            with open(path, "r") as f:
                data = json.load(f)
                experiments.extend(data["experiments"])
        return experiments

    # Load merged experiments
    normal_experiments = load_and_merge(normal_paths)
    seasonal_experiments = load_and_merge(seasonal_paths)

    # Filter by context length
    filtered_normal = [
        exp for exp in normal_experiments
        if exp["context_length"] in [256, 288]
    ]

    filtered_seasonal = [
        exp for exp in seasonal_experiments
        if exp["context_length"] in [256, 288]
    ]

    # Build sessions
    session_normal = {}

    for exp in filtered_normal:
        h_key = f"horizon_{exp['horizon']}"
        if h_key not in session_normal:
            session_normal[h_key] = {}

        session_normal[h_key].update(exp["metrics"])

    session_seasonal = {}

    for exp in filtered_seasonal:
        h_key = f"horizon_{exp['horizon']}"
        if h_key not in session_seasonal:
            session_seasonal[h_key] = {}

        session_seasonal[h_key].update(exp["metrics"])

    horizon_keys = sorted(session_normal.keys(), key=lambda x: int(x.split('_')[-1]))
    horizons = [int(x.split('_')[-1]) for x in horizon_keys]

    models = list(session_normal[horizon_keys[0]].keys())

    rmse_normal = {
        model: [session_normal[h][model]['RMSE'] for h in horizon_keys]
        for model in models
    }

    rmse_seasonal = {
        model: [session_seasonal[h][model]['RMSE'] for h in horizon_keys]
        for model in models
    }

    # Plot
    n_models = len(models)
    ncols = 3
    nrows = math.ceil(n_models / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(15, 4 * nrows),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]

        ax.plot(
            horizons,
            rmse_normal[model],
            linestyle='-',
            marker='o',
            linewidth=1.8,
            markersize=4,
            color=color,
            label=first_label
        )

        ax.plot(
            horizons,
            rmse_seasonal[model],
            linestyle='--',
            marker='o',
            linewidth=1.8,
            markersize=4,
            color=color,
            label=second_label
        )

        ax.set_title(model)
        ax.set_yscale('log')
        ax.set_xticks(horizons)
        ax.grid(True, linestyle='--', alpha=0.5)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel('Forecast Horizon (steps)')
    fig.supylabel('RMSE')
    fig.suptitle(f'RMSE Comparison per Model ({first_label} vs {second_label})')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()



def plot_prediction_vs_real(df: pd.DataFrame, 
                            horizon_val: int, 
                            model_name: Optional[List[str]] = None
                            ) -> None:
    """Plots the actual target series against predictions for a specific horizon.

    Args:
        df: DataFrame generated by `extract_experiment_data`.
        horizon_val: The specific integer horizon to visualize.
        model_name: Optional list of model names to filter. If None, plots all models.

    Raises:
        ValueError: If the specified model_name is not found in the data.
    """
    subset = df[df['Horizon'] == horizon_val].copy()

    # If a specific model is provided, filter
    if model_name is not None:
        subset = subset[subset['Model'].isin(model_name)]
        if subset.empty:
            raise ValueError(f"Model '{model_name}' not found for horizon {horizon_val}")
        
    real_series = (
        subset[['Step', 'Target']]
        .drop_duplicates()
        .sort_values('Step')
    )
    
    plt.figure(figsize=(12, 5))
    plt.plot(real_series['Step'], real_series['Target'], 
             label='Real (Target)', color='black', lw=2)
    
    for model in subset['Model'].unique():
        model_data = subset[subset['Model'] == model]
        plt.plot(model_data['Step'], model_data['Prediction'], ls='--', label=f'Pred {model}')
    
    plt.title(f'Predictions Comparison - Horizon {horizon_val}')
    plt.xlabel('Time point')
    plt.ylabel('Value')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.5)
    plt.show()



def plot_model_multi_horizon(df: pd.DataFrame, 
                             model_name: str, 
                             horizons_to_plot: List[int] = [1, 4, 8, 16, 24, 32]
                             ) -> None:
    """Visualizes how a single model performs across multiple horizons on the same plot.

    Args:
        df: DataFrame containing the experiment data.
        model_name: Name of the model to analyze.
        horizons_to_plot: List of horizons to include in the visualization.
    """
    # Filter by model
    model_df = df[df['Model'] == model_name].copy()
    
    if model_df.empty:
        print(f"No se encontraron datos para el modelo: {model_name}")
        return

    min_horizon = model_df['Horizon'].min()
    real_data = (
        model_df[model_df['Horizon'] == min_horizon][['Step', 'Target']]
        .sort_values('Step')
        .drop_duplicates()
    )

    plt.figure(figsize=(12, 5))

    plt.plot(real_data['Step'], real_data['Target'], 
             color='black', label='Real (Target)', lw=2, zorder=1)
    
    for h in horizons_to_plot:
        h_data = model_df[model_df['Horizon'] == h].sort_values('Step')
        
        if not h_data.empty:
            plt.plot(h_data['Step'], h_data['Prediction'], 
                     ls='--', alpha=0.8,
                     label=f'Pred H+{h}', zorder=2)

    plt.title(f'Performance per Horizon - Model: {model_name}')
    plt.xlabel('Time point')
    plt.ylabel('Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Horizons")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_error_distribution_grid(df: pd.DataFrame) -> None:
    """Generates a grid of KDE plots showing error distributions for each model and horizon.

    Args:
        df: DataFrame containing prediction errors.
    """
    
    g = sns.FacetGrid(df, col="Model", hue="Horizon", 
                      col_wrap=3, height=4, aspect=1.2, 
                      palette="viridis", sharex=True)
    
    # Map function kdeplot to each model
    g.map(sns.kdeplot, "Error", fill=True, alpha=0.3)
    g.add_legend(title="Horizon")
    # Error 0 line
    g.map(plt.axvline, x=0, color='red', linestyle='--', lw=1.5, alpha=0.8)

    for ax in g.axes.flat:
        ax.grid(True, linestyle="--", alpha=0.5)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Error (Predicted - Actual)", "Density")
    
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Error distribution per model and horizon')
    
    plt.show()



def plot_phase_space_grid(df: pd.DataFrame) -> None:
    """Generates a grid of scatter plots comparing Predictions vs. Targets (Phase Space).

    Args:
        df: DataFrame containing Targets and Predictions.
    """
    g = sns.FacetGrid(df, col="Model", col_wrap=3, height=4, sharex=True, sharey=True)
    
    # Map the plotting function
    def plot_diagonal_and_reg(x, y, **kwargs):
        # 45-degree line logic
        full_range = [min(x.min(), y.min()), max(x.max(), y.max())]
        plt.plot(full_range, full_range, color='red', linestyle='--', alpha=0.7, label='Perfect')
        # Regression plot
        sns.regplot(x=x, y=y, scatter_kws={'alpha':0.3, 's':10}, 
                    line_kws={'color':'blue', 'lw':1.5}, **kwargs)

    g.map(plot_diagonal_and_reg, "Target", "Prediction")

    for ax in g.axes.flat:
        ax.grid(True, linestyle="--", alpha=0.5)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Actual Values (Target)", "Predicted Values")
    
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Phase Space Comparison: Model Behavior Across Extremes')
    
    plt.show()