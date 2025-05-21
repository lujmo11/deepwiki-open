"""
This is a script to run hyperparameter optimization for a given neuron target model using Optuna.

Run `just install_dev` to install the required dependencies.

To run the script, you need to provide the following parameters:

- `dlc`: load case name. Used to retrieve default load case config.
- `df_path`: Path to the parquet file containing the training data.
- `targets`: List of target columns to optimize the model for.
- `model_name`: Name of the neuron target model to optimize.
- `features`: List of feature columns to use for training the model.
- `n_trials`: Number of trials to run for the optimization.
- `param_ranges`: Dictionary containing the hyperparameter ranges for the model. 
    
The keys in `param_ranges` are the hyperparameter names and the values are dictionaries 
containing the following keys:
- `type`: Type of the hyperparameter. Supported types are "int", "float", "loguniform" 
    and "categorical".

    For "int","float" and "loguniform" types, the dictionary should contain the following keys:

- `low`: Lower bound of the hyperparameter range.
- `high`: Upper bound of the hyperparameter range.

    For "categorical" type, the dictionary should contain the following key:

- `choices`: List of choices for categorical hyperparameters.    

The script will run the optimization for each target column separately and log the results 
to MLflow.
"""

import json
import os
import tempfile  # Import tempfile for temporary directory management
from functools import partial
from typing import Any, Dict, List

import mlflow
import optuna
import pandas as pd

from neuron.data_splitting.feature_based import FeatureBasedSplitter
from neuron.evaluation.evaluation_utils import get_target_metrics, plot_predictions_versus_actual
from neuron.models.target_models.registry import get_registered_model_class_from_name
from neuron.schemas.training_run_config import LoadCaseDataConfig
from neuron.training_run_default_configs.default_config import (
    get_default_load_case_training_run_config,
)
from neuron.utils import set_seed

mlflow.autolog(disable=True)


def objective(
    trial: optuna.trial.Trial,
    dlc: str,
    model_name: str,
    param_ranges: Dict[str, Dict[str, Any]],
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: List[str],
    target_col: str,
    data_dir: str,
    plots_dir: str,
) -> float:
    set_seed()
    model_params = {}

    for param_name, param in param_ranges.items():
        param_type = param["type"]
        if param_type == "int":
            model_params[param_name] = trial.suggest_int(param_name, param["low"], param["high"])
        elif param_type == "float":
            model_params[param_name] = trial.suggest_float(param_name, param["low"], param["high"])
        elif param_type == "loguniform":
            model_params[param_name] = trial.suggest_loguniform(
                param_name, param["low"], param["high"]
            )
        elif param_type == "categorical":
            model_params[param_name] = trial.suggest_categorical(param_name, param["choices"])
        else:
            raise ValueError(f"Unsupported parameter type '{param_type}' for parameter '{param}'.")

    default_config = get_default_load_case_training_run_config(
        dlc, LoadCaseDataConfig(training_data_file_uri="")
    )

    if default_config.load_case_model.params["feature_scaler_name"] is not None:
        scaler = default_config.load_case_model.params["feature_scaler_name"]
        model_params["feature_scaler_name"] = scaler

    model_class = get_registered_model_class_from_name(model_name)

    model = model_class(
        features=features,
        target_col=target_col,
        verbose=True,
        **model_params,
    )

    # Train the model
    model.fit(df_train)

    predictions = model.predict(df_test, return_std=True)
    pred = predictions.values_as_np
    pred_std = predictions.values_std_as_np

    actuals = df_test[target_col].to_numpy()
    actuals_std = df_test[target_col + "_std"].to_numpy()

    metrics = get_target_metrics(
        target_actuals=actuals,
        target_actuals_std=actuals_std,
        target_predictions=pred,
        target_predictions_std=pred_std,
    )

    # Store metrics in trial
    metrics_dict = {}
    for attr_name, attr_value in metrics.__dict__.items():
        json.dumps(attr_value)
        metrics_dict[attr_name] = attr_value

    trial.set_user_attr("metrics", metrics_dict)

    # Save the predictions to a CSV file
    df_test_with_preds = df_test.copy()
    df_test_with_preds[f"{target_col}_pred"] = pred
    output_file = os.path.join(data_dir, f"df_test_with_preds_{trial.number}.csv")
    df_test_with_preds.to_csv(output_file, index=False)

    cv_plot = plot_predictions_versus_actual(
        actuals=actuals,
        predictions=pred,
        metrics=metrics,
        max_load_evaluation_limit=default_config.max_load_evaluation_limit,
        name_prefix=f"cv_trial_{trial.number}",
    )

    cv_plot.savefig(os.path.join(plots_dir, f"cv_plot_{trial.number}.png"))

    return metrics.mae_norm


def run_trial(
    df_path: str,
    dlc: str,
    model_name: str,
    param_ranges: Dict[str, Dict[str, Any]],
    features: List[str],
    target_col: str,
    n_trials: int,
) -> None:
    df_td = pd.read_parquet(df_path)
    splitter = FeatureBasedSplitter(["twr_frq1", "twr_frq2", "twr_hh"])
    df_train, df_test = splitter.train_test_split(df_td)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Define the storage for Optuna study in temp directory
        optuna_db_path = os.path.join(tmp_dir, "optuna_study.db")
        storage = f"sqlite:///{optuna_db_path}"

        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        plots_dir = os.path.join(tmp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        objective_with_params = partial(
            objective,
            dlc=dlc,
            model_name=model_name,
            param_ranges=param_ranges,
            df_train=df_train,
            df_test=df_test,
            features=features,
            target_col=target_col,
            data_dir=data_dir,
            plots_dir=plots_dir,
        )

        # Create Optuna study and run the optimization
        study = optuna.create_study(direction="minimize", storage=storage)
        study.optimize(objective_with_params, n_trials=n_trials)

        # Collect results
        results = []
        for trial in study.trials:
            trial_metrics = trial.user_attrs.get("metrics", {})
            trial_params = trial.params
            trial_number = trial.number
            data = {"trial_number": trial_number, **trial_params, **trial_metrics}
            results.append(data)

        results_df = pd.DataFrame(results)
        csv_path = os.path.join(tmp_dir, "optuna_results.csv")
        results_df.to_csv(csv_path, index=False)

        df_train_path = os.path.join(tmp_dir, "df_train.parquet")
        df_train.to_parquet(df_train_path, index=False)

        df_test_path = os.path.join(tmp_dir, "df_test.parquet")
        df_test.to_parquet(df_test_path, index=False)

        run_name = f"opt_{model_name}_{target_col}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_artifact(csv_path)
            mlflow.log_artifact(df_train_path, artifact_path="data")
            mlflow.log_artifact(optuna_db_path)
            mlflow.log_artifacts(data_dir, artifact_path="data")
            mlflow.log_artifacts(plots_dir, artifact_path="plots")


if __name__ == "__main__":
    ################ Trial configuration ##################

    mlflow.set_experiment("/Shared/model_opt/model-opt")  # Databricks workspace path

    dlc = ""  # Ex: "dlc11"
    df_path = ""  # Path to the parquet file
    targets = []  # Ex: ["MyTwrBot_Twr_m400",'MyTwrBot_Twr_m800']
    model_name = ""  # Ex: "deep_gpr"
    features = []  # Ex: ["ws","ti","slope","yaw","rho","vexp","twr_frq1","twr_frq2","twr_hh"]
    n_trials = 20

    # Example of parameter ranges setup for deep_gpr
    param_ranges = {
        "n_inducing_points": {"type": "int", "low": 80, "high": 150},
        "num_hidden_dims": {"type": "int", "low": 1, "high": 10},
        "batch_size": {"type": "categorical", "choices": [256, 512, 1024]},
    }

    for target in targets:
        run_trial(
            df_path=df_path,
            dlc=dlc,
            model_name=model_name,
            param_ranges=param_ranges,
            features=features,
            target_col=target,
            n_trials=n_trials,
        )
