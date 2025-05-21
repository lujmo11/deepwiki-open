"""CLI for generating trained models and fixed predictions for model stability testing.

Rerun if any changes are made to a model class that should result in new predictions.
"""
import os
import pickle
import sys
from typing import Annotated

import pandas as pd
import typer

from neuron.models.target_models.registry import TARGET_MODEL_REGISTER
from neuron.utils import set_seed

# Append root directory to sys.path to be able to import fixture
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tests.conftest import model_class_sample_configs  # noqa: E402

app = typer.Typer()
set_seed(42)
TRAINING_DATA_STABILITY_TEST_FILE_PATH = "tests/data/training_pipeline/fixed_training_data.parquet"
FIXED_PREDICTIONS_FOLDER = "tests/data/training_pipeline/fixed_predictions"
PARAMETER_PREDICTIONS_MAP_FILE_NAME = "parameter_predictions_map.pkl"

training_df_for_model_stability = pd.read_parquet(TRAINING_DATA_STABILITY_TEST_FILE_PATH)


def model_init_params_to_str(init_params: dict) -> str:
    """Convert model initialization parameter dict to a string representation for file names."""
    return "__".join([f"{k}_{v}" for k, v in init_params.items()])


@app.command()
def generate_predictions(
    model_class_name: Annotated[
        str,
        typer.Option(
            help=(
                "Name of specific model class to generate predictions for. "
                "If not provided, all models will be used."
            )
        ),
    ] = None,
) -> None:
    """Generate fixed predictions for model testing."""
    if model_class_name:
        try:
            model_classes = [TARGET_MODEL_REGISTER[model_class_name]]
        except KeyError:
            typer.echo(
                f"Model name {model_class_name} not in target model registry. "
                f"Available models are: {list(TARGET_MODEL_REGISTER.keys())}"
            )
            raise typer.Exit(1) from None
    else:
        model_classes = TARGET_MODEL_REGISTER.values()

    for model_class in model_classes:
        param_prediction_map = {}
        for init_params in model_class_sample_configs[model_class]:
            model = model_class(
                features=training_df_for_model_stability.columns.drop(
                    ["target", "target_std"]
                ).to_list(),
                target_col="target",
                **init_params,
            )
            model.fit(training_df_for_model_stability)
            param_prediction_map[str(init_params)] = model.predict(
                training_df_for_model_stability, return_std=True
            )

            typer.echo(f"Save trained {model_class.name} model with parameters: {init_params}.")
            if not os.path.exists(f"{FIXED_PREDICTIONS_FOLDER}/{model_class.name}"):
                os.makedirs(f"{FIXED_PREDICTIONS_FOLDER}/{model_class.name}")

            model.save_model(
                folder_path=f"{FIXED_PREDICTIONS_FOLDER}/{model_class.name}/{model_init_params_to_str(init_params)}_model"
            )

        typer.echo(
            f"Save mapping between initialization parameters and predictions "
            f"for model {model_class.name} to file ."
        )
        with open(
            f"{FIXED_PREDICTIONS_FOLDER}/{model_class.name}/{PARAMETER_PREDICTIONS_MAP_FILE_NAME}",
            "wb",
        ) as f:
            pickle.dump(param_prediction_map, f)


if __name__ == "__main__":
    app()
