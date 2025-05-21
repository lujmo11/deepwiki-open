import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Self, Union

import dobby.mlflow_logging
import mlflow
import pandas as pd

from neuron.evaluation.load_case_evaluator import (
    BinnedLoadCaseMetrics,
    StandardLoadCaseMetrics,
)
from neuron.evaluation.turbine_evaluator import BinnedTurbineRunMetrics
from neuron.experiment_tracking.base import ExperimentTracker
from neuron.models.load_case_model_pipeline import LoadCaseModelPipeline
from neuron.schemas.domain import (
    LoadCasePredictions,
    TurbineVariant,
)
from neuron.schemas.training_run_config import TrainingRunConfig
from neuron.utils import zip_folder


class MLFlowTracker(ExperimentTracker):
    """MLFlow experiment tracker."""

    def __init__(self, experiment_name: str, tracking_uri: Union[str, None] = None):
        """Initialize MLFlow experiment tracker.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        tracking_uri : Union[str, None]
            The tracking uri to use for mlflow. If None, the default mlflow
            tracking uri is used. This is set from the environment variable.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def __enter__(self) -> Self:
        """Start mlflow run."""
        mlflow.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """End mlflow run."""
        mlflow.end_run()

    @staticmethod
    def log_runtime_metadata() -> None:
        """Log things that are important for reproducibility on the ML platform
        using the dobby package.
        """
        dobby.mlflow_logging.log_runtime_metadata()

    def log_experiment_params(self, params: Dict[str, Any]) -> None:
        """Log experiment level parameters to MLFlow run.

        The parameters are logged as params in MLFlow
        """
        mlflow.log_params(params)

    def log_experiment_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log experiment level metrics to MLFlow.

        The parameters are logged as metrics in MLFlow
        """
        mlflow.log_metrics(metrics)

    def log_turbine_training_run_config(self, training_run_config: TrainingRunConfig) -> None:
        """Log the turbine training run config as a json file to experiment"""
        with TemporaryDirectory() as tmp_dirname:
            training_run_config_temp_file = Path(tmp_dirname) / "turbine_training_run_config.json"
            with open(training_run_config_temp_file, "w") as f:
                json.dump(training_run_config.dict(), f, indent=4)
            mlflow.log_artifact(local_path=str(training_run_config_temp_file))

    def log_turbine_variant(self, turbine_variant: TurbineVariant) -> None:
        """Log turbine variant to MLFlow run.

        The turbine_variant_id is logged as tag and the turbine_variant
        is logged as json file in MLFlow.
        """
        mlflow.set_tag("turbine_variant_id", turbine_variant.id)
        with TemporaryDirectory() as tmp_dirname:
            turbine_variant_temp_file = Path(tmp_dirname) / "turbine_variant.json"
            with open(turbine_variant_temp_file, "w") as f:
                json.dump(turbine_variant.dict(), f, indent=4)

            mlflow.log_artifact(local_path=str(turbine_variant_temp_file))

    def log_load_case_metrics(
        self,
        load_case_name: str,
        metrics: StandardLoadCaseMetrics,
        artifact_path: str,
    ) -> None:
        """Log load case metrics to MLFlow run.

        The metrics are logged as json file for each target in MLFlow
        """
        targets = metrics.keys()

        target_metrics_dict = {}
        for target in targets:
            target_metrics_dict[target] = metrics[target].dict()

        target_metrics_df = pd.DataFrame(target_metrics_dict).T
        target_metrics_df = target_metrics_df.reset_index().rename(columns={"index": "target"})

        with TemporaryDirectory() as tmp_dirname:
            metrics_temp_file = Path(tmp_dirname) / f"{load_case_name}_target_metrics.csv"
            target_metrics_df.to_csv(metrics_temp_file)

            mlflow.log_artifact(local_path=str(metrics_temp_file), artifact_path=artifact_path)

    def log_binned_load_case_metrics(
        self, load_case_name: str, metrics: BinnedLoadCaseMetrics
    ) -> None:
        """Log binned load case metrics to MLFlow run.

        The binned metrics are logged as json file for each target in MLFlow
        """
        targets = metrics.keys()

        for target in targets:
            with TemporaryDirectory() as tmp_dirname:
                binned_metrics = metrics[target]
                for feature_key, bin_metrics in binned_metrics.items():
                    binned_metrics_temp_file = (
                        Path(tmp_dirname) / f"{target}_binned_metrics_{feature_key.name}.csv"
                    )
                    metrics_bin_list = []
                    for feature_bin, target_metrics in bin_metrics.items():
                        if target_metrics is None:
                            metrics_bin_dict = {**feature_bin.as_dict(), **{"metrics": "None"}}
                        else:
                            metrics_bin_dict = {**feature_bin.as_dict(), **target_metrics.dict()}
                        metrics_bin_list += [metrics_bin_dict]

                    metrics_df = pd.DataFrame(data=metrics_bin_list)
                    metrics_df.to_csv(binned_metrics_temp_file)

                    mlflow.log_artifact(
                        local_path=str(binned_metrics_temp_file),
                        artifact_path=f"binned_metrics/{load_case_name}",
                    )

    def log_load_case_model_params(
        self, load_case_name: str, load_case_model_params: Dict[str, Dict[str, Any]]
    ) -> None:
        """Log load case parameters to MLFlow run.

        The parameters are logged as json file for each target in MLFlow

        Parameters
        ----------
        load_case_name : str
            Name of the load case.
        load_case_model_params : Dict[str, Dict[str, Any]]
            Dictionary of parameters to log. The key is the name of
            the target name and the value is a dictionary containing
            the parameter values.
        """
        with TemporaryDirectory() as tmp_dirname:
            for target_name, target_params in load_case_model_params.items():
                params_temp_file = Path(tmp_dirname) / f"{target_name}_params.json"
                with open(params_temp_file, "w") as f:
                    json.dump(target_params, f)
                mlflow.log_artifact(
                    local_path=str(params_temp_file), artifact_path=f"params/{load_case_name}"
                )

    def log_load_case_plots(
        self, load_case_name: str, plots: Dict[str, BytesIO], artifact_path: str
    ) -> None:
        """Log load case plots to MLFlow run.

        The plots are saved as png files for each target in MLFlow.

        Parameters
        ----------
        load_case_name : str
            Name of the load case.
        plots : Dict[str, BytesIO]
            Dictionary of plots to log.
            The key is the filename of the plot and the value is a BytesIO
            object containing the plot.
        artifact_path: str
            Name of the folder under the load case folder name to store the plots.
        """
        self._log_plots(plots, f"plots/load_cases/{load_case_name}/{artifact_path}")

    def log_load_case_model(self, load_case_model: LoadCaseModelPipeline) -> None:
        """Log single load case model to MLFlow run.

        The model is saved as an artifact in MLFlow.
        """
        with TemporaryDirectory() as tmp_dirname:
            temp_model_path = str(Path(tmp_dirname) / f"{load_case_model.load_case.name}_model")
            load_case_model.save_model(temp_model_path)
            mlflow.log_artifacts(
                local_dir=temp_model_path,
                artifact_path=f"load_case_models/{load_case_model.load_case.name}_model",
            )

    def log_turbine_variant_load_case_models(
        self, load_case_model_pipelines: List[LoadCaseModelPipeline]
    ) -> None:
        """Log all turbine_variant load case models to mlflow as a zipped folder."""

        with TemporaryDirectory() as tmp_dirname:
            temp_model_path = str(Path(tmp_dirname) / "load_case_models")
            for load_case_model_pipeline in load_case_model_pipelines:
                load_case_model_pipeline.save_model(
                    str(Path(temp_model_path) / load_case_model_pipeline.load_case.name)
                )

            zip_file_path = Path(tmp_dirname) / "load_case_models.zip"
            zip_folder(folder_path=Path(temp_model_path), zip_artifact_path=zip_file_path)
            mlflow.log_artifact(local_path=str(zip_file_path))

    def log_load_case_binned_metrics_plot(
        self, load_case_name: str, plots: Dict[str, BytesIO]
    ) -> None:
        """Log load case binned plots to MLFlow run.

        The plots are saved as png files for each load case in MLFlow.

        Parameters
        ----------
        load_case_name : str
            Name of the load case.
        plots : Dict[str, BytesIO]
            Dictionary of plots to log.
            The key is the filename of the plot and the value is a BytesIO
            object containing the plot.
        """
        self._log_plots(plots, f"plots/load_cases/{load_case_name}/binned_metrics")

    def log_detailed_rel_design_plots(self, plots: Dict[str, Dict[str, BytesIO]]) -> None:
        """Log detailed relative design plots to MLFlow run."""
        for load_case in plots.keys():
            self._log_plots(plots[load_case], f"plots/load_cases/{load_case}/detailed_rel_design")

    def log_turbine_binned_plots(self, turbine_plots: Dict[str, BytesIO]) -> None:
        """Log turbine_variant level run metrics plots.

        Parameters
        ----------
        turbine_plots : Dict[str, BytesIO]
            Dictionary containing the turbine_variant run metrics plots.
            The key is the filename of the plot and the value is a BytesIO
            object containing the plot.
        """
        self._log_plots(turbine_plots, "plots/grouped_metrics_per_target")

    def log_prediction_speed_test_results(
        self, speed_test_results: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Log turbine_variant load case models prediction speed test results.

        Parameters
        ----------
        speed_test_results : Dict[str, Dict[str, float]]
            Dictionary containing the turbine_variant load case models
            prediction speed test results.
            The key is the load case name and the value is a dict containing the results
        """
        results_list = [
            {"load_case_name": load_case_name, **results}
            for load_case_name, results in speed_test_results.items()
        ]

        df_lc_speed_test = pd.DataFrame(results_list)

        with TemporaryDirectory() as tmp_dirname:
            speed_test_temp_file = Path(tmp_dirname) / "load_case_pred_speed_test.csv"
            df_lc_speed_test.to_csv(speed_test_temp_file, index=False)
            mlflow.log_artifact(
                local_path=str(speed_test_temp_file), artifact_path="full_training_run_metrics"
            )

    def log_turbine_extreme_load_plots(self, turbine_plots: Dict[str, BytesIO]) -> None:
        """Log turbine_variant level extreme load overview plots.

        Parameters
        ----------
        turbine_plots : Dict[str, BytesIO]
            Dictionary containing the turbine_variant run metrics plots.
            The key is the filename of the plot and the value is a BytesIO
            object containing the plot.
        """
        self._log_plots(turbine_plots, "plots/extreme_load_overview")

    def log_ext_design_load_bin_metrics(self, binned_metrics: BinnedTurbineRunMetrics) -> None:
        """Log turbine variant extreme design load bin metrics to MLFlow run.

        The metrics are logged as one json file in the "metrics" folder.
        """
        metrics_dict = {}

        for target, target_data in binned_metrics.items():
            metrics_dict[target] = {}
            for dlc, dlc_data in target_data.items():
                metrics_dict[target][dlc] = {}
                for load_bin, metrics in dlc_data.items():
                    if metrics is not None:
                        key = "design_load_bin"
                        combined_metrics = {**load_bin.dict(), **metrics.dict()}
                        metrics_dict[target][dlc][key] = combined_metrics

        with TemporaryDirectory() as tmp_dirname:
            metrics_temp_file = Path(tmp_dirname) / "ext_design_load_bin_metrics.json"
            with open(metrics_temp_file, "w") as f:
                json.dump(metrics_dict, f, indent=4)

            mlflow.log_artifact(local_path=str(metrics_temp_file), artifact_path="metrics")

    def log_turbine_metrics_overview(self, df_list: List[pd.DataFrame]) -> None:
        """Log turbine variant level aggregated and extreme load metrics

        Parameters
        ----------
        df_list : List[pd.DataFrame]"""

        df_all_agg = df_list[0]
        df_dlc_agg = df_list[1]
        df_dlc_ext = df_list[2]

        with TemporaryDirectory() as tmp_dirname:
            temp_data_path_all_agg = str(Path(tmp_dirname) / "aggregated_fatigue_metrics.csv")
            df_all_agg.to_csv(temp_data_path_all_agg)
            mlflow.log_artifact(
                local_path=str(temp_data_path_all_agg), artifact_path="full_training_run_metrics"
            )
            temp_data_path_dlc_agg = str(
                Path(tmp_dirname) / "aggregated_load_case_fatigue_metrics.csv"
            )
            df_dlc_agg.to_csv(temp_data_path_dlc_agg)
            mlflow.log_artifact(
                local_path=str(temp_data_path_dlc_agg), artifact_path="full_training_run_metrics"
            )
            temp_data_path_ext = str(Path(tmp_dirname) / "load_case_extreme_metrics.csv")
            df_dlc_ext.to_csv(temp_data_path_ext)
            mlflow.log_artifact(
                local_path=str(temp_data_path_ext), artifact_path="full_training_run_metrics"
            )

    def log_turbine_overview_html(self, html: str) -> None:
        """Log turbine level model validation overview in html.

        Parameters
        ----------
        html : str
            HTML string of turbine overview data.
        """
        with TemporaryDirectory() as tmp_dirname:
            temp_html_path = str(Path(tmp_dirname) / "turbine_overview.html")
            with open(temp_html_path, "w") as f:
                f.write(html)
            mlflow.log_artifact(
                local_path=str(temp_html_path), artifact_path="full_training_run_metrics"
            )

    @staticmethod
    def _log_plots(plots: Dict[str, BytesIO], folder_path: str) -> None:
        """Log turbine_variant level run metrics plots.

        Parameters
        ----------
        plots : Dict[str, BytesIO]
            Dictionary containing the turbine_variant run metrics plots.
            The key is the filename of the plot and the value is a BytesIO
            object containing the plot.
        """

        with TemporaryDirectory() as tmp_dirname:
            for plot_file_name, plot_bytes_io in plots.items():
                temp_file_plot = Path(tmp_dirname) / f"{plot_file_name}"
                with open(temp_file_plot, "wb") as f:
                    f.write(plot_bytes_io.getvalue())
                mlflow.log_artifact(
                    local_path=str(temp_file_plot),
                    artifact_path=folder_path,
                )

    def log_train_and_test_data(
        self, load_case_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        """Log load case train and test data (csv and parquet) including feature and target
        values.

        Parameters
        ----------
        train_df: pd.DataFrame
            Dataframe including the data used for model training.
        test_df: pd.DataFrame
            Dataframe including the data used for model testing.

        """

        with TemporaryDirectory() as tmp_dirname:
            # Logging training data
            temp_data_path_csv = str(Path(tmp_dirname) / "train_data.csv")
            train_df.to_csv(temp_data_path_csv)
            mlflow.log_artifact(
                local_path=str(temp_data_path_csv), artifact_path=f"data/{load_case_name}"
            )
            temp_data_path_parquet = str(Path(tmp_dirname) / "train_data.parquet")
            train_df.to_parquet(temp_data_path_parquet)
            mlflow.log_artifact(
                local_path=str(temp_data_path_parquet), artifact_path=f"data/{load_case_name}"
            )

            # Logging test data
            temp_data_path_csv = str(Path(tmp_dirname) / "test_data.csv")
            test_df.to_csv(temp_data_path_csv)
            mlflow.log_artifact(
                local_path=str(temp_data_path_csv), artifact_path=f"data/{load_case_name}"
            )
            temp_data_path_parquet = str(Path(tmp_dirname) / "test_data.parquet")
            test_df.to_parquet(temp_data_path_parquet)
            mlflow.log_artifact(
                local_path=str(temp_data_path_parquet), artifact_path=f"data/{load_case_name}"
            )

    def log_test_and_prediction_data(
        self,
        load_case_name: str,
        df: pd.DataFrame,
        load_case_predictions: LoadCasePredictions,
        df_prefix: str = "test",
    ) -> None:
        """Log load case test data (csv and parquet) including features values, targets and
        target predictions.

        Parameters
        ----------
        load_case_name : str
            The name of the load case. Defines the folder to store the data.
        load_case_actuals: LoadCaseActuals
            The values of the targets used on the test dataset.
        load_case_predictions: LoadCasePredictions
            The predicted values for the test dataset.
        """

        test_data_dict = {}
        adjusted_df = df.reset_index(drop=True)
        for target in load_case_predictions.keys():
            test_data_dict[target + "_pred"] = load_case_predictions[target].value_list
            if load_case_predictions[target].value_list_std is not None:
                test_data_dict[target + "_pred_std"] = load_case_predictions[target].value_list_std

        new_columns_df = pd.DataFrame(test_data_dict)
        df_comb = pd.concat([adjusted_df, new_columns_df], axis=1)

        with TemporaryDirectory() as tmp_dirname:
            temp_data_path_csv = str(
                Path(tmp_dirname) / (df_prefix + "_data_including_predictions.csv")
            )
            df_comb.to_csv(temp_data_path_csv)
            mlflow.log_artifact(
                local_path=str(temp_data_path_csv), artifact_path=f"data/{load_case_name}"
            )

            temp_data_path_parquet = str(
                Path(tmp_dirname) / (df_prefix + "_data_including_predictions.parquet")
            )
            df_comb.to_parquet(temp_data_path_parquet)
            mlflow.log_artifact(
                local_path=str(temp_data_path_parquet), artifact_path=f"data/{load_case_name}"
            )

    def log_train_test_data_plots(
        self, load_case_name: str, data_plot_io: BytesIO, postfix: str
    ) -> None:
        """Log load case train and test data plots including the interpolation/extrapolation domain

        Parameters
        ----------
        load_case_name : str
            The name of the load case. Defines the folder to store the data.
        data_plot_io: BytesIO
            Object containing the data plot.
        """
        with TemporaryDirectory() as tmp_dirname:
            plot_file_name = load_case_name + "_" + postfix + "_data.png"
            temp_file_plot = Path(tmp_dirname) / f"{plot_file_name}"
            with open(temp_file_plot, "wb") as f:
                f.write(data_plot_io.getvalue())
                mlflow.log_artifact(
                    local_path=str(temp_file_plot),
                    artifact_path=f"data/{load_case_name}",
                )
