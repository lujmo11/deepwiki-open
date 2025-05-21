"""Interface for experiment trackers."""
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, List

import pandas as pd

from neuron.evaluation.load_case_evaluator import (
    BinnedLoadCaseMetrics,
    StandardLoadCaseMetrics,
)
from neuron.evaluation.turbine_evaluator import BinnedTurbineRunMetrics
from neuron.models.load_case_model_pipeline import LoadCaseModelPipeline
from neuron.schemas.domain import (
    LoadCasePredictions,
    TurbineVariant,
)
from neuron.schemas.training_run_config import TrainingRunConfig


class ExperimentTracker(ABC):
    """Interface for experiment trackers."""

    @abstractmethod
    def log_experiment_params(self, params: Dict[str, Any]) -> None:
        """Log experiment level parameters to experiment tracker."""
        pass

    @abstractmethod
    def log_experiment_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log experiment level metrics to experiment tracker."""
        pass

    @abstractmethod
    def log_turbine_training_run_config(self, training_run_config: TrainingRunConfig) -> None:
        """Log the turbine training run config."""
        pass

    @abstractmethod
    def log_turbine_variant(self, turbine_variant: TurbineVariant) -> None:
        """Log turbine variant to experiment tracker."""
        pass

    @abstractmethod
    def log_load_case_metrics(
        self,
        load_case_name: str,
        metrics: StandardLoadCaseMetrics,
        artifact_path: str,
    ) -> None:
        """Log load case metrics to experiment tracker."""
        pass

    @abstractmethod
    def log_binned_load_case_metrics(
        self, load_case_name: str, metrics: BinnedLoadCaseMetrics
    ) -> None:
        """Log binned load case metrics to experiment tracker."""
        pass

    @abstractmethod
    def log_load_case_model_params(
        self, load_case_name: str, load_case_model_params: Dict[str, Any]
    ) -> None:
        """Log load case parameters to experiment tracker."""
        pass

    @abstractmethod
    def log_load_case_plots(
        self, load_case_name: str, plots: Dict[str, BytesIO], artifact_path: str
    ) -> None:
        """Log load case plots to experiment tracker.

        Parameters
        ----------
        load_case_name : str
            Name of the load case.
        plots : Dict[str, BytesIO]
            Dictionary of plots to log. The key is the name of the plot and the value is a BytesIO
            object containing the plot.
        artifact_path: str
            Name of the folder under the load case folder name to store the plots.
        """
        pass

    @abstractmethod
    def log_load_case_model(self, load_case_model: LoadCaseModelPipeline) -> None:
        """Log load case model to experiment tracker."""
        pass

    @abstractmethod
    def log_turbine_variant_load_case_models(
        self, load_case_model_pipelines: List[LoadCaseModelPipeline]
    ) -> None:
        """Log turbine_variant models to experiment tracker."""
        pass

    @abstractmethod
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
            Dictionary of plots to log. The key is the name of the plot and the value is a BytesIO
            object containing the plot.
        """
        pass

    @abstractmethod
    def log_turbine_binned_plots(self, turbine_plots: Dict[str, BytesIO]) -> None:
        """Log turbine_variant level run metrics plots.

        Parameters
        ----------
        turbine_plots : Dict[str, BytesIO]
            Dictionary containing the turbine_variant run metrics plots. The key corresponds
            to the target name and the value is a BytesIOobject containing the plot.
        """
        pass

    @abstractmethod
    def log_detailed_rel_design_plots(self, plots: Dict[str, Dict[str, BytesIO]]) -> None:
        """Log detailed relative design plots to MLFlow run."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def log_turbine_extreme_load_plots(self, turbine_plots: Dict[str, BytesIO]) -> None:
        """Log turbine_variant level extreme load overview plots.

        Parameters
        ----------
        turbine_plots : Dict[str, BytesIO]
            Dictionary containing the turbine_variant run metrics plots.
            The key is the filename of the plot and the value is a BytesIO
            object containing the plot.
        """
        pass

    @abstractmethod
    def log_ext_design_load_bin_metrics(self, bin_metrics: BinnedTurbineRunMetrics) -> None:
        """Log turbine variant extreme design load bin metrics to MLFlow run.

        The metrics are logged as one json file in the "metrics" folder.
        """
        pass

    @abstractmethod
    def log_train_and_test_data(
        self, load_case_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        pass

    @abstractmethod
    def log_test_and_prediction_data(
        self,
        load_case_name: str,
        df: pd.DataFrame,
        load_case_predictions: LoadCasePredictions,
        df_prefix: str,
    ) -> None:
        pass

    @abstractmethod
    def log_turbine_metrics_overview(self, df_list: List[pd.DataFrame]) -> None:
        """Log turbine variant level aggregated and extreme load metrics

        Parameters
        ----------
        df_list : List[pd.DataFrame]"""
        pass

    @abstractmethod
    def log_turbine_overview_html(self, html: str) -> None:
        """Log turbine level model validation overview in html.

        Parameters
        ----------
        html : str
            HTML string of turbine overview data.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def __enter__(self) -> None:
        """Enter experiment tracking context."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Exit experiment tracking context."""
        pass

    @staticmethod
    @abstractmethod
    def log_runtime_metadata() -> None:
        pass
