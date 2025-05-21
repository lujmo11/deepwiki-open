import json
import logging
import os
import zipfile
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import Dict, List

import mlflow
from pyspark.sql import SparkSession

from neuron.io.training_data_repository import TrainingDataRepository
from neuron.load_case_model_training_pipeline import (
    LoadCaseModelPipeline,
    setup_and_run_load_case_model_training,
)
from neuron.schemas.domain import TurbineVariant
from neuron.schemas.training_run_config import (
    LoadCaseTrainingRunConfig,
    TrainingRunConfig,
)


class TurbineTrainingPipeline(ABC):
    """Abstract class for establishing trained load case models, e.g. by local training,
    distributed training on dbx, fetching allready trained models from mlflow, etc."""

    def __init__(
        self,
        load_case_configs: List[LoadCaseTrainingRunConfig],
        turbine_data_repo: Dict[str, TrainingDataRepository],
        turbine_variant: TurbineVariant,
    ):
        self.load_case_configs = load_case_configs
        self.turbine_data_repo = turbine_data_repo
        self.turbine_variant = turbine_variant

    @abstractmethod
    def train_load_cases(self) -> Dict[str, LoadCaseModelPipeline]:
        pass


class LocalTurbineTrainingPipeline(TurbineTrainingPipeline):
    """Pipeline for training load case models"""

    def __init__(
        self,
        load_case_configs: List[LoadCaseTrainingRunConfig],
        turbine_data_repo: Dict[str, TrainingDataRepository],
        turbine_variant: TurbineVariant,
    ):
        super().__init__(load_case_configs, turbine_data_repo, turbine_variant)

    def train_load_cases(self) -> Dict[str, LoadCaseModelPipeline]:
        results = {}
        for lc_train_config in self.load_case_configs:
            train_df = self.turbine_data_repo[lc_train_config.name].get_load_case_train_df()
            logging.info(f"Training load case: {lc_train_config.name}")
            lc_pipeline = setup_and_run_load_case_model_training(
                load_case_training_run_config=lc_train_config,
                train_df=train_df,
                turbine_variant=self.turbine_variant,
            )
            results[lc_train_config.name] = lc_pipeline
        return results


class DbxTurbineTrainingPipeline(TurbineTrainingPipeline):
    """Pipeline for training load case models on Databricks"""

    def __init__(
        self,
        load_case_configs: List[LoadCaseTrainingRunConfig],
        turbine_data_repo: Dict[str, TrainingDataRepository],
        turbine_variant: TurbineVariant,
    ):
        super().__init__(load_case_configs, turbine_data_repo, turbine_variant)

    def train_load_cases(self) -> Dict[str, LoadCaseModelPipeline]:
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
        # Broadcast the turbine_data_repo to all workers.
        turbine_data_repo_bc = sc.broadcast(self.turbine_data_repo)

        def train_single_loadcase(lc_train_config: LoadCaseTrainingRunConfig):
            data_repo = turbine_data_repo_bc.value[lc_train_config.name]
            train_df = data_repo.get_load_case_train_df()
            output = setup_and_run_load_case_model_training(
                load_case_training_run_config=lc_train_config,
                train_df=train_df,
                turbine_variant=self.turbine_variant,
            )
            return (lc_train_config.name, output)

        # Create an Spark RDD from the list of load-case configurations
        # num_slices equal to # of loadcases ensures that each loadcase training
        # is processed in a separate worker.
        lc_configs_rdd = sc.parallelize(
            self.load_case_configs,
            numSlices=len(self.load_case_configs),
        )
        logging.info("Distributing load case training runs")

        # Run parallel training of load-case configurations.
        # If any of the workers fail, the collect() will raise an exception
        results = lc_configs_rdd.map(train_single_loadcase).collect()
        return dict(results)


class PreTrainedTurbinePipeline:
    """Pipeline for retrieving pre-trained load-case models from MLflow."""

    def __init__(
        self,
        load_case_configs: List[LoadCaseTrainingRunConfig],
        turbine_data_repo: Dict[str, TrainingDataRepository],
        turbine_variant: TurbineVariant,
        mlflow_run_id: str,
    ):
        self.load_case_configs = load_case_configs
        self.turbine_data_repo = turbine_data_repo
        self.turbine_variant = turbine_variant
        self.mlflow_run_id = mlflow_run_id

    def train_load_cases(self) -> Dict[str, LoadCaseModelPipeline]:
        """
        Retrieve pre-trained pipelines from the zipped MLflow artifacts
        `load_case_models.zip` and load the pipeline for each load case.
        """
        original_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        try:
            config_artifact_path = "turbine_training_run_config.json"
            config_path = mlflow.artifacts.download_artifacts(
                run_id=self.mlflow_run_id,
                artifact_path=config_artifact_path,
            )
            with open(config_path, "r") as f:
                remote_training_run_config = TrainingRunConfig(**json.load(f))

            self._check_configuration_compatibility(remote_training_run_config)

            zip_artifact_path = "load_case_models.zip"
            local_zip_path = mlflow.artifacts.download_artifacts(
                run_id=self.mlflow_run_id,
                artifact_path=zip_artifact_path,
            )

            results = {}
            with TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmp_dir)

                for lc_train_config in self.load_case_configs:
                    logging.info(
                        f"Retrieving pre-trained load case: {lc_train_config.name} "
                        f"from mlflow run {self.mlflow_run_id}"
                    )
                    subfolder_path = os.path.join(tmp_dir, "load_case_models", lc_train_config.name)

                    pretrained_pipeline = LoadCaseModelPipeline.load_model(subfolder_path)
                    results[lc_train_config.name] = pretrained_pipeline

            return results

        finally:
            # Restore the original MLflow URI if present
            if original_uri:
                os.environ["MLFLOW_TRACKING_URI"] = original_uri
            else:
                del os.environ["MLFLOW_TRACKING_URI"]

    def _check_configuration_compatibility(self, remote_training_run_config: TrainingRunConfig):
        """
        Compare the TrainingRunConfig loaded from MLflow with your local
        self.load_case_configs and other pipeline attributes to ensure
        they are compatible. If not, raise an error or handle it appropriately.
        """
        if remote_training_run_config.turbine.turbine_variant != self.turbine_variant:
            raise ValueError(
                f"Remote turbine variant {remote_training_run_config.turbine.turbine_variant} "
                f"does not match locally configured turbine variant {self.turbine_variant}"
            )

        remote_load_cases = {
            lc.name: lc for lc in remote_training_run_config.load_case_training_runs
        }
        for local_lc in self.load_case_configs:
            if local_lc.name not in remote_load_cases:
                raise ValueError(
                    f"No corresponding load case '{local_lc.name}' found in "
                    f"the remote TrainingRunConfig from MLflow."
                )
