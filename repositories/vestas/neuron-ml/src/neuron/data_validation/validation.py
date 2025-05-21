import logging
from typing import List, Union

import pandas as pd
from pandera.errors import SchemaError

from neuron.data_splitting.registry import get_data_splitter
from neuron.data_validation.exceptions import DataValidationError
from neuron.data_validation.table_schemas import create_schema_from_columns
from neuron.io.storage_reader import StorageReader
from neuron.io.training_data_repository import TrainingDataRepository
from neuron.safe_domain.safe_domain_validator import SafeDomainValidator
from neuron.schemas.domain import (
    AggregationMethod,
    Feature,
    FeatureType,
    FeatureValueType,
    LoadCase,
    PostprocessorName,
    Target,
)
from neuron.schemas.training_run_config import LoadCaseTrainingRunConfig

logger = logging.getLogger(__name__)

MAX_UNIQUE_VALUES_FOR_DISCRETE_FEATURE = 20


# Required targets for each postprocessor
POSTPROCESSOR_REQUIRED_TARGET_MAP = {
    PostprocessorName.DIRECTIONAL_TWR: [
        Target.MxTwrTop_Twr_m400,
        Target.MyTwrTop_Twr_m400,
        Target.MxTwrBot_Twr_m400,
        Target.MyTwrBot_Twr_m400,
    ],
    PostprocessorName.NO_POSTPROCESSOR: [],
}


def validate_data_schema(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate data for the training pipeline"""

    schema = create_schema_from_columns(columns)
    schema.validate(df)


def check_required_raw_features_for_load_case(df: pd.DataFrame, load_case: LoadCase) -> None:
    """Check that the required features for a load case are present in the dataframe,
    and that they have the correct data types."""
    raw_features = [f for f in load_case.feature_list.features if f.feature_type == FeatureType.RAW]
    for feature in raw_features:
        feature_type = feature.feature_value_type  # continuous or discrete
        feature_name = feature.name  # should be a column in the dataframe
        if feature_name not in df.columns:
            raise DataValidationError(
                f"Feature {feature_name} is required for load case {load_case.name} but not present"
                " in the dataset."
            )
        # Check that data for a discrete feature have a limited number of unique values
        if feature_type == FeatureValueType.DISCRETE:
            unique_values = df[feature_name].nunique()
            if unique_values > MAX_UNIQUE_VALUES_FOR_DISCRETE_FEATURE:
                raise DataValidationError(
                    f"Feature {feature_name} is specified to be a discrete feature for load case"
                    f" {load_case.name}. The data for a discrete feature type should have"
                    f" less than {MAX_UNIQUE_VALUES_FOR_DISCRETE_FEATURE} unique values, but found "
                    f"{unique_values} unique values."
                )


def check_required_input_features_for_load_case(df: pd.DataFrame, load_case: LoadCase) -> None:
    input_features = [f for f in load_case.feature_list.features if f.is_model_input]
    for feature in input_features:
        feature_name = feature.name
        if feature_name not in df.columns:
            raise DataValidationError(
                f"Feature {feature_name} is required input for load case {load_case.name} but not "
                "present in the dataset."
            )


def check_required_safe_domain_features_for_load_case(
    df: pd.DataFrame, load_case: LoadCase
) -> None:
    safe_domain_features = [
        f for f in load_case.feature_list.features if f.safe_domain_method is not None
    ]
    for feature in safe_domain_features:
        feature_name = feature.name
        if feature_name not in df.columns:
            raise DataValidationError(
                f"Feature {feature_name} is required for safe domain validation for load case "
                f"{load_case.name} but not present in the dataset."
            )


def check_required_aggregated_metrics_data(
    df: pd.DataFrame,
    aggregation_method: AggregationMethod,
) -> None:
    """Check that required columns for an aggregation method are present in a DataFrame."""
    if aggregation_method.weightby not in df.columns:
        raise DataValidationError(
            f"Test data frame is missing the required weightby aggregation "
            f"method column {aggregation_method.weightby}."
            f"The dataframe contains these columns: {list(df.columns)}"
        )

    if aggregation_method.groupby not in df.columns:
        raise DataValidationError(
            f"Test data frame is missing the required groupby aggregation "
            f"method column {aggregation_method.groupby}."
            f"The dataframe contains these columns: {list(df.columns)}"
        )


def check_safe_domain_hull_definition(df: pd.DataFrame, feature_list: List[Feature]) -> None:
    safe_domain = SafeDomainValidator(feature_list)
    safe_domain.fit(df)


def check_required_targets_for_postprocessor(
    df: pd.DataFrame,
    postprocessor_name: PostprocessorName,
    dataset_name: str = "",
) -> None:
    required_postprocessor_features = POSTPROCESSOR_REQUIRED_TARGET_MAP[postprocessor_name]
    if not set(required_postprocessor_features).issubset([f for f in df.columns]):
        raise DataValidationError(
            f"The postprocessor '{postprocessor_name}' requires the columns "
            f"{POSTPROCESSOR_REQUIRED_TARGET_MAP[postprocessor_name]} "
            f"but only the columns {[f for f in df.columns]} are "
            f"present in the {dataset_name} dataset."
        )


def validate_data_for_single_load_case(
    load_case: LoadCase,
    data_repo: TrainingDataRepository,
    calculate_aggregated_metrics: Union[AggregationMethod, None],
) -> None:
    """Validate that the data for a load case is present and it has the correct columns."""
    train_df = data_repo.get_load_case_train_df()
    test_df = data_repo.get_load_case_test_df()

    check_required_targets_for_postprocessor(
        df=train_df, dataset_name="train", postprocessor_name=load_case.postprocessor
    )
    check_required_targets_for_postprocessor(
        df=test_df, dataset_name="test", postprocessor_name=load_case.postprocessor
    )
    check_required_raw_features_for_load_case(df=train_df, load_case=load_case)
    check_required_raw_features_for_load_case(df=test_df, load_case=load_case)

    if calculate_aggregated_metrics:
        agg_df = data_repo.get_load_case_agg_df()

        check_required_aggregated_metrics_data(
            df=agg_df, aggregation_method=calculate_aggregated_metrics
        )
        check_required_targets_for_postprocessor(
            df=agg_df, dataset_name="agg", postprocessor_name=load_case.postprocessor
        )
        check_required_raw_features_for_load_case(df=agg_df, load_case=load_case)

    check_safe_domain_hull_definition(df=train_df, feature_list=load_case.feature_list.features)

    raw_feature_names = [
        f.name for f in load_case.feature_list.features if f.feature_type == FeatureType.RAW
    ]
    targets = [t.name for t in load_case.target_list.targets]
    targets_std = [target + "_std" for target in targets]
    required_columns = raw_feature_names + targets + targets_std
    try:
        validate_data_schema(df=train_df, columns=required_columns)
        validate_data_schema(df=test_df, columns=required_columns)
        if calculate_aggregated_metrics:
            validate_data_schema(df=agg_df, columns=required_columns)
    except SchemaError as e:
        raise DataValidationError(f"Data schema validation error: {e}") from e


def validate_data_all_load_cases(
    load_case_configs: List[LoadCaseTrainingRunConfig], storage: StorageReader
) -> None:
    for load_case_training_run_config in load_case_configs:
        load_case = LoadCase(
            name=load_case_training_run_config.name,
            feature_list=load_case_training_run_config.feature_list,
            target_list=load_case_training_run_config.target_list,
            postprocessor=load_case_training_run_config.postprocessor,
        )
        data_splitter = get_data_splitter(
            name=load_case_training_run_config.data_splitting.name,
            params=load_case_training_run_config.data_splitting.params,
        )
        data_repo = TrainingDataRepository(
            training_file_uri=load_case_training_run_config.data.training_data_file_uri,
            test_file_uri=load_case_training_run_config.data.test_data_file_uri,
            agg_file_uri=load_case_training_run_config.data.agg_data_file_uri,
            data_splitter=data_splitter,
            storage=storage,
        )
        try:
            validate_data_for_single_load_case(
                load_case=load_case,
                data_repo=data_repo,
                calculate_aggregated_metrics=load_case_training_run_config.calculate_aggregated_metrics,
            )
        except DataValidationError as e:
            logger.error(f"Error validating data for load case {load_case.name}.")
            raise DataValidationError(
                f"Error validating data for load case {load_case.name}: {e}"
            ) from e


def validate_test_data_for_agg_load_cases(
    load_case_configs: List[LoadCaseTrainingRunConfig], storage: StorageReader
) -> None:
    unique_values_set = None

    for load_case_training_run_config in load_case_configs:
        if load_case_training_run_config.calculate_aggregated_metrics:
            data_splitter = get_data_splitter(
                name=load_case_training_run_config.data_splitting.name,
                params=load_case_training_run_config.data_splitting.params,
            )
            data_repo = TrainingDataRepository(
                training_file_uri=load_case_training_run_config.data.training_data_file_uri,
                test_file_uri=load_case_training_run_config.data.test_data_file_uri,
                agg_file_uri=load_case_training_run_config.data.agg_data_file_uri,
                data_splitter=data_splitter,
                storage=storage,
            )

            agg_df = data_repo.get_load_case_agg_df()

            # get unique values of groupby columns
            unique_values_current = set(
                agg_df[load_case_training_run_config.calculate_aggregated_metrics.groupby]
            )

            if unique_values_set is None:
                unique_values_set = unique_values_current
            elif unique_values_set != unique_values_current:
                raise DataValidationError(
                    "All groupby columns should have the same unique values "
                    "for aggregated load cases"
                )
