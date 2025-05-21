import pytest
from typer import Typer
from typer.testing import CliRunner

from neuron.schemas.domain import TurbineVariant
from neuron_prediction_service.model_store_service.release_cli import (
    validate_turbine_variants_build_configs_ids_are_unique,
)
from neuron_prediction_service.model_store_service.schemas import TurbineVariantBuildConfig

runner = CliRunner()


@pytest.fixture(scope="function")
def app(monkeypatch) -> Typer:  # noqa: ANN001
    monkeypatch.setenv("DATABRICKS_HOST", "some-host")
    from neuron_prediction_service.model_store_service.release_cli import app

    return app


def test_cli_error_config_folder_does_not_exist(app: Typer) -> None:
    # Arrange
    expected_error_message = (
        "Error: --turbine_variant-model-config-folder should be a path to a directory"
    )
    result = runner.invoke(
        app,
        [
            "--local-dir",
            "some-local-output-folder",
            "--turbine-model-config-folder",
            "some-output-path",
        ],
    )
    assert result.exit_code == 1
    assert result.stdout.strip() == expected_error_message


def test_cli_error_when_supplying_both_folder_and_path(tmp_path_factory, app) -> None:  # noqa: ANN001
    # Arrange
    test_folder = tmp_path_factory.mktemp("test_folder")
    test_config_file = test_folder / "config.yaml"
    test_config_file.write_text("some config")

    expected_error_message = (
        "Error: Exactly one of --turbine_variant-model-config-path or "
        "--turbine_variant-model-config-folder should be specified."
    )
    result = runner.invoke(
        app,
        [
            "--local-dir",
            "some-local-output-folder",
            "--turbine-model-config-folder",
            f"{test_folder}",
            "--turbine-model-config-path",
            f"{test_config_file}",
        ],
    )
    assert result.exit_code == 1
    assert result.stdout.strip() == expected_error_message


def test_cli_error_wrong_config_format(tmp_path_factory, app) -> None:  # noqa: ANN001
    # Arrange
    test_folder = tmp_path_factory.mktemp("test_folder")
    test_config_file = test_folder / "config.json"
    test_config_file.write_text('{"a": 1}')

    expected_error_message = (
        "Error: --turbine_variant-model-config-path should be a path to a yaml file."
    )
    result = runner.invoke(
        app,
        [
            "--local-dir",
            "some-local-output-folder",
            "--turbine-model-config-path",
            f"{test_config_file}",
        ],
    )
    assert result.exit_code == 1
    assert result.stdout.strip() == expected_error_message


def test_validate_turbine_variants_build_configs_ids_are_unique() -> None:
    """Test validation that turbine variant build configs are unique."""
    unique_turbine_build_configs = [
        TurbineVariantBuildConfig(
            turbine_variant=TurbineVariant(rotor_diameter=10, rated_power=50, mk_version="test"),
            version=1,
            mlflow_run_id="mlflow_run_ids",
        ),
        TurbineVariantBuildConfig(
            turbine_variant=TurbineVariant(rotor_diameter=10, rated_power=50, mk_version="test"),
            version=2,
            mlflow_run_id="mlflow_run_ids2",
        ),
    ]
    non_unique_turbine_build_configs = unique_turbine_build_configs + [
        TurbineVariantBuildConfig(
            turbine_variant=TurbineVariant(rotor_diameter=10, rated_power=50, mk_version="test"),
            version=1,
            mlflow_run_id="mlflow_run_ids",
        )
    ]

    validate_turbine_variants_build_configs_ids_are_unique(unique_turbine_build_configs)
    with pytest.raises(ValueError):
        validate_turbine_variants_build_configs_ids_are_unique(non_unique_turbine_build_configs)
