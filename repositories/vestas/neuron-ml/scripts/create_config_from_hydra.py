"""CLI for converting a hydra config to a single yaml file representing a TrainingRunConfig.

This is a convenience script for converting a hydra config and overwrites to a single yaml file.
It allows the project to keep using Hydra for local development configuration management,
while using a single yaml file for the actual entrypoint for the ML training pipeline.
"""
import argparse
from pathlib import Path
from typing import Any, Dict, Union

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf


def _replace_undefined_recursively(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively replace all values in dict that are '???' with None"""
    for k, v in d.items():
        if isinstance(v, dict):
            _replace_undefined_recursively(v)
        elif v == "???":
            d[k] = None
    return d


def parse_hydra_config(
    hydra_config: Union[OmegaConf, DictConfig],
) -> dict:
    """Parse a hydra config to a dictionary."""
    omega_conf_dict = OmegaConf.to_container(hydra_config, resolve=True)
    return _replace_undefined_recursively(omega_conf_dict)


def main() -> None:
    """Converts a hydra config and overwrites to a single yaml file
    representing a TrainingRunConfig
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hydra-config-folder",
        action="store",
        type=str,
        help="Path to the hydra config folder",
        required=False,
    )
    parser.add_argument(
        "--train-config-output-path",
        action="store",
        type=str,
        help="Output path to yaml file with training run config.",
        required=True,
    )

    args = parser.parse_args()
    hydra_config_folder = args.hydra_config_folder
    train_config_output_path = args.train_config_output_path

    absolute_hydra_config_folder = str(Path(hydra_config_folder).absolute())
    with hydra.initialize_config_dir(config_dir=absolute_hydra_config_folder, version_base="1.2"):
        hydra_config = hydra.compose(config_name="config")
        hydra_config_dict = parse_hydra_config(hydra_config)

        with open(train_config_output_path, "w") as f:
            yaml.dump(hydra_config_dict, f)


if __name__ == "__main__":
    main()
