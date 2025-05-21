"""Helper script for the transition from csv to parquet files.

Example usage:
    python scripts/convert_csv_to_parquet.py tests/data/training_pipeline/*csv

This will convert all csv files in the tests/data/training_pipeline directory to parquet files, 
and remove the original csv files.
"""
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO)


def convert_to_parquet(input_path: str) -> None:
    """Convert a csv file to parquet file.

    This will:
        - remove any columns starting with "Unnamed"
        - Save the parquet file with the same name as the input file,
            but with the extension .parquet
        - Remove the original csv file.
    """
    if not input_path.endswith(".csv"):
        raise ValueError(f"Input file must be a csv file. Received {input_path}.")
    output_path = input_path.replace(".csv", ".parquet")
    data = pd.read_csv(input_path)
    columns = data.columns
    # There seems to be some index columns leftover in the files. Remove those
    columns = [c for c in columns if not c.startswith("Unnamed")]
    data = data[columns]
    data.to_parquet(output_path, index=False)
    logging.info(f"Converted {input_path} to {output_path}.")
    logging.info(f"Removing {input_path}.")
    os.remove(input_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_paths", type=str, help="Paths to the csv files.", nargs="+")
    args = parser.parse_args()
    for input_path in args.input_paths:
        convert_to_parquet(input_path=input_path)
