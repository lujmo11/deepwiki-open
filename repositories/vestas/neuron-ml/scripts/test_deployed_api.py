"""Test predict and training APIs.
This can be used as part of the deployment pipeline to check that the API is working as expected.
"""

from time import sleep
from typing import Literal

import requests
import urllib3

# We have self-signed certificates, so we cannot make secure https calls. Ignore that for now
urllib3.disable_warnings()

PREDICTION_SAMPLE_DATA = {
    "data": {
        "vexp": [1, 2, 3],
        "ws": [10, 10, 10],
        "yaw": [1, 2, 3],
        "rho": [1, 2, 3],
        "slope": [1, 2, 3],
        "twr_frq1": [1, 2, 3],
        "twr_frq2": [1, 2, 3],
        "ti": [1, 2, 3],
        "twr_hh": [100, 100, 100],
        "epb": [1, 2, 3],
        "vtl": [1, 2, 3],
        "eco": [1, 2, 3],
        "power_rating": [1, 2, 3],
        "wnd_grad": [0.1, 0.1, 0.1],
        "eog_trigger_time": [1, 2, 3],
        "yaw_offset": [1, 2, 3],
        "horz_grad": [1, 2, 3],
        "vert_grad": [1, 2, 3],
        "coherence_category": [1, 2, 3],
    }
}


TRAIN_API_TEST_CONFIG = {
    "load_case_training_runs": [
        {
            "name": "dlc11",
            "data": {"training_data_file_uri": "test/data_reduced_dlc11.parquet"},
        }
    ],
    "turbine": {
        "turbine_variant": {"rotor_diameter": "150", "rated_power": "4000", "mk_version": "mk3f"}
    },
    "evaluation": {
        "alpha_significance_level": 0,
        "generate_coverage_plots": True,
        "fat_model_acceptance_criteria": [
            {
                "metric": "e_mean_norm",
                "value": 0.01,
                "condition": "le",
            }
        ],
        "ext_model_acceptance_criteria": [
            {
                "metric": "e_mean_norm",
                "value": 0.01,
                "condition": "le",
            }
        ],
    },
}


def match_build_id(
    base_url: str, desired_build_id: str, allowed_retries: int, headers: dict
) -> str:
    """Get the build_id of the deployed API. If it does not match the desired build id,
    we will retry a few times before failing"""
    try:
        url = base_url + "/version"
        # Need to set user agent to avoid getting rejected by the server
        response = requests.get(url, verify=False, headers=headers)
        response.raise_for_status()
        build_id = response.json()["build_id"]
        if build_id == desired_build_id:
            return build_id
        else:
            raise Exception(f"Build id mismatch. Desired: {desired_build_id}, actual: {build_id}")
    except Exception as err:
        print(err)
        if allowed_retries > 0:
            print("Failed to get or match build id. Waiting 60 seconds")
            sleep(60)
            return match_build_id(base_url, desired_build_id, allowed_retries - 1, headers=headers)
        else:
            raise Exception("Failed to get or match build id") from err


def get_all_load_case_model_ids(base_url: str, headers: dict) -> list[str]:
    """Call the /load_case_model_ids endpoint to get all the load case model ids.

    Example:
        ["turbine_variant_id=162_5600_mk0a---load_case=dlc11---model_version=1.0"]
    """
    url = base_url + "/load_case_model_ids"
    response = requests.get(url, verify=False, headers=headers)
    response.raise_for_status()
    model_ids = response.json()
    return model_ids


def test_load_case_model(base_url: str, load_case_model_id: str, headers: dict) -> None:
    """Make a call to the predict endpoint with some non-sensical data as the payload.
    If it does not raise an error, we assume the model works.
    """
    url = base_url + f"/predict/{load_case_model_id}"
    response = requests.post(url, verify=False, headers=headers, json=PREDICTION_SAMPLE_DATA)
    try:
        response.raise_for_status()
    except Exception as err:
        print(f"FAILED for load case model id: {load_case_model_id}")
        print(f"Response: {response.text}")
        raise err


def test_train_api_triggering(base_url: str, headers: dict) -> dict:
    """Make a call to the train endpoint with a sample training configuration that points to
    sample data. This will trigger a training run.
    The response which contains the training_input_id and run_id is returned.
    """
    url = base_url + "/train_job"
    response = requests.post(url, verify=False, headers=headers, json=TRAIN_API_TEST_CONFIG)
    response.raise_for_status()
    return response.json()


def main(
    base_url: str,
    desired_build_id: str,
    allowed_retries: int,
    api: Literal["train", "predict"],
    api_keys: str,
) -> None:
    headers = {"User-Agent": "Avoid getting rejected by the server"}
    match_build_id(base_url, desired_build_id, allowed_retries, headers=headers)
    print("Build id matched successfully")
    if api_keys:
        single_key = api_keys.split(",")[0]
        headers["x-api-key"] = single_key
    if api == "predict":
        load_case_ids = get_all_load_case_model_ids(base_url, headers=headers)
        if len(load_case_ids) == 0:
            raise Exception("Did not fetch any load_case_model_ids")
        print("All load case model ids fetched successfully")
        for load_case_id in load_case_ids:
            print(f"Testing load case id: {load_case_id}")
            test_load_case_model(base_url, load_case_id, headers=headers)
            print(f"Load case id {load_case_id} tested successfully")
            # Wait for 1 second to avoid overwhelming the server
            sleep(1)

        print("All models tested successfully!")
    elif api == "train":
        train_api_response = test_train_api_triggering(base_url, headers=headers)
        if set(train_api_response.keys()) != {"training_input_id", "job_run_id"}:
            raise Exception(
                "Train API did not return the expected keys: training_input_id, job_run_id"
            )
        print("Train API successfully triggered Databricks job")
        # TODO: Check that the Databricks job succeeds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--build_id", type=str, required=True)
    parser.add_argument("--allowed_retries", type=int, default=3)
    parser.add_argument("--api", type=str, choices=["train", "predict"])
    parser.add_argument(
        "--api_keys", type=str, required=False, help="Comma separated list of API keys"
    )
    args = parser.parse_args()
    base_url = str(args.base_url).rstrip("/")
    main(
        base_url=base_url,
        desired_build_id=args.build_id,
        allowed_retries=args.allowed_retries,
        api=args.api,
        api_keys=args.api_keys,
    )
