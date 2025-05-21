"""
 Speed/capacity test script for the Prediction API

Step 1. Download the model store: "just create_local_model_store_artifact"
Step 2. Run the API: "just run_pred_api_locally"
Step 3. Run this script
"""

import json
import os
import time

import numpy as np
import requests

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = "http://" + HOST + ":" + str(PORT)
MODEL_ID = "turbine_variant_id=162_5600_mk0a---version=1---load_case=dlc11"

username = "neuron"
password = "neuron"

os.environ["model_store_folder_path"] = "tests/data/api/dummy_model_store"

pred_url = f"{BASE_URL}/predict/{MODEL_ID}"


def generate_data(num_rows: int) -> dict:
    data = {
        "data": {
            # "vexp": list(np.zeros(num_rows)),
            "ws": [10] * num_rows,
            "yaw": list(range(0, num_rows)),
            "rho": list(range(0, num_rows)),
            "slope": list(range(0, num_rows)),
            "twr_frq1": list(range(0, num_rows)),
            "twr_frq2": list(range(0, num_rows)),
            "ti": list(range(0, num_rows)),
            "twr_hh": [100.0] * num_rows,
            "wnd_grad": [0.1] * num_rows,
            "epb": list(range(0, num_rows)),
            "vtl": list(range(0, num_rows)),
            "eco": list(range(0, num_rows)),
            "power_rating": list(range(0, num_rows)),
        },
        "targets": ["MyHub_m400"],
        "grad_features": ["ws"],
    }
    return data


data = generate_data(10)

headers = {"Content-type": "application/json"}
credentials = (username, password)

data_json = json.dumps(data)

n_requests = 1
time_np = np.zeros(n_requests)
for i in range(n_requests):
    t_start = time.time()
    response = requests.post(pred_url, data=data_json, headers=headers, auth=credentials)
    t_end = time.time()
    print(f"Status code: {response.status_code}")

    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response text: {response.text}")
        raise ValueError("Prediction failed")

    time_np[i] = t_end - t_start

print(f"Fastest prediction time: {np.min(time_np)} s")
print(f"Mean prediction time: {np.mean(time_np)} s")
print(f"Slowest prediction time: {np.max(time_np)} s")
