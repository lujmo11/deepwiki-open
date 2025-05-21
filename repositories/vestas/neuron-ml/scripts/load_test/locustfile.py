import numpy as np
from locust import HttpUser, task


def generate_data(num_rows: int) -> dict:
    random_input = np.randon.choice([1, 2, 3], num_rows)
    data = {
        "data": {
            "vexp": random_input,
            "ws": random_input,
            "yaw": random_input,
            "rho": random_input,
            "slope": random_input,
            "twr_frq1": random_input,
            "twr_frq2": random_input,
            "ti": random_input,
            "twr_hh": random_input,
            "epb": random_input,
            "vtl": random_input,
            "eco": random_input,
            "power_rating": random_input,
            "wnd_grad": random_input,
        }
    }
    return data


data = generate_data(800)

headers = {"Content-type": "application/json"}


class PredictionEndpoint(HttpUser):
    @task
    def prediction_request(self) -> None:
        self.client.post(
            "/predict/turbine_variant_id=162_5600_mk0a&load_case=dlc11&model_version=1.0", json=data
        )
