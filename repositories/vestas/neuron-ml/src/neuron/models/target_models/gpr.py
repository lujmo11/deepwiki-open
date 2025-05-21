import os
import pickle
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Self

import gpytorch
import pandas as pd
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import InducingPointKernel, LinearKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from neuron.models.scaler_registry import DataScaler, get_registered_scaler_from_name
from neuron.models.target_models.base import Model
from neuron.schemas.domain import TargetValues
from neuron.utils import check_columns_in_dataframe

logger = getLogger(__name__)


class GPRPytorch(Model, gpytorch.models.ExactGP):
    name = "gpr"

    def __init__(
        self,
        features: List[str],
        target_col: str,
        n_inducing_points: int = 300,
        nu: float = 2.5,
        training_iter: int = 500,
        es_patience: int = 80,
        learning_rate: float = 0.1,
        lr_patience: int = 20,
        cool_down: int = 20,
        validation_size: float = 0.1,
        eval_batch_size: int = 1024,
        feature_scaler_name: DataScaler = DataScaler.ROBUST_SCALER,
        target_scaler_name: DataScaler = DataScaler.ROBUST_SCALER,
        add_linear_kernel: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            torch.zeros(1, len(features)), torch.zeros(1), gpytorch.likelihoods.GaussianLikelihood()
        )
        self._features = features
        self._target_col = target_col
        self.mean_module = None
        self.base_covar_module = None
        self.covar_module = None
        self.likelihood = None
        self.n_inducing_points = n_inducing_points
        self.nu = nu
        self.training_iter = training_iter
        self.es_patience = es_patience
        self.learning_rate = learning_rate
        self.lr_patience = lr_patience
        self.cool_down = cool_down
        self.validation_size = validation_size
        self.eval_batch_size = eval_batch_size
        self.add_linear_kernel = add_linear_kernel
        self.feature_scaler_name = feature_scaler_name
        self.target_scaler_name = target_scaler_name
        self.verbose = verbose

        self.target_scaler = get_registered_scaler_from_name(name=target_scaler_name)
        self.feature_scaler = get_registered_scaler_from_name(name=feature_scaler_name)

        torch_device = os.environ.get("TORCH_DEVICE")
        if torch_device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(torch_device)

    @property
    def target_col(self) -> str:
        return self._target_col

    @property
    def features(self) -> List[str]:
        return self._features

    # Pytorch specific function that's being called internally
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def fit(self, df: pd.DataFrame) -> Self:
        check_columns_in_dataframe(df, self.features + [self.target_col])
        X = df[self.features].to_numpy()
        y = df[self.target_col].to_numpy()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_size, random_state=42
        )
        y_train = self.target_scaler.fit_transform(y_train.reshape(-1, 1))
        X_train = self.feature_scaler.fit_transform(X_train)
        X_val = self.feature_scaler.transform(X_val)

        y_train_t = torch.tensor(y_train, dtype=torch.float32)[:, -1].to(self.device)
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        # Setup of model
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        super().__init__(X_train_t, y_train_t, likelihood)

        self.likelihood = likelihood

        self.mean_module = ConstantMean()
        if self.add_linear_kernel:
            self.base_covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=X_train_t.size(1))
                + LinearKernel(num_dimensions=X_train_t.size(1))
            )
        else:
            self.base_covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=X_train_t.size(1))
            )
        self.covar_module = InducingPointKernel(
            self.base_covar_module,
            inducing_points=X_train_t[: self.n_inducing_points, :].clone(),
            likelihood=self.likelihood,
        )

        self.to(self.device)

        # Training
        self.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.lr_patience, factor=0.5, cooldown=self.cool_down
        )

        best_val_mae = float(10**10)
        best_model_state_dict = self.state_dict()

        for i in range(self.training_iter):
            self.train()
            self.likelihood.train()
            optimizer.zero_grad()

            output = self(X_train_t)
            loss = -mll(output, y_train_t)
            loss.backward()
            optimizer.step()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                self.eval()
                self.likelihood.eval()
                test_preds = self.likelihood(self(X_val_t))

                # Move data back to cpu
                mean = test_preds.mean.cpu()
                pred = self.target_scaler.inverse_transform(mean.reshape(-1, 1))
                r2 = r2_score(y_val, pred)
                mae = round(mean_absolute_error(y_val, pred), 4)

                if mae < best_val_mae:
                    best_val_mae = mae
                    best_model_state_dict = self.state_dict()
                    n = 0
                else:
                    n += 1
                    if i > int(0.2 * self.training_iter) and n > self.es_patience:
                        logger.info(
                            "Validation mae has not improved from the %d epoch. Stopping early."
                            % (i - self.es_patience)
                        )
                        break

            scheduler.step(mae)

            if self.verbose:
                print(
                    "Iter %d - Loss: %.3f - Val mae: %.3f - Val r2: %.3f - LR: %.3f"
                    % (i, loss.item(), mae, r2, optimizer.param_groups[0]["lr"])
                )
        logger.info(
            "Final iter: %d - Loss: %.3f - Val mae: %.3f - Val r2: %.3f" % (i, loss.item(), mae, r2)
        )

        self.load_state_dict(best_model_state_dict)
        self.cast_model_to_64bit()

        return self

    def predict(
        self, df: pd.DataFrame, return_std: bool = False, return_grads: bool = False
    ) -> TargetValues:
        check_columns_in_dataframe(df, self.features)
        X = df[self.features].to_numpy()
        X = self.feature_scaler.transform(X)
        X = torch.tensor(X, requires_grad=return_grads).to(self.device)
        self.eval()
        self.likelihood.eval()

        means = []
        plos = []
        phis = []
        grads = []

        with gpytorch.settings.fast_pred_var():
            data_loader = DataLoader(X, batch_size=self.eval_batch_size)
            for x_batch in data_loader:
                if return_grads:
                    preds = self.likelihood(self(x_batch))

                    grads.append(
                        torch.autograd.grad(
                            outputs=preds.mean.sum(),
                            inputs=x_batch,
                            retain_graph=False,
                        )[0]
                    )

                else:
                    with torch.no_grad():
                        preds = self.likelihood(self(x_batch))

                means.append(preds.mean)

                if return_std:
                    plo, phi = preds.confidence_region()
                    plos.append(plo)
                    phis.append(phi)

        if return_grads:
            gradients = torch.cat(grads, dim=0).detach().cpu().numpy()
            # scale gradients to the orig feature space
            gradients *= self.target_scaler.scale_ / self.feature_scaler.scale_
            gradients_dict = {
                feature: gradients[:, idx].tolist() for idx, feature in enumerate(self.features)
            }

        if return_std:
            pred_low = self.target_scaler.inverse_transform(
                torch.cat(plos, dim=-1).detach().cpu().numpy().reshape(-1, 1)
            )
            pred_high = self.target_scaler.inverse_transform(
                torch.cat(phis, dim=-1).detach().cpu().numpy().reshape(-1, 1)
            )
            pred_std = (pred_high - pred_low) * 0.25
            pred_std_list = list(pred_std[:, 0])

        return TargetValues(
            target_name=self.target_col,
            value_list=list(
                self.target_scaler.inverse_transform(
                    torch.cat(means, dim=-1).detach().cpu().numpy().reshape(-1, 1)
                )[:, 0]
            ),
            value_list_std=pred_std_list if return_std else None,
            gradients_dict=gradients_dict if return_grads else None,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "target_col": self.target_col,
            "n_inducing_points": self.n_inducing_points,
            "nu": self.nu,
            "training_iter": self.training_iter,
            "es_patience": self.es_patience,
            "learning_rate": self.learning_rate,
            "validation_size": self.validation_size,
            "add_linear_kernel": self.add_linear_kernel,
            "feature_scaler_name": self.feature_scaler_name,
            "target_scaler_name": self.target_scaler_name,
        }

    def construct_model_data(self) -> Dict[str, Any]:
        model_data = {
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "X_train": self.train_inputs[0].cpu().numpy(),
            "y_train": self.train_targets.cpu().numpy(),
            "target_scaler": self.target_scaler,
            "feature_scaler": self.feature_scaler,
        }
        model_data.update(self.get_params())
        return model_data

    def _save_model(self, folder_path: str) -> None:
        folder_path = Path(folder_path)
        model_artifact_path = folder_path / "model.pkl"
        if not folder_path.exists():
            folder_path.mkdir(parents=True)

        model_data = self.construct_model_data()

        with open(model_artifact_path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model_from_dict(cls, model_dict: Dict[str, Any]) -> Self:
        n_features = model_dict["X_train"].shape[1]
        X_train_t = torch.tensor(model_dict["X_train"])
        y_train_scaled_t = torch.tensor(model_dict["y_train"])
        # Create an instance of the Gpr_matern_gpytorch model
        model = cls(
            features=model_dict["features"],
            target_col=model_dict["target_col"],
            n_inducing_points=model_dict["n_inducing_points"],
            add_linear_kernel=model_dict["add_linear_kernel"],
            feature_scaler_name=model_dict["feature_scaler_name"],
            target_scaler_name=model_dict["feature_scaler_name"],
        )

        # Setup of model components
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(model.device)
        model.likelihood = likelihood
        model.mean_module = ConstantMean().to(model.device)
        if model.add_linear_kernel:
            model.base_covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=n_features)
                + LinearKernel(num_dimensions=n_features)
            ).to(model.device)
        else:
            model.base_covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=n_features)).to(
                model.device
            )
        model.covar_module = InducingPointKernel(
            model.base_covar_module,
            inducing_points=X_train_t[: model.n_inducing_points, :].clone(),
            likelihood=model.likelihood,
        ).to(model.device)

        # Initialize the model with the saved state_dict
        model.load_state_dict(model_dict["state_dict"])

        # Set train_inputs and train_targets
        model.train_inputs = (X_train_t,)
        model.train_targets = y_train_scaled_t
        model.target_scaler = model_dict["target_scaler"]
        model.feature_scaler = model_dict["feature_scaler"]
        return model

    @classmethod
    def load_model(cls, folder_path: str) -> Self:
        folder_path = Path(folder_path)
        model_artifact_path = folder_path / "model.pkl"
        with open(model_artifact_path, "rb") as f:
            model_dict = pickle.load(f)
        return cls.load_model_from_dict(model_dict)

    def cast_model_to_64bit(self) -> None:
        # Cast all model parameters to 64 bit precision
        self.to(torch.float64)
        self.train_inputs = (self.train_inputs[0].to(torch.float64),)
        self.train_targets = self.train_targets.to(torch.float64)
        self.likelihood = self.likelihood.to(torch.float64)
        self.mean_module = self.mean_module.to(torch.float64)
        self.base_covar_module = self.base_covar_module.to(torch.float64)
        self.covar_module = self.covar_module.to(torch.float64)
