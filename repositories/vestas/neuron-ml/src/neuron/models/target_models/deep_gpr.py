import os
import pickle
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Self, Tuple, Union

import gpytorch
import pandas as pd
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import LinearKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import DeepApproximateMLL, PredictiveLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from neuron.models.scaler_registry import DataScaler, get_registered_scaler_from_name
from neuron.models.target_models.base import Model
from neuron.schemas.domain import TargetValues
from neuron.utils import check_columns_in_dataframe, set_seed

logger = getLogger(__name__)


class HeteroscedasticGaussianLikelihood(GaussianLikelihood):
    def __init__(self, fixed_noise: torch.Tensor | None = None, **kwargs) -> None:
        """
        Extension of the GaussianLikelihood class that enables the use of heteroscedastic
        noise (fixed_noise). The fixed_noise is added to the noise covariance of the model
        during training.

        This approach is useful when target uncertainty is known, i.e. enables the model to put
        less weight on uncertain data points.

        fixed_noise: an optional tensor containing noise variances. The optionality enables
        inference to be performed without fixed noise (log_marginal only called during training
        through the mll)
        """
        super().__init__(**kwargs)
        self.fixed_noise = fixed_noise

    def log_marginal(
        self,
        target: torch.Tensor,
        function_dist: gpytorch.distributions.MultivariateNormal,
        **kwargs,
    ) -> torch.Tensor:
        """
        Overrides log_marginal to include fixed_noise. Note: PredictiveLogLikelihood
        will forward fixed_noise to this method.
        """
        noise = kwargs.get("fixed_noise", self.fixed_noise)
        if noise is None:
            raise RuntimeError(
                "HeteroscedasticGaussianLikelihood: fixed_noise must be provided to log_marginal."
            )
        effective_variance = function_dist.variance + self.noise_covar.noise + noise
        normal_dist = torch.distributions.Normal(function_dist.mean, effective_variance.sqrt())
        lp = normal_dist.log_prob(target)
        return lp.sum(-1)


class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(
        self,
        input_dims: int,
        output_dims: Union[int, None],
        num_inducing: int,
        nu: float,
        mean_type: str = "constant",
        add_linear_kernel: bool = False,
    ):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape,
        )
        # Define the variational strategy for sparse variational inference
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        # Initialize the DeepGPLayer with the variational strategy and input/output dimensions
        super(DeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == "constant":
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        # Define the covariance module (kernel function of the GP)
        if add_linear_kernel:
            self.covar_module = ScaleKernel(
                MaternKernel(nu=nu, batch_shape=batch_shape, ard_num_dims=input_dims)
                + LinearKernel(num_dimensions=input_dims, batch_shape=batch_shape),
                batch_shape=batch_shape,
                ard_num_dims=None,
            )
        else:
            self.covar_module = ScaleKernel(
                MaternKernel(nu=nu, batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape,
                ard_num_dims=None,
            )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x: torch.Tensor, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation
        based skip connections easily. For example, hidden_layer2(hidden_layer1_outputs, inputs)
        will pass the concatenation of the first hidden layer's outputs and the input
        data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()
            processed_inputs = [
                inp.unsqueeze(0).expand(
                    gpytorch.settings.num_likelihood_samples.value(), *inp.shape
                )
                for inp in other_inputs
            ]
            x = torch.cat([x] + processed_inputs, dim=-1)
        return super().__call__(x, are_samples=bool(len(other_inputs)))


class DeepGPRPytorch(Model, DeepGP):
    """
    Deep Gaussian Process model based on gpytorch.

    The model consists of a hidden layer with num_hidden_dims GP's and an output layer
    with a single GP.

    Model description can be found in DMS: 0201-4355

    Parameters:
    - features: List of feature column names.
    - target_col: Name of the target variable column.
    - num_hidden_dims: Number of dimensions (GP's) in the hidden layer.
    - n_inducing_points: Number of inducing points used for variational inference.
    - nu: Smoothness parameter for the Matern kernel.
    - training_iter: Maximum number of training iterations (epochs).
    - es_patience: Number of epochs with no improvement before early stopping.
    - lr_patience: Number of epochs with no improvement before reducing the learning rate.
    - learning_rate: Initial learning rate for the optimizer.
    - lr_factor: Factor by which the learning rate is reduced on plateau.
    - cooldown: Number of epochs to wait after reducing learning rate before
      learning rate can be reduced again.
    - batch_size: Batch size used during training.
    - eval_batch_size: Batch size used during prediction.
    - n_samples_train: Number of samples drawn from the variational distribution
      during training to average over.
    - n_samples_pred: Number of samples drawn during prediction.
    - add_linear_kernel: Add a linear kernel to the Matern kernel.
    - use_noise_from_data: Use uncertainty from training data (requires target_col+'_std').
    - feature_scaler_name: Name of the scaler to use for the features.
    - target_scaler_name: Name of the scaler to use for the target variable.
    - verbose: Prints detailed training progress information.
    """

    name = "deep_gpr"

    def __init__(
        self,
        features: List[str],
        target_col: str,
        num_hidden_dims: int = 5,
        n_inducing_points: int = 100,
        nu: float = 2.5,
        training_iter: int = 1000,
        es_patience: int = 40,
        lr_patience: int = 10,
        learning_rate: float = 0.02,
        lr_factor: float = 0.5,
        cooldown: int = 20,
        batch_size: int = 512,
        eval_batch_size: int = 1024,
        n_samples_train: int = 10,
        n_samples_pred: int = 10,
        max_restarts: int = 3,
        restart_r2_threshold: float = 0.7,
        add_linear_kernel: bool = False,
        use_noise_from_data: bool = True,
        feature_scaler_name: DataScaler = DataScaler.ROBUST_SCALER,
        target_scaler_name: DataScaler = DataScaler.ROBUST_SCALER,
        verbose: bool = False,
    ):
        self.seed = set_seed()
        self._features = features
        self._target_col = target_col
        self.num_hidden_dims = num_hidden_dims
        self.mean_module = None
        self.base_covar_module = None
        self.covar_module = None
        self.likelihood = None
        self.n_inducing_points = n_inducing_points
        self.nu = nu
        self.training_iter = training_iter
        self.es_patience = es_patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.cooldown = cooldown
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.n_samples_train = n_samples_train
        self.n_samples_pred = n_samples_pred
        self.max_restarts = max_restarts
        self.restart_r2_threshold = restart_r2_threshold
        self.add_linear_kernel = add_linear_kernel
        self.use_noise_from_data = use_noise_from_data
        self.feature_scaler_name = feature_scaler_name
        self.target_scaler_name = target_scaler_name
        self.verbose = verbose

        self.feature_scaler = get_registered_scaler_from_name(name=feature_scaler_name)
        self.target_scaler = get_registered_scaler_from_name(name=target_scaler_name)

        # Determine the device to run on (GPU or CPU)
        torch_device = os.environ.get("TORCH_DEVICE")
        if torch_device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(torch_device)

        super().__init__()

        self.init_model()

    def init_model(self) -> None:
        if self.add_linear_kernel:
            mean_type_hidden_layer = "constant"
        else:
            mean_type_hidden_layer = "linear"

        # Defining the layers of the DeepGP

        # First hidden layer
        hidden_layer = DeepGPHiddenLayer(
            input_dims=len(self._features),
            output_dims=self.num_hidden_dims,
            num_inducing=self.n_inducing_points,
            nu=self.nu,
            mean_type=mean_type_hidden_layer,
            add_linear_kernel=self.add_linear_kernel,
        )
        # Last layer (output layer)
        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            num_inducing=self.n_inducing_points,
            nu=self.nu,
            mean_type="constant",
            add_linear_kernel=self.add_linear_kernel,
        )
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        self.to(self.device)

    @property
    def target_col(self) -> str:
        return self._target_col

    @property
    def features(self) -> List[str]:
        return self._features

    def forward(self, inputs: torch.Tensor) -> MultitaskMultivariateNormal:
        """
        Forward pass of inputs through the hidden layer and last layer.

        When forward is called, the number of forward passes to average over is determined by
        the number of samples defined by gpytorch.settings.num_likelihood_samples.

        Process of a forward pass:

        Step 1. Condition each GP in the hidden layer on the input datapoint to
        get the posterior distributions at that datapoint.

        Step 2. Sample each GP posterior dist to get the outputs of the hidden layer.

        Step 3. Take these outputs and use as input to the next layer (final layer).

        Step 4. Condition the GP in the final layer on the outputs from the hidden layer to get the
        posterior dist.

        Step 5. Sample the posterior dist of the GP in the final layer to get the final
        output of the model.

        If 10 samples are defined by gpytorch.settings.num_likelihood_samples, the model output
        will be the average of the 10 final outputs.

        """
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def _prepare_training_data(self, df: pd.DataFrame):
        """Prepare and scale training data."""
        check_columns_in_dataframe(df, self.features + [self.target_col])
        if self.use_noise_from_data:
            check_columns_in_dataframe(df, [self.target_col + "_std"])

        X = df[self.features].to_numpy()
        y = df[self.target_col].to_numpy()

        X_train = self.feature_scaler.fit_transform(X)
        y_train = self.target_scaler.fit_transform(y.reshape(-1, 1))

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device).contiguous()
        y_train_t = torch.Tensor(y_train)[:, -1].to(self.device).contiguous()

        if self.use_noise_from_data:
            noise = df[self.target_col + "_std"].to_numpy() ** 2
            if hasattr(self.target_scaler, "scale_"):
                noise = noise / (self.target_scaler.scale_**2)
            noise_train_t = torch.tensor(noise, dtype=torch.float32).to(self.device)
            return X_train_t, y_train_t, noise_train_t

        return X_train_t, y_train_t, None

    def _train_batch(
        self, batch: Tuple[torch.Tensor, ...], mll: MarginalLogLikelihood
    ) -> torch.Tensor:
        """
        Process a single training batch, returning the computed loss as a torch.Tensor.
        """
        if self.use_noise_from_data:
            x_batch, y_batch, noise_batch = batch
        else:
            x_batch, y_batch = batch
            noise_batch = None

        with gpytorch.settings.num_likelihood_samples(self.n_samples_train):
            output = self(x_batch)
            if noise_batch is not None:
                loss = -mll(output, y_batch, fixed_noise=noise_batch)
            else:
                loss = -mll(output, y_batch)

        return loss

    def fit(self, df: pd.DataFrame) -> Self:
        """
        Train the model. Restart and initialize inducing points with new random samples
        if the training r2 is below the restart_r2_threshold.
        """

        X_train_t, y_train_t, _ = self._prepare_training_data(df)

        for attempt in range(self.max_restarts + 1):
            # Check if n_inducing_points is greater than the number of training datapoints
            if self.n_inducing_points > X_train_t.size(0):
                logger.warning(
                    f"Number of inducing points ({self.n_inducing_points}) is greater than "
                    f"the number of training datapoints ({X_train_t.size(0)}). Consider reducing "
                    "the number of inducing points."
                )
            else:
                # Initialize inducing points with training data samples for each GP in first layer
                with torch.no_grad():
                    inducing_points_list = []
                    for _ in range(self.num_hidden_dims):
                        dim_indices = torch.randperm(X_train_t.size(0))[: self.n_inducing_points]
                        inducing_points_list.append(X_train_t[dim_indices].clone())

                    hidden_inducing_tensor = torch.stack(inducing_points_list)
                    self.hidden_layer.variational_strategy.inducing_points.copy_(
                        hidden_inducing_tensor
                    )

                    self.last_layer.variational_strategy.inducing_points.copy_(
                        torch.randn(self.n_inducing_points, self.hidden_layer.output_dims)
                    )

            self.fit_single_attempt(df)

            with torch.no_grad():
                self.eval()
                predictions = self(X_train_t).mean.mean(0)
                train_r2 = r2_score(y_train_t.cpu().numpy(), predictions.cpu().numpy())

            # If r2 is above threshold, break
            if train_r2 >= self.restart_r2_threshold:
                break
            else:
                logger.info(
                    f"Training r2: {train_r2} below treshold of {self.restart_r2_threshold}."
                )
                if attempt < self.max_restarts:
                    logger.info("Restarting training with new random samples.")
                # initialize the model with new random samples
                self.init_model()

        return self

    def fit_single_attempt(self, df: pd.DataFrame) -> Self:
        # Prepare training data
        X_train_t, y_train_t, noise_train_t = self._prepare_training_data(df)

        if noise_train_t is not None:
            train_dataset = TensorDataset(X_train_t, y_train_t, noise_train_t)
            self.likelihood = HeteroscedasticGaussianLikelihood()
            self.to(self.device)
        else:
            train_dataset = TensorDataset(X_train_t, y_train_t)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        mll = DeepApproximateMLL(
            PredictiveLogLikelihood(self.likelihood, self, X_train_t.shape[-2])
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_patience,
            cooldown=self.cooldown,
            factor=self.lr_factor,
        )

        best_loss = float("inf")
        n = 0  # Counter for early stopping

        for iter in range(self.training_iter):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = self._train_batch(batch, mll)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch[0]) / len(X_train_t)

            if self.verbose:
                print(
                    f"Iter {iter} - Loss: {total_loss:.3f} - "
                    + f"Learning rate: {optimizer.param_groups[0]['lr']}"
                )

            # Early stopping check
            if total_loss < best_loss:
                best_loss = total_loss
                n = 0
            else:
                n += 1
                if iter > int(0.2 * self.training_iter) and n > self.es_patience:
                    logger.info(
                        f"Training loss has not improved for {self.es_patience} epochs. "
                        + f"Stopping early at iter {iter}."
                    )
                    break
            scheduler.step(total_loss)

        return self

    def predict(
        self, df: pd.DataFrame, return_std: bool = False, return_grads: bool = False
    ) -> TargetValues:
        self.eval()
        self.likelihood.eval()
        set_seed()
        check_columns_in_dataframe(df, self.features)
        X = df[self.features].to_numpy()
        X = self.feature_scaler.transform(X)
        X = torch.tensor(X, requires_grad=return_grads, dtype=torch.float32).to(self.device)

        means = []
        plos = []
        phis = []
        grads = []

        # Setting the number of samples to draw from the variational distributions
        # through out the layers - higher number of samples will give more accurate
        # predictions but will be slower.
        with gpytorch.settings.num_likelihood_samples(self.n_samples_pred):
            data_loader = DataLoader(X, batch_size=self.eval_batch_size)
            for x_batch in data_loader:
                x_batch = x_batch.to(self.device)
                if return_grads:
                    preds = self.likelihood(self(x_batch))

                    # Compute gradients of the predictions w.r.t the inputs
                    grads.append(
                        torch.autograd.grad(
                            outputs=preds.mean.mean(0).sum(),
                            inputs=x_batch,
                            retain_graph=False,
                        )[0]
                    )

                else:
                    with torch.no_grad():
                        preds = self.likelihood(self(x_batch))

                means.append(preds.mean)

                if return_std:
                    # Get the lower and upper confidence bounds (95% confidence interval)
                    # this is used to compute the standard deviation
                    plo, phi = preds.confidence_region()
                    plos.append(plo)
                    phis.append(phi)

        if return_grads:
            gradients = torch.cat(grads, dim=0).detach().cpu().numpy()
            # Scale gradients back to the original feature space
            # Multiply by target scale over feature scale as gradients are
            # defined by d_target/d_feature
            gradients *= self.target_scaler.scale_ / self.feature_scaler.scale_
            gradients_dict = {
                feature: gradients[:, idx].tolist() for idx, feature in enumerate(self.features)
            }

        if return_std:
            pred_low = self.target_scaler.inverse_transform(
                torch.cat(plos, dim=-1).mean(0).detach().cpu().numpy().reshape(-1, 1)
            )
            pred_high = self.target_scaler.inverse_transform(
                torch.cat(phis, dim=-1).mean(0).detach().cpu().numpy().reshape(-1, 1)
            )
            pred_std = (pred_high - pred_low) * 0.25
            pred_std_list = list(pred_std[:, 0])

        return TargetValues(
            target_name=self.target_col,
            value_list=list(
                self.target_scaler.inverse_transform(
                    torch.cat(means, dim=-1).mean(0).detach().cpu().numpy().reshape(-1, 1)
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
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_hidden_dims": self.num_hidden_dims,
            "n_samples_train": self.n_samples_train,
            "n_samples_pred": self.n_samples_pred,
            "feature_scaler_name": self.feature_scaler_name,
            "target_scaler_name": self.target_scaler_name,
            "add_linear_kernel": self.add_linear_kernel,
        }

    def construct_model_data(self) -> Dict[str, Any]:
        model_data = {
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
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
        model = cls(
            features=model_dict["features"],
            target_col=model_dict["target_col"],
            n_inducing_points=model_dict["n_inducing_points"],
            nu=model_dict["nu"],
            training_iter=model_dict["training_iter"],
            es_patience=model_dict["es_patience"],
            learning_rate=model_dict["learning_rate"],
            batch_size=model_dict["batch_size"],
            eval_batch_size=model_dict["eval_batch_size"],
            num_hidden_dims=model_dict["num_hidden_dims"],
            n_samples_train=model_dict["n_samples_train"],
            n_samples_pred=model_dict["n_samples_pred"],
            feature_scaler_name=model_dict["feature_scaler_name"],
            target_scaler_name=model_dict["target_scaler_name"],
            add_linear_kernel=model_dict["add_linear_kernel"],
        )

        # Initialize the model with the saved state_dict
        model.load_state_dict(model_dict["state_dict"])
        model.to(model.device)

        # Set scalers
        model.target_scaler = model_dict["target_scaler"]
        model.feature_scaler = model_dict["feature_scaler"]
        return model

    @classmethod
    def load_model(cls, folder_path: str) -> Self:
        folder_path = Path(folder_path)
        model_artifact_path = folder_path / "model.pkl"
        # Load the model data from the pickle file
        with open(model_artifact_path, "rb") as f:
            model_dict = pickle.load(f)
        return cls.load_model_from_dict(model_dict)
