"""Utils for the Laplace approximation."""
import yaml
from typing import List
import enum
from dataclasses import dataclass, field
from flax.training.train_state import TrainState
from functools import partial
import jax
import jax.numpy as jnp
import optax
import numpyro.distributions as dist
from dataclasses import dataclass, field
from typing import (
    List,
)

class Task(enum.Enum):
    """Task type."""

    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'

class SubsetOfParams(enum.Enum):
    """Subset of parameters to be modeled."""

    ALL = 'all'
    LAST_LAYER = 'last_layer'

class HessianFactorization(enum.Enum):
    """
    Factorization type of the Hessian.
    All variants employ the GGN approximation.
    """

    FULL = 'full'
    DIAG = 'diagonal_factorization'  # Not implemented
    KFAC = 'kronecker_factorization'  # Not implemented

class PredictiveApproximation(enum.Enum):
    """Approximation type of the PPD."""

    CF = 'closed_form'
    MC = 'monte_carlo'

def count_params(params: dict) -> int:
    """Get the number of parameters in a model."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))

@jax.jit
def train_step_mlp(
    state: TrainState,
    X: jnp.array,
    y: jnp.array,
) -> TrainState:
    """Train step, only training the endpoints and not the curve points."""

    def nll_loss_fn(params: dict, y: jnp.array):
        """Negative log likelihood loss function."""
        preds = state.apply_fn(params, X)
        nll = constant_variance_nll(preds, y)
        return jnp.mean(nll)

    grad_fn = jax.value_and_grad(nll_loss_fn)
    loss, grads = grad_fn(state.params, y)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@jax.jit
def val_step_mlp(
    state: TrainState,
    X: jnp.array,
    y: jnp.array,
) -> TrainState:
    """Train step, only training the endpoints and not the curve points."""

    def nll_loss_fn(params: dict, y: jnp.array):
        """Negative log likelihood loss function."""
        preds = state.apply_fn(params, X)
        nll = constant_variance_nll(preds, y)
        return jnp.mean(nll)

    loss = nll_loss_fn(state.params, y)
    return loss

@dataclass
class ExpConfigLaplace:
    experiment_name: str = "laplace"
    # LA config
    task: Task = Task.REGRESSION
    aleatoric_var: str = "one"
    subset_of_params: SubsetOfParams = field(default_factory=SubsetOfParams.LAST_LAYER)
    hessian_factorization: HessianFactorization = field(default_factory=HessianFactorization.FULL)
    predictive_approximation: PredictiveApproximation = field(default_factory=PredictiveApproximation.CF)

    # MLP config
    depth: int = 3
    width: int = 16

    # MLP and CNN config
    activation: str = 'relu'
    use_bias: bool = True

    dataset_list: List[str] = field(
        default_factory=lambda: [
            "airfoil.data",
            "concrete.data",
            "energy.data",
            "yacht.data",
            "bikesharing.data",
        ]
    )

    # config for evaluation
    test_size: float = 0.2
    val_size: float = 0.1

    epochs: int = 10_000
    steps: str = "all"
    lr: int = 5e-3
    optim: str = "adamw"
    val_loss_stopping: bool = False

    # seed config
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

@dataclass
class ExpConfigMFVI:
    experiment_name: str = "mfvi"
    task: Task = Task.REGRESSION

    # MFVI config
    sigma_obs: float = 0.1
    prior_scale: float = 1.0

    # MLP config
    depth: int = 3
    width: int = 16

    # MLP and CNN config
    activation: str = 'relu'
    use_bias: bool = True

    dataset_list: List[str] = field(
        default_factory=lambda: [
            "airfoil.data",
            "concrete.data",
            "energy.data",
            "yacht.data",
            "bikesharing.data",
        ]
    )

    # config for evaluation
    test_size: float = 0.2
    val_size: float = 0.1
    datapoint_limit: int = None

    epochs: int = 10_000
    steps: str = "all"
    lr: int = 5e-3
    optim: str = "adamw"
    val_loss_stopping: bool = False

    # seed config
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


def task_constructor(loader: yaml.Loader, node: yaml.Node) -> Task:
    value = node.value
    try:
        return Task(value)
    except ValueError:
        raise ValueError(f"Invalid value for Task enum: {value}")


def subset_of_params_constructor(
    loader: yaml.Loader, node: yaml.Node
) -> SubsetOfParams:
    value = node.value
    try:
        return SubsetOfParams(value)
    except ValueError:
        raise ValueError(f"Invalid value for SubsetOfParams enum: {value}")

def hessian_factorization_constructor(
    loader: yaml.Loader, node: yaml.Node
) -> HessianFactorization:
    value = node.value
    try:
        return HessianFactorization(value)
    except ValueError:
        raise ValueError(f"Invalid value for HessianFactorization enum: {value}")
    
def predictive_approximation_constructor(
    loader: yaml.Loader, node: yaml.Node
) -> PredictiveApproximation:
    value = node.value
    try:
        return PredictiveApproximation(value)
    except ValueError:
        raise ValueError(f"Invalid value for PredictiveApproximation enum: {value}")
        
    
def load_config_from_yaml(file_path: str) -> ExpConfigLaplace:
    with open(file_path, "r") as file:
        yaml_data = yaml.load(file, Loader=yaml.Loader)

    config = ExpConfigLaplace(**yaml_data)
    return config

def load_config_from_yaml_vi(file_path: str) -> ExpConfigMFVI:
    with open(file_path, "r") as file:
        yaml_data = yaml.load(file, Loader=yaml.Loader)

    config = ExpConfigMFVI(**yaml_data)
    return config


def constant_variance_nll(batch_preds, batch_target, scale=0.1):
    return -jax.scipy.stats.norm.logpdf(
        batch_target,
        loc=batch_preds.squeeze(),
        scale=scale,
    ).mean()


def categorical_nll(batch_preds, batch_target):
    return -dist.Categorical(logits=batch_preds).log_prob(batch_target).mean()


def train_step_cnn(
    state: TrainState,
    X: jnp.array,
    y: jnp.array,
) -> TrainState:
    """Train step for the CNN."""

    def nll_loss_fn(params: dict, y: jnp.array):
        """Negative log likelihood loss function."""
        logits = state.apply_fn(params, X)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.mean(nll)

    grad_fn = jax.value_and_grad(nll_loss_fn)
    loss, grads = grad_fn(state.params, y)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def val_step_cnn(
    state: TrainState,
    X: jnp.array,
    y: jnp.array,
) -> float:
    """Validation step for the CNN."""
    logits = state.apply_fn(state.params, X)
    nll = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(nll)


def pointwise_lppd(lvals: jnp.ndarray, y: jnp.ndarray, task: Task) -> jnp.ndarray:
    """Calculate the pointwise log predictive probability density.

    Parameters:
    ----------
    lvals : jnp.ndarray
        Logits or mean and variance of the predictive distribution.
        Either shape (n_chains, n_samples, n_obs, 2) for regression or
        (n_chains, n_samples, n_obs, n_classes) for classification.
    y : jnp.ndarray
        target values of shape (n_obs).
    task : Task
        Type of learning target (classification or regression).

    Notes:
    ----------
    If len(lvals.shape) == 3 a chain dimension is added.
    If len(lvals.shape) == 2 a chain and sample dimension is added.

    Returns:
    ----------
    jnp.ndarray:
        Pointwise log predictive probability density.
    """
    if len(lvals.shape) >= 4:
        lvals.reshape(*lvals.shape[:2], -1)
    if len(lvals.shape) == 3:
        lvals = lvals.reshape(1, *lvals.shape)
    elif len(lvals.shape) == 2:
        lvals = lvals.reshape(1, 1, *lvals.shape)

    if task == Task.REGRESSION:
        lppd_pointwise = jnp.stack(
            [
                dist.Normal(
                    loc=x[:, :, 0],
                    scale=jnp.exp(x[:, :, 1]).clip(min=1e-6, max=1e6),
                )
                .log_prob(y)
                .squeeze()
                for x in lvals
            ]
        )
    elif task == Task.CLASSIFICATION:
        lppd_pointwise = jnp.stack(
            [dist.Categorical(logits=x).log_prob(y).squeeze() for x in lvals]
        )
    return lppd_pointwise


def lppd(lppd_pointwise: jnp.ndarray) -> jnp.ndarray:
    """Calculate the log predictive probability density."""

    b = 1 / jnp.prod(jnp.array(lppd_pointwise.shape[:-1]))
    axis = tuple(range(len(lppd_pointwise.shape) - 1))
    return jax.scipy.special.logsumexp(lppd_pointwise, b=b, axis=axis).mean()

