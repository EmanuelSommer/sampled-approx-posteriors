"""Various metrics for evaluating the performance of a probabilistic model."""

import pickle
import warnings
from dataclasses import Field
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence

import blackjax.diagnostics as blackjax_diag
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from flax.struct import PyTreeNode

from src.sai.config.data import Task

class Metrics(PyTreeNode):
    """Metrics base class."""

    step: jnp.ndarray

    def __len__(self):
        """Len of the metrics."""
        return self.shape[-1]

    @property
    def n_chains(self):
        """Number of chains."""
        return self.shape[0]

    @property
    def shape(self):
        """Shape of the metrics."""
        return self.step.shape

    def __getitem__(self, index: int | str):
        """Get item by index or key."""
        if isinstance(index, str):
            return getattr(self, index)
        return self.replace(
            **{k: getattr(self, k)[index] for k, _ in self.__dict__.items()}
        )

    def pad(self, length: int):
        """Pad the metrics with nan."""
        if length > self.shape[-1]:
            return self.replace(
                **{
                    k: jnp.pad(
                        getattr(self, k),
                        pad_width=(
                            jnp.array((0, 0, 0, length - self.shape[-1])).reshape(2, 2)
                        ),
                        mode="constant",
                        constant_values=jnp.nan,
                    )
                    for k, _ in self.__dict__.items()
                }
            )
        return self

    @classmethod
    def empty(cls):
        """Create an empty metrics object."""
        em = jnp.empty(shape=(1, 0))
        return cls(**{k: em for k, _ in cls.__dataclass_fields__.items()})

    @classmethod
    def vstack(cls, metrics: list["Metrics"]):
        """Stack the metrics vertically."""
        if len(set([m.shape[-1] for m in metrics])) != 1:
            # pad with nan
            max_len = max([m.shape[-1] for m in metrics])
            for i, m in enumerate(metrics):
                if m.shape[-1] < max_len:
                    m.replace(
                        step=jnp.pad(m.step, ((0, 0), (0, max_len - m.shape[-1])))
                    )
                    metrics[i] = m.pad(max_len)
        return cls._stack(metrics, jnp.vstack)

    @classmethod
    def cstack(cls, metrics: list["Metrics"]):
        """Merge metrics columnwise."""
        return cls._stack(metrics, jnp.column_stack)

    @classmethod
    def _stack(
        cls,
        metrics: list["Metrics"],
        stack_fn: Callable[[Sequence[jax.Array]], jax.Array],
    ):
        """Stack the metrics."""
        if not metrics:
            return cls.empty()
        return cls(
            **{
                k: stack_fn([getattr(m, k) for m in metrics])
                for k, _ in metrics[0].__dict__.items()
            }
        )

    @property
    def is_empty(self):
        """Check if the metrics is empty."""
        return self.step.size == 0

    def savez(self, path: str):
        """Save the metrics to a npz file."""
        jnp.savez(path, **self.__dict__)


class MetricsStore(PyTreeNode):
    """Metrics store class."""

    train: Metrics
    valid: Metrics
    test: Optional[Metrics] = None

    def save(self, path: str):
        """Save the metrics store to a file."""
        parent = Path(path).parent
        if not Path(parent).exists():
            Path(parent).mkdir(parents=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def vstack(cls, metrics: list["MetricsStore"]):
        """Stack the metrics store vertically."""
        return cls(
            **{
                k: v.vstack([m[k] for m in metrics])
                for k, v in metrics[0].__dict__.items()
                if isinstance(v, Metrics)
            }
        )

    @classmethod
    def load(cls, path: str) -> "MetricsStore":
        """Load the metrics store from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_batch(cls, paths: Sequence[str]):
        """Load a batch of metrics stores."""
        metrics = [cls.load(p) for p in paths]
        cls_name = metrics[0].train.__class__

        return cls(
            train=cls_name.vstack([m.train for m in metrics]),
            valid=cls_name.vstack([m.valid for m in metrics]),
            test=cls_name.vstack([m.test for m in metrics]),  # type: ignore
        )

    def __getitem__(self, key: str):
        """Get item by key."""
        return getattr(self, key)

    @classmethod
    def empty(cls):
        """Create an empty metrics store."""
        return cls(
            **{
                k: v.type.empty()
                for k, v in cls.__dataclass_fields__.items()
                if isinstance(v, Field) and v.type is Metrics
            }
        )


class RegressionMetrics(Metrics):
    """Regression metrics."""

    nlll: jnp.ndarray
    rmse: jnp.ndarray


# LPPD ---------------------------------------------------------------------------


def lppd(lppd_pointwise: jnp.ndarray) -> jnp.ndarray:
    """Calculate the log predictive probability density (LPPD).

    Args:
        lppd_pointwise (jnp.ndarray):

    Returns:
        jnp.ndarray: Aggregated log predictive probability density. (Scalar)

    """
    b = 1 / jnp.prod(jnp.array(lppd_pointwise.shape[:-1]))
    axis = tuple(range(len(lppd_pointwise.shape) - 1))
    return jax.scipy.special.logsumexp(lppd_pointwise, b=b, axis=axis).mean()


def lppd_pointwise(pred_dist: jnp.ndarray, y: jnp.ndarray, task: Task) -> jnp.ndarray:
    """Compute pointwise log predictive probability density (LPPD) for predictions.

    Args:
        pred_dist: Predicted distribution parameters with shape (n_chains, n_samples, n_obs, ...).

        y: Target values with shape (n_obs).
        task: The task type.

    Returns:
        Pointwise LPPD values with shape (n_chains, n_samples, n_obs).

    """
    if len(pred_dist.shape) == 3:
        pred_dist = pred_dist.reshape(1, *pred_dist.shape)
    elif len(pred_dist.shape) == 2:
        pred_dist = pred_dist.reshape(1, 1, *pred_dist.shape)
    elif len(pred_dist.shape) == 1:
        raise ValueError("Predictions must have at least 2 dimensions.")

    if task in [Task.REGRESSION, Task.MEAN_REGRESSION]:
        return stats.norm.logpdf(
            x=y,
            loc=pred_dist[..., 0],
            scale=jnp.exp(pred_dist[..., 1]).clip(min=1e-6, max=1e6),
        )

# Performance Metrics ------------------------------------------------------------


def accuracy(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Compute the accuracy of predictions over chains and samples.

    Args:
        pred (jnp.ndarray): Predicted class labels of shape (n_chains, n_samples, n_obs).
        target (jnp.ndarray): True class labels of shape (n_obs).

    Returns:
        jnp.ndarray: Classification accuracy.
    """
    mode_pred = stats.mode(pred.reshape(-1, *pred.shape[2:]), axis=0).mode
    return jnp.mean(mode_pred == target)


def rmse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Compute the root mean squared error of predictions over chains and samples.

    Args:
        pred (jnp.ndarray): Predicted values of shape (n_chains, n_samples, n_obs).
        target (jnp.ndarray): True values of shape (n_obs).

    Returns:
        jnp.ndarray: Root mean squared error.
    """
    return jnp.sqrt(jnp.mean((pred.mean(axis=(0, 1)) - target) ** 2))


# Regression loss functions -----------------------------------------------------


def GaussianNLLLoss(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray):
    """Compute the Gaussian negative log-likelihood loss.

    Args:
        y: The target values of shape (n_obs).
        mu: The predicted mean values of shape (..., n_obs).
        sigma: The predicted standard deviation values of shape (..., n_obs).
    """
    sigma = jnp.clip(sigma, 1e-5)  # Ensure stability in calculations
    return 0.5 * jnp.log(2 * jnp.pi * sigma**2) + ((y - mu) ** 2) / (2 * sigma**2)


def SELoss(y: jnp.ndarray, mu: jnp.ndarray):
    """Compute the squared error loss."""
    return (y - mu) ** 2


# Trace Variances ---------------------------------------------------------------


def between_chain_var(x: jnp.ndarray) -> jnp.ndarray:
    """Calculate the between chain variance of an array.

    Args:
        x: Array of shape (n_chains, n_samples, ...).
    """
    return x.mean(axis=1).var(axis=0, ddof=1)


def within_chain_var(x: jnp.ndarray) -> jnp.ndarray:
    """Calculate the within chain variance of an array.

    Args:
        x: Array of shape (n_chains, n_samples, ...).
    """
    return x.var(axis=1, ddof=1).mean(axis=0)


# Running metrics ----------------------------------------------------------------


def running_mean(x: jnp.ndarray, axis: int):
    """Compute running mean along a specified axis."""
    cumsum = jnp.cumsum(x, axis=axis)
    shape = [1] * len(x.shape)
    shape[axis] = x.shape[axis]
    count = jnp.arange(1, x.shape[axis] + 1).reshape(shape)
    return cumsum / count


def running_chainwise_lppd(lppd_pointwise: jnp.ndarray):
    """Calculate the chainwise log predictive probability density on n_samples axis.

    Args:
        lppd_pointwise: Should have shape (n_chains, n_samples, n_obs).

    Returns:
        jnp.ndarray: Running chainwise log predictive probability density of shape
            (n_chains, n_samples).
    """
    run_mean_log = jnp.log(running_mean(jnp.exp(lppd_pointwise), axis=1))
    masked_run_mean_log = jnp.where(jnp.isneginf(run_mean_log), 0.0, run_mean_log)
    return masked_run_mean_log.mean(axis=-1)


def running_lppd(lppd_pointwise: jnp.ndarray):
    """Calculate the running log predictive probability density on n_samples axis.

    Args:
        lppd_pointwise: Should have shape (n_chains, n_samples, n_obs).

    Returns:
        jnp.ndarray: Running log predictive probability density of shape
            (n_samples).
    """
    n_chains, n_samples, _ = lppd_pointwise.shape
    densities_lin = jnp.exp(lppd_pointwise)  # shape: (n_chains, n_samples, n_obs)
    sum_over_chains = densities_lin.sum(axis=0)  # shape: (n_samples, n_obs)
    cumsum_over_samples = jnp.cumsum(sum_over_chains, axis=0)
    counts = jnp.arange(1, n_samples + 1)[:, None]  # shape: (n_samples, 1)
    avg_density = cumsum_over_samples / (counts * n_chains)  # shape: (n_samples, n_obs)
    log_avg_density = jnp.log(avg_density)  # shape: (n_samples, n_obs)
    lppd_cumulative = log_avg_density.mean(axis=1)  # shape: (n_samples,)
    return lppd_cumulative


def running_seq_chain_lppd(lppd_pointwise: jnp.ndarray):
    """Calculate the sequential chainwise log predictive probability density.

    Basically we evaluate how much an additional chain improves the LPPD.

    Args:
        lppd_pointwise: Should have shape (n_chains, n_samples, n_obs).

    Returns:
        jnp.ndarray: Sequential chainwise log predictive probability density of shape
            (n_chains,).
    """
    n_chains, n_samples, _ = lppd_pointwise.shape
    densities_lin = jnp.exp(lppd_pointwise)  # shape: (n_chains, n_samples, n_obs)
    sum_over_samples = densities_lin.sum(axis=1)  # shape: (n_chains, n_obs)
    cumsum_over_chains = jnp.cumsum(sum_over_samples, axis=0)
    counts = jnp.arange(1, n_chains + 1)[:, None]  # shape: (n_chains, 1)
    avg_density = cumsum_over_chains / (counts * n_samples)  # shape: (n_chains, n_obs)
    log_avg_density = jnp.log(avg_density)  # shape: (n_chains, n_obs)
    lppd_cumulative = log_avg_density.mean(axis=1)  # shape: (n_chains,)
    return lppd_cumulative

# Calibration -------------------------------------------------------------------


def coverage_weighting(
    nominal_coverage: jnp.ndarray | list[float],
    kappa: float = 1.0,
) -> jnp.ndarray:
    """Calculate a possibly increasing coverage weighting.

    Defaullt is a linear weighting. For a constant weighting, set kappa=0.

    Args:
        nominal_coverage: The nominal coverage of the intervals.
        kappa: The kappa parameter for the weighting.
    """
    if isinstance(nominal_coverage, list):
        nominal_coverage = jnp.array(nominal_coverage)
    if kappa == 0:
        return jnp.ones_like(nominal_coverage) / len(nominal_coverage)
    return (nominal_coverage**kappa) / jnp.sum(nominal_coverage**kappa)


def get_coverage_quantiles(coverage: float) -> jnp.ndarray:
    """Get the quantiles for a given coverage level."""
    return jnp.array([0.5 - coverage / 2, 0.5 + coverage / 2])


def calculate_coverage_regression(
    nominal_coverages: list[float],
    y: jnp.ndarray,
    preds: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate the coverage of the prediction intervals.

    Args:
        nominal_coverage: The nominal coverage of the intervals.
        y: The true values.
        preds: The predicted values of shape (n_chains, n_samples, n_obs).

    Returns:
        The observed coverage of the prediction intervals of shape (nominal_coverages).

    """

    @partial(jax.vmap, in_axes=(0, None, None))
    @jax.jit
    def coverage(nc, y, preds):
        lower_upper = jnp.quantile(a=preds, q=get_coverage_quantiles(nc), axis=0)
        return jnp.mean((lower_upper[0, :] <= y) & (lower_upper[1, :] >= y))

    return coverage(jnp.array(nominal_coverages), y, preds.reshape(-1, preds.shape[-1]))


def calibration_error_regression(
    nominal_coverage: jnp.ndarray | list[float],
    observed_coverage: jnp.ndarray,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Calculate the (squared) calibration error for regression.

    Args:
        nominal_coverage: The nominal coverage of the intervals.
        observed_coverage: The observed coverage of the intervals.
        weights: The weights for the calibration error. Must sum up to 1.

    Returns:
        The calibration error.
    """
    if isinstance(nominal_coverage, list):
        nominal_coverage = jnp.array(nominal_coverage)

    if weights is None:
        return jnp.sqrt(jnp.mean(jnp.square(nominal_coverage - observed_coverage)))
    assert jnp.isclose(jnp.sum(weights), 1)
    return jnp.sqrt(
        jnp.mean(weights * jnp.square(nominal_coverage - observed_coverage))
    )


def calibration_gap_regression(
    nominal_coverage: jnp.ndarray | list[float],
    observed_coverage: jnp.ndarray,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Calculate the calibration gap."""
    if isinstance(nominal_coverage, list):
        nominal_coverage = jnp.array(nominal_coverage)
    assert nominal_coverage.shape == observed_coverage.shape

    if weights is None:
        return jnp.sum(nominal_coverage - observed_coverage)

    assert jnp.isclose(jnp.sum(weights), 1)
    return (
        jnp.sum(weights * (nominal_coverage - observed_coverage))
        * nominal_coverage.size
    )


def calibration_gap_classification(
    nominal_coverages: list[float],
    y: jnp.ndarray,
    preds: jnp.ndarray,
    pred_dist: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the difference between the accuracy and the confidence level.

    Args:
        nominal_coverages: the bin edges like (0, nominal_coverages[0]], ...,
          (nominal_coverages[-1], 1]
        y: The true values. (n_obs)
        preds: The predicted values of shape (n_chains, n_samples, n_obs).
        pred_dist: The logits of shape (n_chains, n_samples, n_obs, n_classes).

    Returns:
        The calibration gap of shape (n_bins) and the bin sizes of shape (n_bins).
    """
    preds_probs = jax.nn.softmax(pred_dist, axis=-1)
    preds_probs = preds_probs.reshape(-1, *preds_probs.shape[2:])

    bins = []
    bin_sizes = []
    left_bound = 0.0
    for i in range(len(nominal_coverages)):
        right_bound = nominal_coverages[i]
        reduced_preds = jnp.take_along_axis(
            preds_probs, y[None, :, None], axis=-1
        ).squeeze(-1)
        avg_probs = jnp.mean(reduced_preds, axis=0)
        assert reduced_preds.shape == preds_probs.shape[:-1]
        assert avg_probs.shape == y.shape

        mask = (avg_probs > left_bound) & (avg_probs <= right_bound)
        majority_preds = stats.mode(
            preds.reshape(-1, preds.shape[-1]), axis=0
        ).mode.squeeze()
        # assert majority_preds.shape == y.shape
        bin_preds = majority_preds[mask]
        bin_y = y[mask]
        # assert bin_preds.shape[-1] == bin_y.size

        # calculate the accuracy of the bin
        bin_accuracy = jnp.mean(bin_preds == bin_y)
        # calculate the confidence level of the bin on the true class
        bin_confidence_level = jnp.mean(avg_probs[mask])
        if bin_y.size == 0:
            bins.append(0)
        else:
            bins.append(bin_accuracy - bin_confidence_level)
        bin_sizes.append(bin_y.size)
        left_bound = right_bound

    return jnp.array(bins), jnp.array(bin_sizes)


def calibration_error_classification(
    calibration_gaps: jnp.ndarray,
    bin_sizes: jnp.ndarray,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Calculate the expected calibration error (ECE) for classification.

    Args:
        calibration_gaps: The calibration gaps of shape (n_bins).
        bin_sizes: The sizes of the bins of shape (n_bins).
        weights: The weights for the calibration error. Must sum up to 1.

    Returns:
        The expected calibration error.
    """
    all_samples = bin_sizes.sum()
    if weights is None:
        return jnp.sum(jnp.abs(calibration_gaps) * bin_sizes) / all_samples

    assert jnp.isclose(jnp.sum(weights), 1)
    return jnp.sum(weights * jnp.abs(calibration_gaps) * bin_sizes) / all_samples


# Diagnostics -------------------------------------------------------------------


def rank_normalize_array(samples: jnp.ndarray) -> jnp.ndarray:
    """Rank normalization of a JAX array.

    Args:
        samples: Expects an array of shape (n_chains, n_samples, ...).
    """
    n_samples = jnp.prod(jnp.array(samples.shape))
    # get the overall ranks
    ranks = jax.scipy.stats.rankdata(samples, axis=None).reshape(samples.shape)
    tmp = (ranks - 0.375) / (n_samples + 0.25)
    return jax.scipy.stats.norm.ppf(tmp)


def gelman_split_r_hat(
    samples: jnp.ndarray,
    n_splits: int,
    rank_normalize: bool = True,
) -> jnp.ndarray:
    """Calculate the split Gelman-Rubin R-hat statistic for samples of MCMC chains.

    Args:
        samples: MCMC chains of shape (n_chains, n_samples, ...).
        n_splits: Number of splits to split each chain into.
        rank_normalize: Whether to rank normalize before calculating the R-hat.
    """
    n_chains = samples.shape[0]
    n_samples = samples.shape[1] // n_splits
    samples = samples[:, : n_samples * n_splits, ...]  # discard overflow

    if (n_samples % 1) != 0:
        raise ValueError("Number of samples must be divisible by n_splits")

    if n_samples < 50:
        warnings.warn(
            message="Number of samples should be at least 50x the number of splits",
            category=UserWarning,
        )

    splits_total = n_chains * n_splits

    if rank_normalize:
        samples = jnp.apply_along_axis(
            rank_normalize_array, 0, samples.reshape(-1, *samples.shape[2:])
        ).reshape(samples.shape)

    splits = samples.reshape(splits_total, -1, *samples.shape[2:])
    wcv = within_chain_var(splits)
    bcv = between_chain_var(splits)
    numerator = ((n_samples - 1) / n_samples) * wcv + bcv
    rhat = jnp.sqrt(numerator / wcv)
    return rhat


def split_chain_r_hat(
    samples: jnp.ndarray,
    n_splits: int,
    rank_normalize: bool = True,
) -> jnp.ndarray:
    """Calculate the split chain R-hat statistic for samples of MCMC chains.

    Args:
        samples: MCMC chains of shape (n_chains, n_samples, ...).
        n_splits: Number of splits to split each chain into.
        rank_normalize: Whether to rank normalize before calculating the R-hat.

    Returns:
        jnp.ndarray: Split chain R-hat of shape (n_chains, ...).
    """
    return jax.vmap(gelman_split_r_hat, in_axes=(0, None, None))(
        samples[:, None, ...], n_splits, rank_normalize
    )


def effective_sample_size(
    samples: jnp.ndarray, rank_normalize: bool = True
) -> jnp.ndarray:
    """Calculate the effective sample size of an array.

    Args:
        samples: Array of shape (n_chains, n_samples, ...). Where n_samples is the
            number of samples in the chain. And `...` can be any number of dimensions.
            The effective sample size is calculated along the n_samples axis.
        rank_normalize: Whether to rank normalize before calculating the ESS.

    Returns:
        jnp.ndarray: Effective sample size of the array of shape (n_chains, ...).
    """
    if rank_normalize:
        samples = jnp.apply_along_axis(
            rank_normalize_array, 0, samples.reshape(-1, *samples.shape[2:])
        ).reshape(samples.shape)
    return jax.vmap(blackjax_diag.effective_sample_size)(samples[:, None, ...])


@jax.jit
def wasserstein_distance_gaussian(gaussian_1, gaussian_2):
    """
    Compute the squared Wasserstein distance between two multivariate Gaussian distributions
    with diagonal covariance matrices.

    Args:
    gaussian_1: nx2 matrix with initial column resembling the mean values and the second resembling the diagonal of the covariance matrix
    gaussian_2: nx2 matrix with initial column resembling the mean values and the second resembling the diagonal of the covariance matrix

    Returns:
    - The squared Wasserstein distance.
    """
    # https://www.stat.cmu.edu/~larry/=sml/Opt.pdf
    # only for diagonal variance

    def _wasserstein_distance_gaussian(gaussian_1, gaussian_2):
        mu_1 = gaussian_1[..., 0]
        mu_2 = gaussian_2[..., 0]
        sigma_squared_1 = gaussian_1[..., 1]
        sigma_squared_2 = gaussian_2[..., 1]

        mean_diff = jnp.sum((mu_1 - mu_2) ** 2)
        B2 = jnp.sum(sigma_squared_1 + sigma_squared_2 - 2 *
                     jnp.abs(jnp.sqrt(sigma_squared_1) * jnp.sqrt(sigma_squared_2)))

        return mean_diff + B2

    def _dist_equal():
        # Avoid numerical error
        return jnp.zeros((), dtype=jnp.float32)

    return jax.lax.cond(jnp.all(gaussian_1 == gaussian_2),
                        lambda x: _dist_equal(),
                        lambda x: _wasserstein_distance_gaussian(x[0], x[1]),
                        (gaussian_1, gaussian_2))