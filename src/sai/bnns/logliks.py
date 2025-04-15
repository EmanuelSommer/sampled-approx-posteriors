"""Definition of ProbabilisticModels for Bayesian training."""

import logging

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from src.sai.bnns.priors import Prior
from src.sai.config.data import Task
from src.sai.types import ParamTree, PRNGKey

logger = logging.getLogger(__name__)


class ProbabilisticModel:
    """Convert frequentist Flax modules to Bayesian modules."""

    def __init__(self, module: nn.Module, prior: Prior, task: Task, n_batches: int = 1):
        """Basic Probabilistic Model implmentation for Bayesian training.

        Args:
            module: Flax module to use for Bayesian training
            prior: Prior distribution for the parameters.
            task: learning task of the model.
            n_batches: Number of batches sampler will see in one epoch.
        """
        self.task = task
        self.module = module
        self.n_batches = n_batches
        self.prior = prior

    def __str__(self):
        """Return informative string representation of the model."""
        return (
            f"{self.__class__.__name__}:\n"  # noqa
            f" | Task: {self.task.value}\n"
            f" | Batches: {self.n_batches}\n"
            f" | Prior: {self.prior.name.value}"
        )

    @property
    def minibatch(self):
        return self.n_batches > 1

    def init(self, rng: PRNGKey, x: jax.Array, train: bool = False):
        """Initialize from the prior of the probabilistic model."""
        position = self.module.init(rng, x, train=train)
        return self.prior.init(rng, position)

    def log_prior(self, position: ParamTree) -> jnp.ndarray:
        """Compute log prior for given parameters."""
        return self.prior.log_prior(position)

    def log_likelihood(
        self,
        position: ParamTree,
        x: jnp.ndarray,
        y: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Evaluate Log likelihood for given Parameter Tree and data.

        Args:
            position: Current position of the sampler.
            x: Input data of shape (batch_size, ...).
            y: Target data of shape (batch_size, ...).
            kwargs: Additional keyword arguments to pass to the model forward pass.
        """
        if self.task == Task.REGRESSION:
            lvals = self.module.apply({"params": position}, x, **kwargs)
            return jnp.nansum(
                stats.norm.logpdf(
                    x=y,
                    loc=lvals[..., 0],
                    scale=jnp.exp(lvals[..., 1]).clip(min=1e-6, max=1e6),
                )
            )
        if self.task == Task.MEAN_REGRESSION:
            lvals = self.module.apply({"params": position["params"]}, x, **kwargs)
            sigma = position["sigma"]

            return jnp.nansum(
                stats.norm.logpdf(
                    x=y,
                    loc=lvals.squeeze(),  # type: ignore[union-attr]
                    scale=jnp.exp(sigma.squeeze()).clip(min=1e-6, max=1e6),  # type: ignore[union-attr]
                )
            )
        else:
            raise NotImplementedError(
                f"Likelihood computation for {self.task} not implemented"
            )

    def log_unnormalized_posterior(
        self,
        position: ParamTree,
        x: jnp.ndarray,
        y: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Log unnormalized posterior (potential) for given parameters and data.

        Args:
            position: Current position of the sampler.
            x: Input data of shape (batch_size, ...).
            y: Target data of shape (batch_size, ...).
            **kwargs: Additional keyword arguments to pass to the model forward pass.
        """
        return (
            self.log_prior(position)
            + self.log_likelihood(position, x, y, **kwargs) * self.n_batches
        )
