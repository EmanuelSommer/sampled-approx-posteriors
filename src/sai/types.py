"""Type definitions for the module."""

import typing
from pathlib import Path
from typing import Callable, Protocol

import jax
from blackjax.base import SamplingAlgorithm

ParamTree: typing.TypeAlias = dict[str, "jax.Array | ParamTree"]
FileTree: typing.TypeAlias = dict[str, "Path | FileTree"]
PRNGKey: typing.TypeAlias = jax.Array
DataSet: typing.TypeAlias = tuple[jax.Array, jax.Array]
Kernel = Callable[..., SamplingAlgorithm]


class PosteriorFunction(Protocol):
    """Protocol for Posterior Function used in full-batch sampling.

    Signature:
        `(position: ParamTree) -> jax.Array`
    """

    def __call__(self, position: ParamTree) -> jax.Array:
        """Posterior Function for full-batch sampling."""
        ...


class GradEstimator(Protocol):
    """Protocol for Gradient Estimator function used in mini-batch sampling.

    Signature:
        `(position: ParamTree, x: jax.Array, y: jax.Array) -> jax.Array`
    """

    def __call__(self, position: ParamTree, x: jax.Array, y: jax.Array) -> jax.Array:
        """Gradient Estimator function for mini-batch sampling."""
        ...

# Warmup Functions must return warmup state and tuned parameters as a dictionary
WarmupResult = tuple[typing.Any, dict[str, typing.Any]]
