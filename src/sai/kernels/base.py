"""Base Sampler Class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from src.sai.types import DataSet, GradEstimator, ParamTree, PRNGKey


class Sampler(ABC):
    """Sampler Class."""

    def __init__(
        self,
        grad_estimator: GradEstimator,
        position: Optional[ParamTree] = None,
        state: "Optional[Sampler.State]" = None,
        **kwargs,
    ):
        """Initialize Sampler.

        Args:
            grad_estimator: Gradient function of the log-density at current position.
            position: Passed to Sampler.State.
            state: Passed to Sampler.State.
            **kwargs: Runtime static args passed to sampling step.
        """
        self._grad_estimator = grad_estimator
        self._compile_sample_step(**kwargs)
        self.state = self.State(position=position, state=state)

    def update_state(self, *args):
        """Generate a new sample and save the sampler state internally."""
        self.state = self._sample_step(self.state, *args)

    def _compile_sample_step(self, **kwargs):
        """Parallelize and jit function that generates a new sample."""
        self._sample_step = jax.pmap(
            partial(self._sample_step, **kwargs),
            in_axes=(0, 0, 0, None),
        )

    @partial(
        jax.tree_util.register_dataclass,
        data_fields=["position"],
        meta_fields=[],
    )
    @dataclass
    class State:
        """The class for the state of the sampler."""

        position: ParamTree

        def __init__(
            self,
            position: Optional[ParamTree] = None,
            state: "Optional[Sampler.State]" = None,
        ):
            """Initialize the state for a sampler.

            Args:
                position: Initialize without a state or replace the position of the state.
                state: Initialize from (warmup) state if given.
            """
            if position is not None:
                self.position = position
            elif state is not None:
                self.position = state.position
            else:
                raise ValueError("Initialization needs either a state or a position.")

        @property
        def zeros(self):
            """Get an array of zeros with the shape of state.position."""
            return jax.tree.map(jnp.zeros_like, self.position)

        @property
        def ones(self):
            """Get an array of ones with the shape of state.position."""
            return jax.tree.map(jnp.ones_like, self.position)

    @abstractmethod
    def _sample_step(
        self,
        state: State,
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float,
    ) -> State:
        """Generate a new sample.

        Note: Define default values here.
        """
