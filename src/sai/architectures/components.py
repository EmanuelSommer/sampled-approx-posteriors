"""Basic NN building blocks."""

from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class FullyConnected(nn.Module):
    """Fully connected Neural Network."""

    hidden_sizes: tuple[int, ...]
    activation: Callable
    use_bias: bool = True
    last_layer_activation: Callable | None = None
    blockid: str | None = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass."""
        if self.blockid is None:
            blockid = ""
        else:
            blockid = f"{self.blockid}_"
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(
                features=hidden_size,
                dtype=self.dtype,
                use_bias=self.use_bias,
                name=f"{blockid}layer{i}",
            )(x)
            if i < len(self.hidden_sizes) - 1:
                x = self.activation(x)
            else:
                if self.last_layer_activation is not None:
                    x = self.last_layer_activation(x)
        return x