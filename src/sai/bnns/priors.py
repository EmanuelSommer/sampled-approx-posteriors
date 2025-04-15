"""Priors for Bayesian Neural Networks."""

import logging
import math
from collections.abc import Sequence
from enum import Enum
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax.flatten_util import ravel_pytree
from jax.tree_util import DictKey, tree_map_with_path

from src.sai.training.utils import split_key_by_tree
from src.sai.types import ParamTree, PRNGKey

from blackjax.util import generate_gaussian_noise

logger = logging.getLogger(__name__)


def _compute_fans(
    shape: Sequence[int],
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
) -> tuple[float, float]:
    """Compute effective input and output sizes for a linear or convolutional layer.

    Axes not in in_axis, out_axis, or batch_axis are assumed to constitute the
        "receptive field" of a convolution (kernel spatial dimensions). Taken from
        https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L200
    """
    if len(shape) <= 1:
        raise ValueError(
            f"Can't compute input and output sizes of a {len(shape)}"
            "-dimensional weights tensor. Must be at least 2D."
        )

    if isinstance(in_axis, int):
        in_size = shape[in_axis]
    else:
        in_size = math.prod([shape[i] for i in in_axis])
    if isinstance(out_axis, int):
        out_size = shape[out_axis]
    else:
        out_size = math.prod([shape[i] for i in out_axis])
    if isinstance(batch_axis, int):
        batch_size = shape[batch_axis]
    else:
        batch_size = math.prod([shape[i] for i in batch_axis])
    receptive_field_size = math.prod(shape) / in_size / out_size / batch_size
    fan_in = in_size * receptive_field_size
    fan_out = out_size * receptive_field_size
    return fan_in, fan_out


class PriorDist(str, Enum):
    """Prior Distribution Names.

    Note:
        This can either be a general distribution, where the user can pass the
        `parameters` from the `configuration` file like: `Normal`, `Laplace`, etc.

        Example::
        ```yaml
        name: Normal
        parameters:
            loc: 5.0
            scale: 2.0
        ```

        ```python
        print(get_prior(**parameters))
        # Normal(loc=5.0, scale=2.0)
        ```

        Or a `pre-defined` distribution, where the user chooses the `name`
        and the `parameters` are fully or partially defined.

        ```yaml
        name: StandardNormal
        parameters:
            limit: 5.0
        ```

        ```python
        print(get_prior(**parameters))
        # Normal(loc=0.0, scale=1.0, limit=5.0)
        ```
    """

    NORMAL = "Normal"
    StandardNormal = "StandardNormal"
    LAPLACE = "Laplace"
    GLOROT = "Glorot"
    HE = "He"
    GloLap = "GloLap"
    GloUnif = "GloUnif"
    EpsStandardNormal = "EpsStandardNormal" # Ziyin


    def get_prior(self, **parameters):
        """Return a prior class."""
        return Prior.from_name(self, **parameters)


class Prior(NamedTuple):
    """Base Class for Priors."""

    init: Callable[[ParamTree, PRNGKey], jax.Array]
    log_prior: Callable[[ParamTree], jax.Array]
    name: str

    @classmethod
    def from_name(cls, name: PriorDist, **parameters):
        """Initialize the prior class instance."""
        if name == PriorDist.StandardNormal:
            return cls(
                name=PriorDist.StandardNormal,
                init=init_normal(),
                log_prior=log_prior_normal(),
            )
        elif name == PriorDist.NORMAL:
            return cls(
                name=PriorDist.NORMAL,
                init=init_normal(**parameters),
                log_prior=log_prior_normal(**parameters),
            )
        elif name == PriorDist.LAPLACE:
            return cls(
                name=PriorDist.LAPLACE,
                init=init_laplace(**parameters),
                log_prior=log_prior_laplace(**parameters),
            )
        elif name == PriorDist.GLOROT:
            return cls(
                name=PriorDist.GLOROT,
                init=init_glorot(**parameters),
                log_prior=log_prior_glorot(**parameters),
            )
        elif name == PriorDist.HE:
            return cls(
                name=PriorDist.HE,
                init=init_he(**parameters),
                log_prior=log_prior_he(**parameters),
            )
        elif name == PriorDist.GloLap:
            return cls(
                name=PriorDist.GloLap,
                init=init_glorot_laplace(**parameters),
                log_prior=log_prior_glorot_laplace(**parameters),
            )
        elif name == PriorDist.GloUnif:
            return cls(
                name=PriorDist.GloUnif,
                init=init_glorot_uniform(**parameters),
                log_prior=log_prior_glorot_uniform(**parameters),
            )
        elif name == PriorDist.EpsStandardNormal:
            return cls(
                name=PriorDist.EpsStandardNormal,
                init=init_eps_normal(**parameters),
                log_prior=log_prior_eps_normal(**parameters),
            )
        raise NotImplementedError(f"Prior Distribution {name} not implemented.")


def init_normal(
    loc: float = 0.0, scale: float = 1.0
) -> Callable[[PRNGKey, ParamTree], jax.Array]:
    """Initialize from Normal distribution."""

    @jax.jit
    def rng_normal(leaf: jax.Array, rng: PRNGKey) -> jax.Array:
        sample = jax.random.normal(rng, shape=jnp.shape(leaf), dtype=jnp.dtype(leaf))
        return sample * scale + loc

    def init(rng: PRNGKey, params: ParamTree) -> jax.Array:
        rng_tree = split_key_by_tree(rng, pytree=params)
        return jax.tree.map(rng_normal, params, rng_tree)

    return init


def log_prior_normal(
    loc: float = 0.0, scale: float = 1.0
) -> Callable[[ParamTree], jax.Array]:
    """Evaluate Normal prior on all weights."""

    def log_prior(params: ParamTree) -> jax.Array:
        scores = stats.norm.logpdf(ravel_pytree(params)[0], loc=loc, scale=scale)
        return jnp.sum(scores)

    return log_prior

def init_eps_normal(
    loc: float = 0.0, 
    scale: float = 1.0, 
    mean_eps_scale: float = 0.1, 
    var_eps_scale: float = 0.1, 
    offset_rng: int = 42
) -> Callable[[PRNGKey, ParamTree], tuple[jax.Array, jax.Array, jax.Array]]:
    """Initialize from Normal distribution with epsilon mean and variance offset for every param."""

    @jax.jit
    def rng_eps_normal(leaf: jax.Array, rng: PRNGKey) -> tuple[jax.Array, jax.Array]:
        sample = jax.random.normal(rng, shape=jnp.shape(leaf), dtype=jnp.dtype(leaf))
        return sample * scale + loc, leaf  # leaf is placeholder for offset shape

    def init(rng: PRNGKey, params: ParamTree) -> tuple[ParamTree, ParamTree, ParamTree]:
        rng_tree = split_key_by_tree(rng, pytree=params)
        params_and_shapes = jax.tree.map(rng_eps_normal, params, rng_tree)
        params = jax.tree.map(lambda x: x[0], params_and_shapes)
        
        # Generate mean and variance offsets using blackjax
        offset_key1, offset_key2 = jax.random.split(jax.random.key(offset_rng))
        mean_offsets = generate_gaussian_noise(
            rng_key=offset_key1, 
            position=params,
            mu=0.0, 
            sigma=mean_eps_scale
        )
        var_offsets = generate_gaussian_noise(
            rng_key=offset_key2, 
            position=params,
            mu=1.0, 
            sigma=var_eps_scale
        )
        
        # Ensure variance offsets are positive using softplus
        var_offsets = jax.tree.map(jax.nn.softplus, var_offsets)
        
        # Add mean offsets to params
        final_params = jax.tree.map(lambda p, m: p + m, params, mean_offsets)
        return final_params, mean_offsets, var_offsets

    return init

def log_prior_eps_normal(
    loc: float = 0.0, 
    scale: float = 1.0, 
    mean_eps_scale: float = 0.5, 
    var_eps_scale: float = 0.5, 
    offset_rng: int = 42
) -> Callable[[ParamTree], jax.Array]:
    """Evaluate Normal prior on all weights with epsilon mean and variance offset."""

    def log_prior(params: ParamTree) -> jax.Array:
        # Generate mean and variance offsets internally
        offset_key1, offset_key2 = jax.random.split(jax.random.key(offset_rng))
        mean_offsets = generate_gaussian_noise(
            rng_key=offset_key1, 
            position=params, 
            mu=0.0, 
            sigma=mean_eps_scale
        )
        var_offsets = generate_gaussian_noise(
            rng_key=offset_key2, 
            position=params, 
            mu=1.0, 
            sigma=var_eps_scale
        )
        
        # Ensure variance offsets are positive using softplus
        var_offsets = jax.tree.map(jax.nn.softplus, var_offsets)
        
        params_flat = ravel_pytree(params)[0]
        mean_offsets_flat = ravel_pytree(mean_offsets)[0]
        var_offsets_flat = ravel_pytree(var_offsets)[0]
        
        scores = stats.norm.logpdf(
            params_flat, 
            loc=mean_offsets_flat, 
            scale=scale * var_offsets_flat
        )
        return jnp.sum(scores)

    return log_prior


def init_laplace(
    loc: float = 0.0, scale: float = 1.0
) -> Callable[[PRNGKey, ParamTree], jax.Array]:
    """Initialize from Laplace distribution."""

    @jax.jit
    def rng_laplace(leaf: jax.Array, rng: PRNGKey) -> jax.Array:
        sample = jax.random.laplace(rng, shape=jnp.shape(leaf), dtype=jnp.dtype(leaf))
        return sample * scale + loc

    def init(rng: PRNGKey, params: ParamTree) -> jax.Array:
        rng_tree = split_key_by_tree(rng, pytree=params)
        return jax.tree.map(rng_laplace, params, rng_tree)

    return init


def log_prior_laplace(
    loc: float = 0.0, scale: float = 1.0
) -> Callable[[ParamTree], jax.Array]:
    """Evaluate Laplace prior on all weights."""

    def log_prior(params: ParamTree) -> jax.Array:
        scores = stats.laplace.logpdf(ravel_pytree(params)[0], loc=loc, scale=scale)
        return jnp.sum(scores)

    return log_prior


def init_glorot(
    weight_names: list[str] = ["kernel"],
    factor: float = 1.0,
) -> Callable[[PRNGKey, ParamTree], jax.Array]:
    """Glorot initialization from untruncated Normal."""

    def rng_glorot(path: tuple[DictKey], leaf: jax.Array, rng: PRNGKey) -> jax.Array:
        shape = jnp.shape(leaf)
        # standard deviation based on Glorot
        if path[-1].key in weight_names:
            fan_in, fan_out = _compute_fans(shape)
            fan_avg = jnp.mean(jnp.array((fan_in, fan_out)))
            scale = jax.lax.rsqrt(fan_avg) * factor
        else:  # standard normal prior
            scale = 1
        # untruncated normal because of sampler
        sample = jax.random.normal(rng, shape=shape, dtype=jnp.dtype(leaf))
        return sample * scale

    def init(rng: PRNGKey, params: ParamTree) -> jax.Array:
        rng_tree = split_key_by_tree(rng, pytree=params)
        return tree_map_with_path(rng_glorot, params, rng_tree)

    return init


def log_prior_glorot(
    weight_names: list[str] = ["kernel"],
    factor: float = 1.0,
) -> Callable[[ParamTree], jax.Array]:
    """Evaluate Glorot inspired prior on specified weights."""

    def glorot_score(path: tuple[DictKey], leaf: jax.Array) -> jax.Array:
        # standard deviation based on Glorot
        if path[-1].key in weight_names:
            fan_in, fan_out = _compute_fans(jnp.shape(leaf))
            fan_avg = jnp.mean(jnp.array((fan_in, fan_out)))
            scale = jax.lax.rsqrt(fan_avg) * factor
        else:  # standard normal prior
            scale = 1
        return stats.norm.logpdf(leaf, scale=scale)

    def log_prior(params: ParamTree) -> jax.Array:
        scores = tree_map_with_path(glorot_score, params)
        return jnp.sum(ravel_pytree(scores)[0])

    return log_prior


def init_he(
    weight_names: list[str] = ["kernel"],
) -> Callable[[PRNGKey, ParamTree], jax.Array]:
    """Glorot initialization from untruncated Normal."""

    def rng_he(path: tuple[DictKey], leaf: jax.Array, rng: PRNGKey) -> jax.Array:
        shape = jnp.shape(leaf)
        # standard deviation based on He
        if path[-1].key in weight_names:
            fan_in, _ = _compute_fans(shape)
            scale = jnp.sqrt(2 / fan_in)
        else:  # standard normal prior
            scale = 1
        # untruncated normal because of sampler
        sample = jax.random.normal(rng, shape=shape, dtype=jnp.dtype(leaf))
        return sample * scale

    def init(rng: PRNGKey, params: ParamTree) -> jax.Array:
        rng_tree = split_key_by_tree(rng, pytree=params)
        return tree_map_with_path(rng_he, params, rng_tree)

    return init


def log_prior_he(
    weight_names: list[str] = ["kernel"],
) -> Callable[[ParamTree], jax.Array]:
    """Evaluate Glorot inspired prior on specified weights."""

    def he_score(path: tuple[DictKey], leaf: jax.Array) -> jax.Array:
        # standard deviation based on He
        if path[-1].key in weight_names:
            fan_in, _ = _compute_fans(jnp.shape(leaf))
            scale = jnp.sqrt(2 / fan_in)
        else:  # standard normal prior
            scale = 1
        return stats.norm.logpdf(leaf, scale=scale)

    def log_prior(params: ParamTree) -> jax.Array:
        scores = tree_map_with_path(he_score, params)
        return jnp.sum(ravel_pytree(scores)[0])

    return log_prior


def init_glorot_laplace(
    weight_names: list[str] = ["kernel"],
) -> Callable[[PRNGKey, ParamTree], jax.Array]:
    """Glorot initialization from untruncated Normal."""

    def rng_glo_lap(path: tuple[DictKey], leaf: jax.Array, rng: PRNGKey) -> jax.Array:
        shape = jnp.shape(leaf)
        # standard deviation based on Glorot
        if path[-1].key in weight_names:
            fan_in, fan_out = _compute_fans(shape)
            fan_avg = jnp.mean(jnp.array((fan_in, fan_out)))
            scale = jax.lax.rsqrt(2.0 * fan_avg)
        else:  # laplace prior with sd 1
            scale = jax.lax.rsqrt(2.0)
        # untruncated normal because of sampler
        sample = jax.random.laplace(rng, shape=shape, dtype=jnp.dtype(leaf))
        return sample * scale

    def init(rng: PRNGKey, params: ParamTree) -> jax.Array:
        rng_tree = split_key_by_tree(rng, pytree=params)
        return tree_map_with_path(rng_glo_lap, params, rng_tree)

    return init


def log_prior_glorot_laplace(
    weight_names: list[str] = ["kernel"],
) -> Callable[[ParamTree], jax.Array]:
    """Evaluate Glorot inspired prior on specified weights."""

    def glo_lap_score(path: tuple[DictKey], leaf: jax.Array) -> jax.Array:
        # standard deviation based on Glorot
        if path[-1].key in weight_names:
            fan_in, fan_out = _compute_fans(jnp.shape(leaf))
            fan_avg = jnp.mean(jnp.array((fan_in, fan_out)))
            scale = jax.lax.rsqrt(2.0 * fan_avg)
        else:  # laplace prior with sd 1
            scale = jax.lax.rsqrt(2.0)
        return stats.laplace.logpdf(leaf, scale=scale)

    def log_prior(params: ParamTree) -> jax.Array:
        scores = tree_map_with_path(glo_lap_score, params)
        return jnp.sum(ravel_pytree(scores)[0])

    return log_prior


def init_glorot_uniform(
    weight_names: list[str] = ["kernel"],
) -> Callable[[PRNGKey, ParamTree], jax.Array]:
    """Glorot initialization from untruncated Normal."""

    def rng_glo_unif(path: tuple[DictKey], leaf: jax.Array, rng: PRNGKey) -> jax.Array:
        shape = jnp.shape(leaf)
        # standard deviation based on Glorot
        if path[-1].key in weight_names:
            fan_in, fan_out = _compute_fans(shape)
            fan_avg = jnp.mean(jnp.array((fan_in, fan_out)))
            a = jnp.sqrt(3.0 / fan_avg)
        else:  # uniform prior with sd 1
            a = jnp.sqrt(3.0)
        return jax.random.uniform(
            rng, shape=shape, dtype=jnp.dtype(leaf), minval=-a, maxval=a
        )

    def init(rng: PRNGKey, params: ParamTree) -> jax.Array:
        rng_tree = split_key_by_tree(rng, pytree=params)
        return tree_map_with_path(rng_glo_unif, params, rng_tree)

    return init


def log_prior_glorot_uniform(
    weight_names: list[str] = ["kernel"],
) -> Callable[[ParamTree], jax.Array]:
    """Evaluate Glorot inspired prior on specified weights."""

    def glo_unif_score(path: tuple[DictKey], leaf: jax.Array) -> jax.Array:
        # standard deviation based on Glorot
        if path[-1].key in weight_names:
            fan_in, fan_out = _compute_fans(jnp.shape(leaf))
            fan_avg = jnp.mean(jnp.array((fan_in, fan_out)))
            a = jnp.sqrt(3.0 / fan_avg)
        else:  # uniform prior with sd 1
            a = jnp.sqrt(3.0)
        leaf = jnp.where(
            jnp.logical_or(leaf < -a, leaf > a),
            jnp.power(1000, -jnp.abs(leaf) + a),
            1,
        )
        return jnp.log(leaf)

    def log_prior(params: ParamTree) -> jax.Array:
        scores = tree_map_with_path(glo_unif_score, params)
        return jnp.sum(ravel_pytree(scores)[0])

    return log_prior
