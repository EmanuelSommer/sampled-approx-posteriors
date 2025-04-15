"""Efficiently get Prediction Samples from a Bayesian Model."""

from pathlib import Path
from typing import Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from tqdm import tqdm

from src.sai.config.data import Task
from src.data import BaseLoader, Split
from src.sai.training.utils import count_chains_and_samples, load_posterior_samples


def predict_from_post_samples_batch(
    model: nn.Module,
    batched_samples,
    data_loader: BaseLoader,
    data_split: Split,
    data_batch_size: int,
    **kwargs,
) -> jnp.ndarray:
    """Efficient forward pass predictions for a batch of samples.

    Args:
        model: Flax module
        batched_samples: Batch of posterior samples tree of arrays of shape
          (n_samples, ...)
        data_loader: Data loader
        data_split: Data split to predict on
        data_batch_size: Batch size for data loader
        kwargs: Additional arguments to pass to model forward pass

    Returns:
        Predictions of shape (n_samples, n_obs, ...)
    """
    subdevided_params = False
    if "sigma" in batched_samples:
        subdevided_params = True
        sigma = batched_samples["sigma"]
        batched_samples = batched_samples["params"]

    @jax.jit
    def predict_fn(params, x):
        """Forward pass for a single set of parameters."""
        return model.apply({"params": params}, x, **kwargs)

    @jax.jit
    def predict_fn_batch_stats(params, x):
        """Forward pass for a single set of parameters."""
        return model.apply(
            {"params": params["params"], "batch_stats": params["batch_stats"]},
            x,
            train=False,
            **kwargs,
        )

    pred_fn = predict_fn_batch_stats if "batch_stats" in batched_samples else predict_fn

    predictions = []

    for batch_x, _ in data_loader.iter(
        split=data_split,
        batch_size=data_batch_size,
        chains=jnp.array((0,)),
        shuffle=False,
    ):
        batch_x = batch_x.squeeze(axis=0)
        batch_preds = jax.vmap(lambda params: pred_fn(params, batch_x))(batched_samples)
        # squeeze the second dimension if it is 1
        if batch_preds.shape[1] == 1:
            batch_preds = jnp.squeeze(batch_preds, axis=1)
        predictions.append(batch_preds)
    predictions = jnp.concatenate(predictions, axis=1)

    if subdevided_params:
        datalen = predictions.shape[1]  # type: ignore
        sigma = jnp.expand_dims(sigma, axis=-1)
        sigma = jnp.repeat(sigma, datalen, axis=1)
        sigma = jnp.expand_dims(sigma, axis=-1)
        predictions = jnp.concatenate([predictions, sigma], axis=-1)

    return predictions


def predict_from_post_samples(
    model: nn.Module,
    params_path: Union[str, Path],
    tree_path: Union[str, Path],
    data_loader: BaseLoader,
    data_split: Split,
    data_batch_size: int,
    samples_batch_size: int = 100,
    progress: bool = True,
    **kwargs,
) -> jnp.ndarray:
    """Efficient forward pass predictions.

    Args:
        model: Flax module
        params_path: Path to the chain directories of the samples
        tree_path: Path to the tree
        data_loader: Data loader
        data_split: Data split to predict on
        data_batch_size: Batch size for data loader
        samples_batch_size: Batch size for processing the posterior samples
        progress: Show progress bar
        kwargs: Additional arguments to pass to model forward pass

    Returns:
        Predictions of shape (n_chains, n_samples, n_obs, ...)
    """
    n_chains, n_samples = count_chains_and_samples(directory=params_path)

    if progress:
        chain_iterator = tqdm(range(n_chains), desc="Predicting Chains")
    else:
        chain_iterator = range(n_chains)

    predictions = []
    for i in chain_iterator:
        # now loop over the samples by always loading samples_batch_size samples
        # and predict on the data
        chain_predictions = []
        for j in range(0, n_samples, samples_batch_size):
            samples = load_posterior_samples(
                directory=params_path,
                tree_path=tree_path,
                chain_indices=[i],
                sample_indices=list(range(j, min(j + samples_batch_size, n_samples))),
            )
            # squeeze the chain dimension for the next function
            samples = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), samples)
            preds = predict_from_post_samples_batch(
                model=model,
                batched_samples=samples,
                data_loader=data_loader,
                data_split=data_split,
                data_batch_size=data_batch_size,
                **kwargs,
            )
            chain_predictions.append(preds)
        chain_predictions = jnp.concatenate(chain_predictions, axis=0)
        predictions.append(chain_predictions)
    return jnp.stack(predictions)


def sample_from_predictions(
    predictions: jnp.ndarray, task: Task, rng_key: PRNGKey, sampling_factor: int = 1
) -> jnp.ndarray:
    """Sample predictions from n_sample forward pass predictions.

    Args:
        predictions: Predictions of shape (n_chain, n_samples, n_obs, ...)
        task: Task of the model
        rng_key: Random key
        sampling_factor: Factor to sample more predictions

    Returns:
        Sampled predictions of shape (n_chain, n_samples * sampling_factor, n_obs).
    """
    if sampling_factor < 1:
        raise ValueError("sampling_factor must be >= 1")

    # Generate multiple keys for sampling if needed
    rng_keys = jax.random.split(rng_key, sampling_factor)

    if task in (Task.REGRESSION, Task.MEAN_REGRESSION):
        loc = predictions[..., 0]
        scale = jnp.exp(predictions[..., 1]).clip(min=1e-6, max=1e6)

        # Sample with broadcasting to match sampling_factor
        samples = [
            jax.random.normal(rng, shape=loc.shape) * scale + loc for rng in rng_keys
        ]
        predictions = (
            jnp.concatenate(samples, axis=1) if sampling_factor > 1 else samples[0]
        )

    return predictions
