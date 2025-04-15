import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.flatten_util import ravel_pytree




def log_prior(theta, scale=1.0):
    """
    Standard normal prior on flattened parameters:
      log p(theta) = -0.5 * sum(theta^2 / scale^2).
    """
    return -0.5 * jnp.sum((theta / scale)**2)

def forward_apply(params, batch_x, model_apply_fn):
    """
    Applies the model with un-frozen parameters to the given batch_x.
    `model_apply_fn` is partially applied from model's init or apply call.
    """
    return model_apply_fn({"params": params}, batch_x)

def log_likelihood(params, batch_x, batch_y, model_apply_fn, sigma_obs=0.1):
    """
    Gaussian log-likelihood with fixed observation noise sigma_obs.
    """
    preds = forward_apply(params, batch_x, model_apply_fn)
    return -0.5 * jnp.sum((batch_y.squeeze() - preds.squeeze()) ** 2) / (sigma_obs**2)
