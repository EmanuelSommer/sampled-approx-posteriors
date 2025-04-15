"""Convert frequentist Flax modules to Bayesian NNs."""
import operator
from functools import reduce
from typing import Optional

import chex
import flax.linen as nn
import jax.numpy as jnp
import numpyro.distributions as dist


def get_by_path(root: dict, items: list):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)


def get_flattened_keys(d: dict, sep='.'):

    """Recursively get & concat the keys of a dictionary and its subdictionaries.
    
    Arguments
    ---------
    d: dict
        Dictionary to get the keys from
    
    sep: str
        Separator for the keys (default: '.')

    Returns
    -------
    list
        List of keys.
    """    
    keys = []
    for k, v in d.items():
        if isinstance(v, dict):
            keys.extend([f'{k}{sep}{kk}' for kk in get_flattened_keys(v)])
        else:
            keys.append(k)
    return keys


class ProbModelBuilder:
    """Convert frequentist Flax modules to Bayesian Numpyro modules."""

    def __init__(
        self,
        module: nn.Module,
        prior_config: dict,
        params: dict,
        seed: int,
        evaluate: bool = False,
        full_batch_len: Optional[int] = None,
    ):
        """Initialize the model builder."""
        self.module = module
        self.prior_config = prior_config
        self.params = params
        self.seed = seed
        self.evaluate = evaluate
        self.full_batch_len = full_batch_len if full_batch_len is not None else 1
        self.priorsdict = self.set_prior()

    def map_distribution(self, config: dict) -> dist.Distribution:
        """Map the distribution from string to callable."""
        if config['dist'] == 'Normal':
            return dist.Normal(scale=config['sd'])
        elif config['dist'] == 'Laplace':
            return dist.Laplace(scale=config['sd'])
        elif config['dist'] == 'LogNormal':
            return dist.LogNormal(scale=config['sd'])
        else:
            raise ValueError(f'Distribution {config["dist"]} not implemented')

    def set_constant_prior(
        self, params: dict, distr: dist.Distribution = dist.Normal()
    ) -> dict:
        """
        Set a constant prior for the parameters of the model.

        Arguments
        ---------

        params: dict
            Dictionary of parameters of the model.
        
        distr: dist.Distribution
            Distribution to use as prior.

        Returns
        -------
        dict 
            Dictionary of priors.
        """
        return {k: distr for k in get_flattened_keys(params)}

    def set_prior(self):
        """
        Set the prior for the parameters of the model.

        Returns
        -------
            dict: 
            Dictionary of priors with the keys being the concatenated parameter names.
        """
        if self.prior_config['scheme'] == 'equal':
            distr = self.map_distribution(self.prior_config['details'])
            priors = self.set_constant_prior(self.params, distr=distr)
        else:
            raise ValueError(f'Scheme {self.prior_config["scheme"]} not implemented')

        # Add prior for the scale parameter if specified
        if 'scale' in self.prior_config:
            scale_prior_config = self.prior_config['scale']
            priors['scale'] = self.map_distribution(scale_prior_config)

        return priors

    def log_prior(self, params: dict):
        """Log prior for the parameters of the model."""
        prior_logprobs = jnp.array([0.0])
        for k in self.priorsdict.keys():
            levels = k.split('.')
            prior_logprobs = jnp.concatenate(
                [
                    prior_logprobs,
                    self.priorsdict[k].log_prob(jnp.ravel(get_by_path(params, levels))),
                ]
            )
        return jnp.sum(prior_logprobs)

    def log_prior_hessian(self, params: dict, layer: str = None): 
        """Hessian of the log prior. Only applicable to isotropic Gaussian priors.

        Arguments
        ---------
        params: dict
            Dictionary of parameters of the model.

        layer: str
            Layer to compute the prior hessian for.
            Needs to correspond to the name of the layer in the model.
        
        Returns
        -------
        jnp.ndarray
            Hessian of the log prior.
        """
        prior_precisions = jnp.array([])
        layers = self.priorsdict.keys()
        if layer:
            prefixes = [key.split('.')[0] for key in self.priorsdict.keys()]
            last_prefix = next(
                (prefix for prefix in reversed(prefixes) if prefix.startswith(layer)),
                None,
            )
            layers = {
                k: v for k, v in self.priorsdict.items() if k.startswith(last_prefix)
            }

        for k in layers:
            levels = k.split('.')
            prior_precisions = jnp.concatenate(
                [
                    prior_precisions,
                    jnp.repeat(
                        (1 / self.priorsdict[k].variance),
                        repeats=jnp.ravel(get_by_path(params['params'], levels)).shape[
                            0
                        ],
                    ),
                ]
            )
        return -jnp.diag(prior_precisions)

    def log_likelihood(
        self, params: dict, X: jnp.ndarray, Y: jnp.ndarray, type: str = 'regr'
    ):
        """
        Evaluate Log likelihood of the model.

        Arguments
        ---------
        params: dict
            Parameters of the model (potentially many nested levels).

        X: jnp.ndarray
            Input data either full batch or mini-batch of dimensions
            with the first dimension being the batch size.

        Y: jnp.ndarray
            Target data either full batch or mini-batch. The dimension has
            to be the same as the first dimension of X (batch size,). For classification
            problems, Y is expected to be integer-encoded.

        type: str
            Type of learning target (either 'regr' or 'class').
        """
        chex.assert_shape(Y, (X.shape[0],))

        if 'varphi' in params.keys():
            lvals = self.module.apply({'params': {'varphi': params['varphi']}}, X)
        else:
            lvals = self.module.apply({'params': params}, X)
        if type == 'regr':
            if lvals.shape[-1] == 2:
                return jnp.sum(
                    dist.Normal(
                        loc=lvals[..., 0],
                        scale=jnp.exp(lvals[..., 1]).clip(min=1e-6, max=1e6),
                    ).log_prob(Y)
                )
            else:
                return jnp.sum(
                    dist.Normal(loc=lvals.squeeze(), scale=params['scale']).log_prob(Y)
                )
        else:
            return jnp.sum(dist.Categorical(logits=lvals).log_prob(Y))

    def log_unnormalized_posterior(
        self,
        params: dict,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        type: str = 'regr',
        temp: float = 1.0,
    ):
        """
        Log unnormalized posterior (potential) for the model.

        Arguments
        ---------

        params: dict
            Parameters of the model (potentially many nested levels).

        X: jnp.ndarray
            Input data either full batch or mini-batch of dimensions
            with the first dimension being the batch size.

        Y: jnp.ndarray
            Target data either full batch or mini-batch. The dimension has
            to be the same as the first dimension of X (batch size,). For classification
            problems, Y is expected to be integer-encoded.

        type: str
            Type of learning target (either 'regr' or 'class').

        temp: float
            Temperature for the posterior. Default is 1.0. 
        """
        if self.full_batch_len > 1:
            adjusted_batch_len = self.full_batch_len / X.shape[0]
        else:
            adjusted_batch_len = 1.0
        return (
            self.log_prior(params)
            + (1 / temp)
            * self.log_likelihood(params, X, Y, type=type)
            * adjusted_batch_len
        )