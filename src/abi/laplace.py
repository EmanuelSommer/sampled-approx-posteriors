"""Implementation of the Laplace approximation for BNNs."""
from dataclasses import dataclass
from typing import Callable, Final
import typing

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from numpyro.distributions import (
    Categorical,
    MultivariateNormal,
    Normal,
)

from src.flax2bnn import ProbModelBuilder
from src.abi.utils import (
    count_params,
    HessianFactorization,
    PredictiveApproximation,
    SubsetOfParams,
    Task,
)

ParamTree: typing.TypeAlias = dict[str, typing.Union[jax.Array, 'ParamTree']]

@dataclass(frozen=True)
class LaplaceApproximation:
    """Main class for the Laplace approximation.

    Parameters:
    -----------
    task: Task
        Task type (classification or regression).
    subset_of_params: SubsetOfParams
        Subset of parameters to be modeled with the LA.
    hessian_factorization: HessianFactorization
        Type of Hessian approximation/factorization.
    predictive_approximation: PredictiveApproximation
        Predictive approximation type.
    _input_to_pred_fn: Callable
        Function that maps input to predictions, e.g., a neural network.
        Should take as arguments batch_input and model_params. If number of params
        modeled exceeds one, the natural parameters need to be modeled with the
        predictive function.
    _pred_to_loss_fn: Callable
        Function that maps from predictions and true target to scalar loss.
        Should take as arguments batch_preds and batch_target.

    """

    task: Task
    subset_of_params: SubsetOfParams
    hessian_factorization: HessianFactorization
    predictive_approximation: PredictiveApproximation
    _input_to_pred_fn: Callable
    _pred_to_loss_fn: Callable

    def __post_init__(self):
        """
        Initialize differential functions.
        """

        def _extract_last_layer(model_params):
            """Extract only the last layer of the model param pytree."""
            # Traverse the 'params' subtree and identify the last layer.
            params = model_params.get('params', {})
            if not params:
                raise ValueError("Expected 'params' key in the parameter pytree.")
            # this assumes ordered layers
            last_layer_key = list(params.keys())[-1]
            last_layer_params = {last_layer_key: params[last_layer_key]}
            return {'params': last_layer_params}

        def _extract_except_last_layer(model_params):
            """Extract all layers except the last layer of the model param pytree."""
            # Traverse the 'params' subtree and identify the last layer.
            params = model_params.get('params', {})
            if not params:
                raise ValueError("Expected 'params' key in the parameter pytree.")
            # this assumes ordered layers
            last_layer_key = list(params.keys())[-1]
            except_last_layer_params = {
                k: v for k, v in params.items() if k != last_layer_key
            }
            return {'params': except_last_layer_params}
        
        _input_to_pred_jacobian = jax.jacrev(
            self._input_to_pred_fn, argnums=0, has_aux=False
        )
        # jacrev better for wide jacobians (number of natural parameters may exceed 1)

        if self.subset_of_params == SubsetOfParams.LAST_LAYER:

            _input_to_pred_fn_ll = lambda ll_params, except_ll_params, batch_input: self._input_to_pred_fn(
                {'params': {**except_ll_params['params'], **ll_params['params']}}, batch_input
            )
            _input_to_pred_jacobian = jax.jacrev(
                _input_to_pred_fn_ll, argnums=0, has_aux=False
            )

        _pred_to_loss_hessian = jax.vmap(
            jax.hessian(self._pred_to_loss_fn, argnums=0, has_aux=False),
            in_axes=(0, None),
            out_axes=0,
        )

        lax_factorization_index = (
            0
            if self.hessian_factorization == HessianFactorization.FULL
            else (1 if self.hessian_factorization == HessianFactorization.DIAG else 2)
        )

        object.__setattr__(self, '_input_to_pred_jacobian', _input_to_pred_jacobian)
        object.__setattr__(self, '_pred_to_loss_hessian', _pred_to_loss_hessian)
        object.__setattr__(self, '_extract_last_layer', _extract_last_layer)
        object.__setattr__(
            self, '_extract_except_last_layer', _extract_except_last_layer
        )

        object.__setattr__(self, 'lax_factorization_index', lax_factorization_index)

    def get_posterior_precision(
        self,
        full_batch_train_features: jnp.ndarray,
        full_batch_train_labels: jnp.ndarray,
        model_params: ParamTree,
        prob_model: ProbModelBuilder,
        batch_size: int = 64,
    ):
        """Compute the posterior precision matrix.

        Parameters
        ----------
        full_batch_train_features : jnp.ndarray
            Full batch array of features.
        full_batch_train_labels : jnp.ndarray
            Full batch array of labels.
        model_params : ParamTree
            Should be a flax param dict. The MAP estimate of the model parameters.
        prob_model : ProbModelBuilder
            A initialized instance of the ProbModelBuilder class.
        batch_size : int, optional
            Batch size for the generalized Gauss-Newton computation, by default 64.

        Returns
        -------
        jnp.ndarray
            Returns the posterior precision matrix. 
        """
        if self.subset_of_params == SubsetOfParams.LAST_LAYER:
            ll_model_params = self._extract_last_layer(model_params)
            except_ll_params = self._extract_except_last_layer(model_params)
            ll_amount_params = int(count_params(ll_model_params))
            ll_name = list(ll_model_params['params'].keys())[0]

        def _full_factorization():
            amount_train_samples = full_batch_train_features.shape[0]
            amount_model_params = int(count_params(model_params))

            def _ggn_sum_fn(
                model_params,
                batch_input,
                batch_target,
            ):
                def _ggn_prod_fn(pred_jacobian, loss_hessian):
                    chex.assert_rank(pred_jacobian, 2)
                    chex.assert_rank(loss_hessian, 2)
                    return pred_jacobian.T @ loss_hessian @ pred_jacobian

                _ggn_prod_vmap = jax.vmap(_ggn_prod_fn, in_axes=(0, 0))

                chex.assert_rank(batch_target, 1)
                chex.assert_shape(batch_input, (batch_target.shape[0], ...))

                batch_preds = self._input_to_pred_fn(model_params, batch_input)

                if self.subset_of_params == SubsetOfParams.ALL:
                    batch_preds_jacobian = self._input_to_pred_jacobian(
                        model_params, batch_input
                    )
                
                elif self.subset_of_params == SubsetOfParams.LAST_LAYER:
                    batch_preds_jacobian = self._input_to_pred_jacobian(
                        ll_model_params, except_ll_params, batch_input
                    )

                # Flatten jacobian
                batch_preds_jacobian = jnp.concatenate(
                    jax.tree_util.tree_flatten(
                        jax.tree_util.tree_map(
                            lambda x: x.reshape(x.shape[0], x.shape[1], -1),
                            batch_preds_jacobian,
                        )
                    )[0],
                    axis=2,
                )
                if self.subset_of_params == SubsetOfParams.LAST_LAYER:
                    chex.assert_shape(
                        batch_preds_jacobian,
                        (batch_preds.shape[0], batch_preds.shape[1], ll_amount_params),
                    )
                elif self.subset_of_params == SubsetOfParams.ALL:
                    chex.assert_shape(
                        batch_preds_jacobian,
                        (
                            batch_preds.shape[0],
                            batch_preds.shape[1],
                            amount_model_params,
                        ),
                    )  # batch_size x C x amount_model_params

                batch_loss_hessian = self._pred_to_loss_hessian(
                    batch_preds, batch_target
                )

                chex.assert_shape(
                    batch_loss_hessian,
                    (batch_input.shape[0], batch_preds.shape[1], batch_preds.shape[1]),
                )  # batch_size x C x C

                batch_ggn = _ggn_prod_vmap(batch_preds_jacobian, batch_loss_hessian)

                if self.subset_of_params == SubsetOfParams.LAST_LAYER:
                    chex.assert_shape(
                        batch_ggn,
                        (batch_preds.shape[0], ll_amount_params, ll_amount_params),
                    )  # batch_size x D x D
                elif self.subset_of_params == SubsetOfParams.ALL:
                    chex.assert_shape(
                        batch_ggn,
                        (
                            batch_preds.shape[0],
                            amount_model_params,
                            amount_model_params,
                        ),
                    )

                return batch_ggn.sum(axis=0)
            
            def get_batch_tabular(i):
                start = i * batch_size

                feature_batch = jax.lax.dynamic_slice(
                    full_batch_train_features,
                    (start, 0),
                    (batch_size, full_batch_train_features.shape[1]),
                )
                label_batch = jax.lax.dynamic_slice(
                    full_batch_train_labels, (start,), (batch_size,)
                )

                return feature_batch, label_batch

            def get_batch_images(i):
                start = i * batch_size

                # assumes shape (batch_dim, channel_dim, height_dim, width_dim)
                feature_batch = jax.lax.dynamic_slice(
                    full_batch_train_features,
                    (start, 0, 0, 0),
                    (
                        batch_size,
                        full_batch_train_features.shape[1],
                        full_batch_train_features.shape[2],
                        full_batch_train_features.shape[3],
                    ),
                )

                label_batch = jax.lax.dynamic_slice(
                    full_batch_train_labels, (start,), (batch_size,)
                )

                return feature_batch, label_batch

            num_batches = amount_train_samples // batch_size
            if full_batch_train_features.ndim == 4:
                feature_batches, label_batches = jax.vmap(get_batch_images)(
                    jnp.arange(num_batches)
                )
                chex.assert_shape(
                    feature_batches,
                    (
                        num_batches,
                        batch_size,
                        full_batch_train_features.shape[1],
                        full_batch_train_features.shape[2],
                        full_batch_train_features.shape[3],
                    ),
                )
            elif full_batch_train_features.ndim == 2:
                feature_batches, label_batches = jax.vmap(get_batch_tabular)(
                    jnp.arange(num_batches)
                )
                chex.assert_shape(
                    feature_batches,
                    (num_batches, batch_size, full_batch_train_features.shape[1]),
                )

            # Compute GGN as sum over batches
            if self.subset_of_params == SubsetOfParams.LAST_LAYER:
                initial_ggn_sum = jnp.zeros(shape=(ll_amount_params, ll_amount_params))
            elif self.subset_of_params == SubsetOfParams.ALL:
                initial_ggn_sum = jnp.zeros(
                    shape=(amount_model_params, amount_model_params)
                )

            def _scan_ggn_sum(carry_ggn_sum, batches):
                batch_input, batch_target = batches
                batch_ggn_sum = _ggn_sum_fn(
                    model_params,
                    batch_input,
                    batch_target,
                )
                updated_ggn_sum = carry_ggn_sum + batch_ggn_sum
                return updated_ggn_sum, None  # scan requires returning a pair

            final_ggn_sum, _ = jax.lax.scan(
                _scan_ggn_sum, initial_ggn_sum, (feature_batches, label_batches)
            )

            if self.subset_of_params == SubsetOfParams.LAST_LAYER:
                log_prior_hessian = prob_model.log_prior_hessian(model_params, layer=ll_name)
            else:
                log_prior_hessian = prob_model.log_prior_hessian(model_params)

            posterior_precision = final_ggn_sum - log_prior_hessian
            return posterior_precision

        def _diag_factorization():
            raise NotImplementedError('Diagonal factorization not implemented.')

        def _kfac_factorization():
            #####################################################################
            #                                                                   #
            #                                                                   #   
            #                                                                   #
            #                                                                   #
            #                                                                   #
            #                APPLIED DL PROJECT GOES HERE                       #    
            #                                                                   #
            #                                                                   #
            #                                                                   #
            #                                                                   #
            #                                                                   #
            #####################################################################
            raise NotImplementedError('Kronecker factorization not implemented.')

        if self.hessian_factorization == HessianFactorization.FULL:
            posterior_precision = _full_factorization()
        elif self.hessian_factorization == HessianFactorization.DIAG:
            posterior_precision = _diag_factorization()
        elif self.hessian_factorization == HessianFactorization.KFAC:
            posterior_precision = _kfac_factorization()
        else:
            raise ValueError('Hessian factorization not implemented.')

        return posterior_precision

    def get_approximate_ppd(
        self,
        batch_input: jnp.ndarray,
        model_params: ParamTree,
        posterior_precision: jnp.ndarray,
        aleatoric_var: int,
        rng_key: jax.random.key,
    ):
        """Generate the approximate PPD for a given batch of inputs.

        Parameters
        ----------
        batch_input : jnp.ndarray
            Input feature batch.
        posterior_precision : jnp.ndarray
            Posterior precision matrix. Possibly obtained from `get_posterior_precision`.
        aleatoric_var : int
            When using a closed-form approximation, this is the aleatoric variance added to
            the epistemic variance. For details, see e.g. the Laplace Redux paper by Daxberger et al. (2021).
        """

        if self.subset_of_params == SubsetOfParams.LAST_LAYER:
            ll_model_params = self._extract_last_layer(model_params)
            except_ll_params = self._extract_except_last_layer(model_params)

        def _cf_ppd():
            """Closed form approximation of the PPD, assuming Gaussian likelihood."""

            batch_preds = self._input_to_pred_fn(model_params, batch_input)

            if self.subset_of_params == SubsetOfParams.ALL:
                    batch_preds_jacobian = self._input_to_pred_jacobian(
                        model_params, batch_input
                    )
                
            elif self.subset_of_params == SubsetOfParams.LAST_LAYER:
                batch_preds_jacobian = self._input_to_pred_jacobian(
                    ll_model_params, except_ll_params, batch_input
                )

            # Flatten jacobian, gives shape (batch_size, C, D)
            batch_preds_jacobian = jnp.concatenate(
                jax.tree_util.tree_flatten(
                    jax.tree_util.tree_map(
                        lambda x: x.reshape(x.shape[0], x.shape[1], -1),
                        batch_preds_jacobian,
                    )
                )[0],
                axis=2,
            )
            # compute J^T @ posterior_precision^{-1} @ J
            # posterior_precision^{-1} = (L @ L^T)^{-1} = L^{-T} @ L^{-1}
            # solve for v = L^{-1} @ J
            L = jnp.linalg.cholesky(posterior_precision, upper=False)
            v = jax.scipy.linalg.solve_triangular(
                a=L,
                b=batch_preds_jacobian[:, 0, :].T,
                lower=True,
            )
            # compute v^T v
            batch_preds_epistemic_var = jnp.linalg.norm(v, axis=0) ** 2

            batch_preds_total_var = (
                batch_preds_epistemic_var + aleatoric_var
            )

            return batch_preds[..., 0], batch_preds_total_var

        def _mc_ppd():
            """Monte Carlo approximation of the PPD. Feasible for any likelihood."""

            N_SAMPLES: Final[int] = 100

            if self.subset_of_params == SubsetOfParams.LAST_LAYER:
                model_params_array = jax.flatten_util.ravel_pytree(
                    self._extract_last_layer(model_params)
                )[0]
            else:
                model_params_array = jax.flatten_util.ravel_pytree(model_params)[0]

            # sample from posterior
            posterior = MultivariateNormal(
                loc=model_params_array.squeeze(), precision_matrix=posterior_precision
            )

            samples = posterior.sample(
                rng_key, sample_shape=(N_SAMPLES,)
            )  # apparently 10-20 samples is standard (Daxberger, Laplace Redux paper)

            chex.assert_shape(samples, (N_SAMPLES, model_params_array.shape[0]))

            def _vec_to_pytree(vec, param_pytree):
                leafs_params, structure = jax.tree.flatten(param_pytree)
                if self.subset_of_params == SubsetOfParams.LAST_LAYER:
                    ll_param_pytree = self._extract_last_layer(param_pytree)
                    not_ll_param_pytree = self._extract_except_last_layer(param_pytree)

                leaf_vec_shapes = [p.shape for p in leafs_params]

                if self.subset_of_params == SubsetOfParams.LAST_LAYER:
                    flattened_leafs = jax.tree_flatten(not_ll_param_pytree)[0]
                    index = 0
                    for s in leaf_vec_shapes[len(flattened_leafs) :]:
                        upper_index = index + np.prod(s)
                        flattened_leafs.append(jnp.reshape(vec[index:upper_index], s))
                        index = upper_index
                else:
                    flattened_leafs = []
                    index = 0
                    for s in leaf_vec_shapes:
                        upper_index = index + np.prod(s)
                        flattened_leafs.append(np.reshape(vec[index:upper_index], s))
                        index = upper_index
                return jax.tree.unflatten(structure, flattened_leafs)

            samples_pytree = jax.vmap(
                lambda x: _vec_to_pytree(x, model_params), in_axes=0
            )(samples)

            if True:
                vectorized_apply = jax.vmap(
                    self._input_to_pred_fn, in_axes=(0, None), out_axes=0
                )
                lvals = vectorized_apply(samples_pytree, batch_input)
            else:
                # in case jax.vmap allocates too much memory,
                # should however not be the case for batched prediction.
                lvals_list = []
                params = jax.tree_map(lambda x: x[i], samples_pytree)
                lvals_list.append(self._input_to_pred_fn(params, batch_input))
                lvals = jnp.stack(lvals)

            if self.task == Task.REGRESSION:
                predictions = Normal(loc=lvals.squeeze(), scale=1.0).sample(rng_key)
            elif self.task == Task.CLASSIFICATION:
                predictions = Categorical(logits=lvals).sample(rng_key)
            else:
                raise ValueError('Learning target not implemented.')

            if self.task == Task.CLASSIFICATION:
                return jax.scipy.stats.mode(predictions)[0], lvals

            return (
                jnp.mean(predictions, axis=0).squeeze(),
                jnp.var(predictions, axis=0).squeeze(),
            )

        if self.predictive_approximation == PredictiveApproximation.CF:
            approximate_ppd = _cf_ppd()
        elif self.predictive_approximation == PredictiveApproximation.MC:
            approximate_ppd = _mc_ppd()
        else:
            raise ValueError('Predictive approximation not implemented.')

        return approximate_ppd
