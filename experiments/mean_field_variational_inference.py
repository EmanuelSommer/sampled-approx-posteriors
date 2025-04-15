"""Carry out Mean-Field Variational Inference on a UCI regression dataset."""
import os
import jax
import optax
import jax.numpy as jnp
import pandas as pd
from jax import random
from jax.flatten_util import ravel_pytree
from functools import partial

from src.abi.mfvi import forward_apply, log_prior, log_likelihood
from src.models import MLPModelUCI
from src.data import TabularLoader
from src.config import DataConfig
from src.abi.laplace import Task
from src.abi.utils import load_config_from_yaml_vi, lppd, pointwise_lppd

exp_config = load_config_from_yaml_vi("experiments/configs/mfvi_uci_benchmark.yaml")

log_prior = partial(log_prior, scale=exp_config.prior_scale)

@jax.jit
def update(var_params, opt_state, X, y, rng):
    """
    Single SGD/Adam step over the negative ELBO.
    """
    def loss_fn(vp):
        return -elbo(vp, apply_fn, X, y, rng, 10)  # negative ELBO

    grads = jax.grad(loss_fn)(var_params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_var_params = optax.apply_updates(var_params, updates)
    return new_var_params, new_opt_state

def elbo(var_params, apply_fn, X, y, rng, num_samples=10):
    """
    Estimates the ELBO via reparameterization.
    ELBO = E_q[log p(y | theta) + log p(theta) - log q(theta)].
    """
    mean, log_std = var_params["mean"], var_params["log_std"]
    std = jnp.exp(log_std)
    dim = mean.shape[0]

    def single_sample(rng_key):
        eps = random.normal(rng_key, (dim,))
        theta_sample = mean + std * eps

        # Log-likelihood
        ll = log_likelihood(unravel_fn(theta_sample), X, y, apply_fn)
        # Log-prior
        lp = log_prior(theta_sample)
        # Log q(theta)
        lq = -0.5 * jnp.sum(((theta_sample - mean) / std) ** 2) \
             - jnp.sum(log_std) \
             - 0.5 * dim * jnp.log(2.0 * jnp.pi)
        return ll + lp - lq

    rng_keys = random.split(rng, num_samples)
    samples = jax.vmap(single_sample)(rng_keys)
    return jnp.mean(samples)

if exp_config.task == Task.REGRESSION:
    model = MLPModelUCI(
        depth=exp_config.depth,
        width=exp_config.width,
        activation=exp_config.activation,
        use_bias=exp_config.use_bias,
    )

results = []
params_samples_list = []

for seed in exp_config.seeds:
    seed_dataloader = seed
    seed_param_init = seed + 1
    seed_inference = seed + 2
    seed_predictions = seed + 3

    for dataset in exp_config.dataset_list:
        print("V" * 100)
        print("=" * 100)
        print(f"MFVI on dataset: {dataset}")

        if exp_config.task == Task.REGRESSION:

            config_data = DataConfig(
                path=f"./data/{dataset}",
                source="local",
                data_type="tabular",
                task="regr",
                target_column=None,
                features=None,
                datapoint_limit=exp_config.datapoint_limit,
                normalize=True,
                train_split=(1 - exp_config.val_size - exp_config.test_size),
                valid_split=exp_config.val_size,
                test_split=exp_config.test_size,
            )

            loader = TabularLoader(
                config_data,
                rng_key=jax.random.key(seed_dataloader),
                n_chains=1,
            )

            train_loader = partial(loader.iter, split="train", batch_size=128)

            X_train, y_train = loader.data_train
            X_val, y_val = loader.data_valid
            X_test, y_test = loader.data_test

        params_init = model.init(jax.random.key(exp_config.seeds[0]), X_train)["params"]

        # Flatten/unflatten functions for the model parameters
        flat_params_init, unravel_fn = ravel_pytree(params_init)

        num_params = flat_params_init.shape[0]
        var_params = {
            "mean": jnp.zeros(num_params),
            "log_std": jnp.full((num_params,), -2.0),  # modest initial std
        }

        def apply_fn(var_dict, x):
            return model.apply(var_dict, x)

        learning_rate = 1e-2
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(var_params)

        rng = jax.random.key(seed_inference)

        for epoch in range(exp_config.epochs):
            for i, train_data in enumerate(train_loader()):
                X_train, y_train = train_data
                rng, subkey = random.split(rng)
                var_params, opt_state = update(var_params, opt_state, X_train, y_train, subkey)

            if epoch % 200 == 0:
                current_elbo = elbo(var_params, apply_fn, X_train, y_train, subkey, num_samples=50)
                print(f"Step {epoch}, ELBO: {current_elbo:.4f}")

        mean, log_std = var_params["mean"], var_params["log_std"]
        std = jnp.exp(log_std)

        params_mean = mean
        params_mean = unravel_fn(params_mean)
        test_preds_just_mean = forward_apply(params_mean, X_test, apply_fn)
        rmse_just_mean = jnp.sqrt(jnp.mean((y_test - test_preds_just_mean.squeeze()) ** 2))

        num_samples = 100
        eps = random.normal(jax.random.key(seed_predictions), (num_samples, num_params))
        theta_samples = mean + std * eps
        params_samples = jax.vmap(unravel_fn)(theta_samples)

        test_preds = jax.vmap(lambda p: forward_apply(p, X_test, apply_fn))(params_samples)
        test_preds_mean = jnp.mean(test_preds, axis=0)
        test_preds_var = jnp.var(test_preds, axis=0)
        sigma_obs = exp_config.sigma_obs
        test_preds_total_var = test_preds_var + sigma_obs**2
        mfvi_predictions_array = jnp.concatenate([test_preds_mean, test_preds_total_var], axis=1)
        rmse = jnp.sqrt(jnp.mean((y_test - test_preds_mean.squeeze()) ** 2))
        print(f"Test RMSE (averaged over {num_samples} posterior samples): {rmse:.4f}")
        result = {
            "dataset": dataset,
            "seed": seed,
            "lppd": lppd(
                pointwise_lppd(mfvi_predictions_array, y_test, Task.REGRESSION)
            ),
            "rmse": jnp.sqrt(jnp.mean((y_test - test_preds_mean.squeeze()) ** 2)),
            "rmse_just_mean": rmse_just_mean,
            }
        print(f"LPPD: {result['lppd']}, RMSE: {result['rmse']}, RMSE (just mean): {result['rmse_just_mean']}")
        results.append(result)
        params_samples_list.append(params_samples)

results_df = pd.DataFrame(results)

exp_path = f"results/{exp_config.experiment_name}"

if not os.path.exists(exp_path):
    os.mkdir(exp_path)

results_df.to_pickle(
    f"{exp_path}/mfvi_results_{exp_config.task.value}.pkl"
)

# Compute mean and std grouped by dataset
grouped = results_df.groupby("dataset").agg(
    lppd_mean=("lppd", "mean"),
    lppd_std=("lppd", "std"),
    rmse_mean=("rmse", "mean"),
    rmse_std=("rmse", "std"),
)

# Combine mean and std into "mean ± std" format
grouped["lppd"] = grouped["lppd_mean"].round(3).astype(str) + " ± " + grouped["lppd_std"].round(3).astype(str)
grouped["rmse"] = grouped["rmse_mean"].round(3).astype(str) + " ± " + grouped["rmse_std"].round(3).astype(str)

# Select only the formatted columns
result_agg = grouped[["lppd", "rmse"]].reset_index()

result_agg.to_csv(
    f"{exp_path}/mfvi_results_{exp_config.task.value}_agg.csv",
    index=False,
)
