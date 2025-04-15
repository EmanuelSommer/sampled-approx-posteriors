"""Fit a Laplace approximation on a UCI regression dataset."""
import os
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import yaml
from functools import partial
from flax.training.train_state import TrainState

from src.data import TabularLoader, ImageLoader
from src.abi.laplace import (
    HessianFactorization,
    LaplaceApproximation,
    PredictiveApproximation,
    Task,
    SubsetOfParams,
)
from src.config import DataConfig, LeNetConfig
from src.flax2bnn import ProbModelBuilder
from src.models import MLPModelUCI, LeNet

from src.abi.utils import (
    constant_variance_nll,
    categorical_nll,
    pointwise_lppd,
    lppd,
    load_config_from_yaml,
    train_step_mlp,
    val_step_mlp,
    task_constructor,
    subset_of_params_constructor,
    hessian_factorization_constructor,
    predictive_approximation_constructor,
)

from tqdm import tqdm

yaml.add_constructor("!Task", task_constructor)
yaml.add_constructor("!SubParams", subset_of_params_constructor)
yaml.add_constructor("!Fac", hessian_factorization_constructor)
yaml.add_constructor("!PredApp", predictive_approximation_constructor)

exp_config = load_config_from_yaml("experiments/configs/laplace_uci_benchmark.yaml")

if exp_config.task == Task.REGRESSION:
    model = MLPModelUCI(
        depth=exp_config.depth,
        width=exp_config.width,
        activation=exp_config.activation
    )
else:
    raise ValueError("Only regression is supported.")

results = []

for seed in exp_config.seeds:
    seed_dataloader = seed
    seed_param_init = seed + 1
    seed_inference = seed + 2
    seed_predictions = seed + 3

    rng_iter = jax.random.key(seed)

    for dataset in exp_config.dataset_list:
        print("V" * 100)
        print("=" * 100)
        print(f"Laplace on dataset: {dataset}")

        if exp_config.task == Task.REGRESSION:

            config_data = DataConfig(
                path=f"./data/{dataset}",
                source="local",
                data_type="tabular",
                task="regr",
                target_column=None,
                features=None,
                datapoint_limit=None,
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

            X_train, y_train = loader.data_train
            X_val, y_val = loader.data_valid
            X_test, y_test = loader.data_test

        # MAP estimate
        params = model.init(rng_iter, X_train)
        optimizer = getattr(optax, exp_config.optim)(exp_config.lr)
        model_train_state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
        )

        print("=" * 100)
        print(f"Training the MAP estimate.")

        lowest_loss = float("inf")
        lowest_val_loss = float("inf")

        
        for K in range(exp_config.epochs):
            rng_iter, subkey = jax.random.split(rng_iter)
            model_train_state, loss = train_step_mlp(
                model_train_state, X_train, y_train
            )
            val_loss = val_step_mlp(
                model_train_state, X_val, y_val
            )
            if K % 1_000 == 0:
                print(f"Loss at epoch {K}: {loss}; val loss: {val_loss}.")

            if exp_config.val_loss_stopping:
                if val_loss < lowest_val_loss:
                    lowest_loss = loss
                    trained_model_params = model_train_state.params
            else:
                if loss < lowest_loss:
                    lowest_loss = loss
                    trained_model_params = model_train_state.params

       
        prior_config = {
            "scheme": "equal",
            "details": {"dist": "Normal", "sd": 1},
        }
        
        pmb = ProbModelBuilder(
            model,
            prior_config=prior_config,
            params=trained_model_params["params"],
            seed=seed,
        )

        
        la = LaplaceApproximation(
            task=Task.REGRESSION,
            subset_of_params=exp_config.subset_of_params,
            hessian_factorization=exp_config.hessian_factorization,
            predictive_approximation=exp_config.predictive_approximation,
            _input_to_pred_fn=model.apply,
            _pred_to_loss_fn=partial(constant_variance_nll, scale=1.0),
        )

        posterior_precision = la.get_posterior_precision(
            full_batch_train_features=X_train,
            full_batch_train_labels=y_train,
            model_params=trained_model_params,
            prob_model=pmb,
            batch_size=64,
        )

        # Maximum likelihood estimate of the aleatoric variance on validation set
        if exp_config.aleatoric_var == "mle":
            aleatoric_var = jnp.var(
                y_val - model.apply(trained_model_params, X_val)
            )
        else:
            aleatoric_var = float(exp_config.aleatoric_var)

        la_predictions = la.get_approximate_ppd(
            batch_input=X_test,
            model_params=trained_model_params,
            posterior_precision=posterior_precision,
            aleatoric_var=aleatoric_var,
            rng_key=jax.random.key(0),
        )

        la_predictions_array = jnp.vstack([la_predictions[0], la_predictions[1]]).T

        
        result = {
            "dataset": dataset,
            "seed": seed,
            "lppd": lppd(
            pointwise_lppd(la_predictions_array, y_test, Task.REGRESSION)
            ),
            "rmse": jnp.sqrt(jnp.mean((la_predictions[0] - y_test) ** 2)),
        }

        print("=" * 100)
        print(f'RMSE: {result["rmse"]}')
        print(f'LPPD: {result["lppd"]}')
        print("=" * 100)

    results.append(result)

results_df = pd.DataFrame(results)

if not os.path.exists(f"results/{exp_config.experiment_name}"):
    os.mkdir(f"results/{exp_config.experiment_name}")

results_df.to_pickle(
    f"results/{exp_config.experiment_name}/laplace_results_{exp_config.task.value}.pkl"
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
    f"results/{exp_config.experiment_name}/laplace_results_{exp_config.task.value}_agg.csv",
    index=False,
)