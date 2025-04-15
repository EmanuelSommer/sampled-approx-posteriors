"""Toolbox for Handling a (flax) BNN with Blackjax."""

import logging
import pickle
from functools import partial
from pathlib import Path
from typing import Optional

import jax
import jax.experimental
import jax.numpy as jnp
import optax
from blackjax import nuts
from dataserious.base import JsonSerializableDict
from optax._src.base import GradientTransformation
from tqdm import tqdm

from src.sai.config.sampler import GetSampler, SamplerConfig, Scheduler
from src.sai.dataset.base import BaseLoader
from src.sai.kernels.base import Sampler
from src.sai.kernels.warmup import custom_mclmc_warmup, custom_window_adaptation
from src.sai.training.callbacks import (
    progress_bar_scan,
    save_position,
)
from src.sai.types import (
    GradEstimator,
    Kernel,
    ParamTree,
    PosteriorFunction,
    PRNGKey,
    WarmupResult,
)

logger = logging.getLogger(__name__)


def inference_loop(
    unnorm_log_posterior: PosteriorFunction,
    config: SamplerConfig,
    rng_key: jax.Array,
    init_params: ParamTree,
    step_ids: jax.Array,
    saving_path: Path,
    saving_path_warmup: Optional[Path] = None,
):
    """Blackjax inference loop for full-batch sampling.

    Args:
        unnorm_log_posterior: PosteriorFunction
        config: Sampler configuration.
        rng_key: Random chainwise key.
        init_params: Initial parameters to start the sampling from.
        step_ids: Step ids of the chain to be sampled.
        saving_path: Path to save the sampling samples.
        saving_path_warmup: Path to save the warmup samples, by default None.

    Note:
        - Currently only supports nuts & mclmc kernel.
    """
    info: JsonSerializableDict = {}  # Put any information you might need later for analysis
    n_devices = len(step_ids)
    assert config.warmup_steps > 0, "Number of warmup steps must be greater than 0."

    keys = jax.vmap(jax.random.split)(rng_key)

    # Warmup
    logger.info(f"> Starting {config.name.value} Warmup Sampling...")
    match config.name:
        case GetSampler.NUTS:
            warmup_state, parameters = warmup_nuts(
                kernel=nuts,  # config.kernel,
                config=config,
                rng_key=keys[..., 0],
                init_params=init_params,
                step_ids=step_ids,
                unnorm_log_posterior=unnorm_log_posterior,
                n_devices=n_devices,
                saving_path=saving_path_warmup,
            )
        case GetSampler.MCLMC:
            warmup_state, parameters = warmup_mclmc(
                config=config,
                rng_key=keys[..., 0],
                init_params=init_params,
                unnorm_log_posterior=unnorm_log_posterior,
            )
            # Save the warmup parameters in a .txt file
            if not saving_path.exists():
                saving_path.mkdir(parents=True)
            with open((saving_path.parent / "warmup_params.txt"), "w") as f:
                if parameters["step_size"].shape == ():
                    parameters["step_size"] = jnp.array([parameters["step_size"]])
                    parameters["L"] = jnp.array([parameters["L"]])
                f.write(",".join([str(sts) for sts in parameters["step_size"]]) + "\n")
                f.write(",".join([str(sts) for sts in parameters["L"]]) + "\n")
        case _:
            raise NotImplementedError(
                f"{config.name} does not have warmup implemented."
            )

    warmup_state = jax.block_until_ready(warmup_state)

    logger.info(f"> {config.name.value} Warmup sampling completed successfully.")

    # Sampling with tuned parameters
    if config.n_thinning == 1:

        def _inference_loop(
            rng_key: PRNGKey,
            state: Sampler.State,
            parameters: ParamTree,
            step_id: jax.Array,
        ):
            def one_step(state: Sampler.State, xs: tuple[jax.Array, jax.Array]):
                idx, rng_key = xs
                state, info = sampler.step(rng_key, state)
                # dump y to disk
                jax.experimental.io_callback(
                    partial(save_position, base=saving_path),
                    result_shape_dtypes=state.position,
                    position=state.position,
                    idx=step_id,
                    n=idx,
                )
                return state, info

            one_step_ = jax.jit(
                progress_bar_scan(
                    n_steps=config.n_samples * n_devices,
                    name=f"{config.name.value} Sampling",
                )(one_step)
            )

            sampler = config.kernel(logdensity_fn=unnorm_log_posterior, **parameters)
            keys = jax.random.split(rng_key, config.n_samples)
            _, infos = jax.lax.scan(
                f=one_step_, init=state, xs=(jnp.arange(config.n_samples), keys)
            )
            return infos

    elif config.n_thinning > 1:
        # only save every n_thinning samples and thus do not scan but loop
        def _inference_loop(
            rng_key: PRNGKey,
            state: Sampler.State,
            parameters: ParamTree,
            step_id: jax.Array,
        ):
            def one_step(state: Sampler.State, xs: tuple[jax.Array, jax.Array]):
                idx, rng_key = xs
                state, info = sampler.step(rng_key, state)

                def save_if_thinned():
                    jax.experimental.io_callback(
                        partial(save_position, base=saving_path),
                        result_shape_dtypes=state.position,
                        position=state.position,
                        idx=step_id,
                        n=idx,
                    )
                    return None

                jax.lax.cond(
                    idx % config.n_thinning == 0, save_if_thinned, lambda: None
                )
                return state, info

            one_step_ = jax.jit(
                progress_bar_scan(
                    n_steps=config.n_samples * n_devices,
                    name=f"{config.name.value} Sampling",
                )(one_step)
            )

            sampler = config.kernel(logdensity_fn=unnorm_log_posterior, **parameters)
            keys = jax.random.split(rng_key, config.n_samples)
            _, infos = jax.lax.scan(
                f=one_step_, init=state, xs=(jnp.arange(config.n_samples), keys)
            )
            return infos

    # Run the sampling loop
    logger.info(f"> Starting {config.name.value} Sampling...")
    runner_info = jax.pmap(_inference_loop)(
        keys[..., 1], warmup_state, parameters, step_ids
    )

    # Explicitly wait for the computation to finish, doesnt matter if
    # we need runner_info or not.
    runner_info = jax.block_until_ready(runner_info)

    logger.info(f"> {config.name.value} Sampling completed successfully.")

    match config.name:
        case GetSampler.NUTS:
            info.update(
                {
                    "num_integration_steps": runner_info.num_integration_steps,
                    "acceptance_rate": runner_info.acceptance_rate,
                    "num_trajectory_expansions": runner_info.num_trajectory_expansions,
                    "is_divergent": runner_info.is_divergent,
                    "energy": runner_info.energy,
                    "is_turning": runner_info.is_turning,
                }
            )

    # Save information to disk
    if not saving_path.exists():
        saving_path.mkdir(parents=True)

    # Dump Information
    with open(saving_path / "info.pkl", "wb") as f:  # type: ignore[assignment]
        pickle.dump(info, f)  # type: ignore[arg-type]


def warmup_nuts(
    kernel: Kernel,
    config: SamplerConfig,
    rng_key: jax.Array,  # chainwise key!
    init_params: ParamTree,
    step_ids: jax.Array,
    unnorm_log_posterior: PosteriorFunction,
    n_devices: int,
    saving_path: Optional[Path] = None,
) -> WarmupResult:
    """Perform warmup for NUTS."""
    warmup_algo = custom_window_adaptation(
        algorithm=kernel,
        logdensity_fn=unnorm_log_posterior,
        progress_bar=True,
        saving_path=saving_path,
    )
    warmup_state, parameters = jax.pmap(
        fun=warmup_algo.run,
        in_axes=(0, 0, 0, None, None),
        static_broadcasted_argnums=(3, 4),
    )(
        rng_key,
        init_params,
        step_ids,
        config.warmup_steps,
        n_devices,
    )
    return warmup_state, parameters


def warmup_mclmc(
    config: SamplerConfig,
    rng_key: jax.Array,  # chainwise key!
    init_params: ParamTree,
    unnorm_log_posterior: PosteriorFunction,
) -> WarmupResult:
    """Perform warmup for MCLMC."""
    warmup_algo = custom_mclmc_warmup(
        logdensity_fn=unnorm_log_posterior,
        diagonal_preconditioning=config.diagonal_preconditioning,
        desired_energy_var_start=config.desired_energy_var_start,
        desired_energy_var_end=config.desired_energy_var_end,
        trust_in_estimate=config.trust_in_estimate,
        num_effective_samples=config.num_effective_samples,
        step_size_init=config.step_size_init,
    )
    warmup_state, parameters = jax.pmap(
        fun=warmup_algo.run,
        in_axes=(0, 0, None),
        static_broadcasted_argnums=2,
    )(
        rng_key,
        init_params,
        config.warmup_steps,
    )
    parameters = {"step_size": parameters.step_size, "L": parameters.L}
    return warmup_state, parameters


@partial(jax.pmap, in_axes=(0, 0, None, None, 0, None))
def one_sgd_step(
    state: Sampler.State,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    step_size: float,
    func: GradEstimator,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> tuple[Sampler.State, optax.OptState]:
    """Single SG step."""
    grads = func(state.position, batch[0], batch[1])
    grads_scaled = jax.tree.map(lambda x: -1.0 * step_size * x, grads)
    updates, new_opt_state = optimizer.update(grads_scaled, opt_state)
    state.position = optax.apply_updates(state.position, updates)
    return state, new_opt_state
