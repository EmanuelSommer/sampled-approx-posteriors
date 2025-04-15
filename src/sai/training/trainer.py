"""Trainer module for training BDEs."""

import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax.tree_util import PyTreeDef

import src.sai.inference.metrics as bm_metrics
import src.sai.training.utils as train_utils
from src.sai.bnns.logliks import ProbabilisticModel
from src.sai.config.core import Config
from src.sai.config.data import DatasetType, Task
from src.sai.dataset.base import BaseLoader
from src.sai.dataset.tabular import TabularLoader
from src.sai.inference.evaluation import EvaluationName, Evaluator
from src.sai.inference.metrics import (
    MetricsStore,
    RegressionMetrics,
)
from src.sai.inference.predict import sample_from_predictions
from src.sai.inference.sample_loader import SampleLoader
from src.sai.training.sampling import inference_loop
from src.sai.training.utils import earlystop, get_nn_size
from src.sai.types import ParamTree, PRNGKey
from src.sai.utils import measure_time, pretty_string_dict

logger = logging.getLogger(__name__)


class TrainState(train_state.TrainState):
    """Extended TrainState class for BDE training."""

    batch_stats: Any


class BDETrainer:
    """Trainer class for training BDEs.

    Notes:
        Training is started by calling the `.train_bde()` method of the class instance.

        - Steps:
            1. Warmstart Phase: Train the Deep Ensemble Members: `.train_warmstart()`.
            2. Sampling Phase: Perform the sampling using MCMC: `.start_sampling()`.
            3. Evaluation Phase: TODO
    """

    def __init__(self, config: Config, folder: Optional[Path] = None):
        """Initialize the trainer.

        Args:
            config: A configuration class instance.
            folder: Reuse folder of experiment.
        """
        assert isinstance(config, Config)
        self.config = config
        self.task = self.config.data.task
        self.n_devices = jax.device_count()
        self.n_chains = self.config.n_chains
        self.metrics_warmstart = MetricsStore.empty()
        self._key = jax.random.key(self.config.rng)
        self._key_de, self._key_sample, self._key_eval = jax.random.split(self.key, 3)
        self._key_chains = jax.random.split(self.key, self.n_chains)

        # Setup directory
        logger.info("> Setting up directories...")
        self.config.setup_dir(folder)
        self.exp_dir = self.config.experiment_dir

        # Setup chain layout
        if self.n_chains % self.n_devices == 0:
            self.train_plan = jnp.array_split(
                jnp.arange(self.n_chains), self.n_chains / self.n_devices
            )
        else:
            raise ValueError(
                "n_chains must be divisible by the number of devices."
                f"{self.n_chains} % {self.n_devices} != 0."
            )

        # Setup DataLoader
        logger.info("> Setting up DataLoader...")

        self.loader: BaseLoader
        match self.config.data.data_type:
            case DatasetType.TABULAR:
                self.loader = TabularLoader(
                    config=self.config.data, rng_key=self.key, n_chains=self.n_chains
                )
            case _:
                raise NotImplementedError(
                    f"Data Type {self.config.data.data_type} not implemented."
                )

        # Setup Probabilistic Model
        logger.info("> Setting up Probabilistic Model...")
        self.module = self.config.get_flax_model()
        self.batch_size_s = self.config_sampler.batch_size or len(
            self.loader.data_train[0]
        )

        if self.config_sampler.batch_size is None:
            self.batch_size_s_test = len(self.loader.data_test[0])
        else:
            self.batch_size_s_test = min(
                self.config_sampler.batch_size, len(self.loader.data_test[0])
            )

        self.batch_size_w = self.config_warmstart.batch_size or len(
            self.loader.data_train[0]
        )
        self.n_batches = (len(self.loader.data_train[0]) // self.batch_size_s) or 1
        self.prob_model = ProbabilisticModel(
            module=self.module,
            prior=self.config.training.prior,
            n_batches=self.n_batches,
            task=self.config.data.task,
        )

        logger.info(f"> Trainer has been successfully initialized\n{self}")

    def __str__(self):
        """Return the string representation of the trainer."""
        return (
            f"{'-'*50}\n\t{self.__class__.__name__}:\n"
            f" | Experiment: {self.config.experiment_name}\n"
            f" | Chains: {self.n_chains}\n"
            f" | Devices: {self.n_devices}\n"
            f" | Warmstart: {self.has_warmstart}\n"
            f" | Module: {self.module.__class__.__name__}\n"
            f"{'-'*50}\n\tData Loader:\n | {self.loader}\n"
            f"{'-'*50}\n\t{self.prob_model}\n"
            f"{'-'*50}\n\tSampler Configuration:\n"
            f"{pretty_string_dict(self.config_sampler.to_dict())}\n"
        )

    def get_single_input(
        self, batch_size: int = 1, chains: jax.Array = jnp.array((0,))
    ) -> jax.Array:
        """Return a random input of shape (batch_size, ...)."""
        gen = self.loader.iter(split="train", batch_size=batch_size, chains=chains)
        try:
            return next(gen)[0]  # NOTE: Can be turned into an argument later if needed.
        finally:
            gen.close()

    def get_pytree_def(self) -> tuple[PyTreeDef, PyTreeDef]:
        """Return the Pytree definition of the model."""
        return jax.tree_util.tree_structure(
            tree=self.init_module_params(chains=jnp.array((0,)))[0]
        ), jax.tree_util.tree_structure(
            tree=self.init_module_params(chains=jnp.array((0,)))[1]
        )

    def init_module_params(
        self, chains: Optional[jax.Array] = None, from_prior: bool = True
    ) -> tuple[ParamTree, Optional[ParamTree]]:
        """Initialize the parameters for the module.

        Args:
            chains: Chains for which to initialize.
            from_prior: Whether to use custom initialization from prior.

        Returns:
            Initialized parameters, with leaf nodes of shape (N_DEVICE, ...).
        """
        variables = jax.vmap(self.prob_model.init if from_prior else self.module.init)(
            self.chainwise_key(chains=chains),
            self.get_single_input(chains=chains),
            train=jnp.repeat(
                False, len(chains) if chains is not None else self.n_chains
            ),
        )
        return variables["params"], (
            variables["batch_stats"] if "batch_stats" in variables else None
        )

    def init_training_state(self, chains: jax.Array) -> TrainState:
        """Initialize the training state for n_devices."""
        return jax.vmap(get_initial_state, in_axes=(0, 0, None, None))(
            self.chainwise_key(chains=chains),
            self.get_single_input(chains=chains),
            self.module,
            self.optimizer,
        )

    @property
    def key(self) -> PRNGKey:
        """Maintain single RNG key."""
        self._key, key = jax.random.split(self._key)
        return key

    def chainwise_key(self, chains: Optional[jax.Array] = None) -> jax.Array:
        """Handle the RNG state for all chains."""
        if chains is None:
            chains = jnp.arange(self.n_chains)

        keys = jax.vmap(jax.random.split)(self._key_chains[chains])
        self._key_chains = self._key_chains.at[chains].set(keys[..., 0])
        return keys[..., 1]

    def reset_rng_state(self, rng_key: PRNGKey):
        """Reset RNG states using the key given."""
        self._key = rng_key
        self._key_chains = jax.random.split(self.key, self.n_chains)
        self.loader._key = self.key
        self.loader._key_chains = jax.random.split(self.key, self.n_chains)

    @property
    def config_warmstart(self):
        """Return the warmstart configuration."""
        return self.config.training.warmstart

    @property
    def config_sampler(self):
        """Return the sampler configuration."""
        return self.config.training.sampler

    @property
    def has_warmstart(self):
        """Check if warmstart is enabled."""
        return self.config.training.has_warmstart

    @property
    def optimizer(self):
        """Return the optimizer."""
        return self.config.training.optimizer

    @property
    def tree_path(self):
        """Return the tree path."""
        filename = "tree"
        e_dir = self.exp_dir / filename
        if not e_dir.exists():
            raise FileNotFoundError(f"Tree not found at {e_dir}")
        return e_dir

    @property
    def tree_path_batch_stats(self):
        """Return the tree path for batch stats."""
        filename = "batch_stats_tree"
        e_dir = self.exp_dir / filename
        if not e_dir.exists():
            raise FileNotFoundError(f"Tree not found at {e_dir}")
        return e_dir

    @property
    def sampling_tree_path(self):
        """Return the sampling tree path."""
        file_name = "tree_sampling"
        e_dir = self.exp_dir / file_name
        if not e_dir.exists():
            raise FileNotFoundError(f"Tree not found at {e_dir}")
        return e_dir

    @property
    def _sampling_warmup_dir(self):
        if self.config_sampler.keep_warmup:
            return self.exp_dir / self.config_sampler._warmup_dir_name
        else:
            return None

    @measure_time("time.warmstart")
    def train_warmstart(self):
        """Start Warmstart phase of BDE (Deep Ensemble Training before sampling)."""
        logger.info("Preparing chain initialization...")
        self.reset_rng_state(rng_key=self._key_de)
        cfg_warm = self.config_warmstart
        if self.has_warmstart:  # Warmstart
            if cfg_warm.warmstart_exp_dir:  # With checkpoint
                logger.info("Loading chain initialization from disk.")
                warm_path = Path(cfg_warm.warmstart_exp_dir) / cfg_warm._dir_name
                deep_ensemble = [
                    i for i in os.listdir(warm_path) if i.startswith("params")
                ]
                n_nns = len(deep_ensemble)
                if n_nns == 0:
                    raise ValueError(f"No initialization found at {warm_path}")
                elif n_nns < self.n_chains:
                    raise ValueError(
                        f"Not enough initial values found at {warm_path}: "
                        f"{n_nns} < {self.n_chains}."
                    )
                elif n_nns > self.n_chains:
                    logger.warning(
                        f"Using first {self.n_chains} from {n_nns} initial values."
                    )
                # copy 'tree' file to the new directory as it is important in the inference
                tree = train_utils.load_tree(path=warm_path.parent / "tree")
                train_utils.save_tree(path=self.exp_dir / "tree", tree=tree)
            else:  # Without checkpoint
                logger.info("Training deep ensemble...")
                warm_path = self.exp_dir / cfg_warm._dir_name

                metrics_list: list[MetricsStore] = []
                for step in self.train_plan:
                    state = self.init_training_state(chains=step)
                    logger.info(f"\t| Starting training neural networks {step}...")
                    state, metrics = self.train_de_member(state=state, chains=step)
                    logger.info(f"\t| Training completed for neural networks {step}.")
                    metrics_list.append(metrics)

                    # Save Checkpoints of Deep Ensemble Members
                    for i, chain_n in enumerate(step):
                        if len(step) > 1:
                            params = jax.tree.map(lambda x: x[i], state.params)
                            sigma = jax.tree.map(
                                lambda x: x[i],
                                getattr(state.opt_state[0], "sigma", None),
                            )
                        else:
                            params = state.params
                            sigma = getattr(state.opt_state[0], "sigma", None)

                        # Save parameters (as before)
                        train_utils.save_params(
                            warm_path,
                            params,
                            chain_n,
                        )

                        # Save uncertainties with a different prefix
                        if sigma is not None:
                            train_utils.save_params(
                                warm_path,
                                sigma,
                                chain_n,
                                prefix="sigma_",
                            )
                        logger.info(
                            f"\t| Neural networks {chain_n} saved at {warm_path}."
                        )
                        if state.batch_stats is not None:
                            train_utils.save_params(
                                warm_path,
                                jax.tree.map(lambda x: x[i], state.batch_stats),
                                chain_n,
                                prefix="batch_stats_",
                            )
                # NOTE: save metrics once all chains are trained (might change in future)
                self.metrics_warmstart = MetricsStore.vstack(metrics_list)
                logger.info(f"\t| Saving Warmstart Metrics at {warm_path}")
                self.metrics_warmstart.save(
                    path=warm_path / self.config_warmstart._metrics_fname
                )
                if cfg_warm.permutation_warmstart:
                    logger.info("Permuting warmstart parameters.")
                    permuted_params = train_utils.permute_warmstart(
                        warm_path, n_chains=self.n_chains, base_param=9, key=self.key
                    )
                    for i in range(self.n_chains):
                        train_utils.save_params(
                            warm_path,
                            jax.tree.map(lambda x: x[i], permuted_params),
                            i,
                            prefix="permuted",
                        )
                logger.info(f"\t| Network params {step} saved at {warm_path}.")
                x_test = self.get_single_input()
                # base_param must be the last warmstart such that params variable is assigned appropriately
                # self.module.apply({"params": jax.tree.map(lambda x: x[0, 0], permuted_params)}, x_test)
                # self.module.apply({"params": jax.tree.map(lambda x: x[0], params)}, x_test)
        else:
            logger.info("Initializing from prior.")
            train_utils.save_tree(self.exp_dir / "tree", self.get_pytree_def())

    @measure_time("time.sampling")
    def start_sampling(self):
        """Start Sampling Phase of BDE (MCMC Sampling either Full- or Mini-Batch)."""
        logger.info("Starting sampling...")
        self.reset_rng_state(rng_key=self._key_sample)
        if self.has_warmstart:
            if warm_exp := self.config_warmstart.warmstart_exp_dir:
                logger.info(
                    f'\t| Using Warmstart from experiment: {warm_exp.split("/")[-1]}'
                )
            # We check whether we take warmstart from other exp or from current.
            warm_exp = warm_exp or self.exp_dir.__str__()
            warm_path = Path(warm_exp) / self.config_warmstart._dir_name

            files = [
                file for file in os.listdir(warm_path) if file.startswith("params")
            ]
            if self.config_warmstart.permutation_warmstart:
                files = [
                    file
                    for file in os.listdir(warm_path)
                    if file.startswith("permutedparams")
                ]
            batch_stats_files = [
                file for file in os.listdir(warm_path) if file.startswith("batch_stats")
            ]
            # sort before train_plan loop!
            files = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
            chains = [warm_path / file for file in files]
            chains_batch_stats = [warm_path / file for file in batch_stats_files]
        else:
            chains = []

        for step in self.train_plan:
            logger.info(f"\t| Starting Sampling for chains {step}")
            batch_stats = None
            if chains:  # Start from deep ensemble
                params = train_utils.load_params_batch(
                    paths=[chains[i] for i in step], tree_path=self.tree_path
                )
                if chains_batch_stats:
                    batch_stats = train_utils.load_params_batch(
                        paths=[chains_batch_stats[i] for i in step],
                        tree_path=self.tree_path_batch_stats,
                    )
            else:  # Start from prior
                params, batch_stats = self.init_module_params(
                    chains=step, from_prior=True
                )
                warm_path = self.exp_dir / self.config_warmstart._dir_name
                for i, chain_n in enumerate(step):
                    train_utils.save_params(
                        warm_path,
                        jax.tree.map(lambda x: x[i], params),
                        chain_n,
                    )
                    if batch_stats is not None:
                        train_utils.save_params(
                            warm_path,
                            jax.tree.map(lambda x: x[i], batch_stats),
                            chain_n,
                            prefix="batch_stats_",
                        )
                logger.info(
                    f"\t| Initialization for neural networks {step} saved at {warm_path}."
                )

            if self.task == Task.MEAN_REGRESSION:
                # initialize the sigma to 0.0 for mean regression (lognormal)
                # replicate for each chain
                params = {
                    "params": params,
                    "sigma": jnp.zeros(jax.tree_util.tree_leaves(params)[0].shape[0]),
                }
            if batch_stats is not None and self.task != Task.MEAN_REGRESSION:
                params = {"params": params, "batch_stats": batch_stats}
            elif batch_stats is not None and self.task == Task.MEAN_REGRESSION:
                params["batch_stats"] = batch_stats

            # save the sampling tree
            _, tree_def = jax.tree.flatten(params)
            if not (self.exp_dir / "tree_sampling").exists():
                train_utils.save_tree(self.exp_dir / "tree_sampling", tree=tree_def)

            if self.prob_model.minibatch:  # Mini-Batch Sampling
                log_post = partial(self.prob_model.log_unnormalized_posterior)
                inference_loop_batch(
                    grad_estimator=jax.grad(log_post, allow_int=True),
                    config=self.config_sampler,
                    rng_key=self.chainwise_key(step),
                    init_params=params,
                    loader=self.loader,
                    step_ids=step,
                    saving_path=self.exp_dir / self.config_sampler._dir_name,
                    saving_path_warmup=self._sampling_warmup_dir,
                )
            else:  # Full Batch Sampling
                log_post = partial(
                    self.prob_model.log_unnormalized_posterior,
                    x=self.loader.data_train[0],
                    y=self.loader.data_train[1],
                )
                inference_loop(
                    unnorm_log_posterior=log_post,
                    config=self.config_sampler,
                    rng_key=self.chainwise_key(step),
                    init_params=params,
                    step_ids=step,
                    saving_path=self.exp_dir / self.config_sampler._dir_name,
                    saving_path_warmup=self._sampling_warmup_dir,
                )

        # force all jax pmap processes to be completed before exiting this function!
        # Implicit synchronization via a dummy operation
        dummy_sync = jnp.zeros(len(step))
        dummy_res = jax.pmap(lambda x: x + 1)(dummy_sync)
        dummy_res = jax.block_until_ready(dummy_res)
        logger.info("BDE training completed successfully.")
        logger.info(f"Single NN size: {get_nn_size(params)}")

    @measure_time("time.evaluation")
    def evaluate(self):
        """Evaluate the model on the given data."""
        self.reset_rng_state(rng_key=self._key_eval)

        # Set up sample loader
        sample_loader = SampleLoader(
            root_dir=self.exp_dir,
            config=self.config,
            sample_warmup_dir=self._sampling_warmup_dir,
        )

        # Set up evaluators
        postpred_names = ["Lppd", "PredPerf", "Calibration", "PredSaver"]
        if (
            any(evaluator in postpred_names for evaluator in self.config.evaluations)
            and "Predict" not in self.config.evaluations
        ):
            self.config.evaluations += ["Predict"]
        evaluators = {
            name: Evaluator.from_name(
                name=name,
                task=self.task,
                path=self.exp_dir,
            )
            for name in self.config.evaluations
        }

        # Default phase
        try:
            phases = self.config.evaluation_args["phases"]
        except KeyError:
            phases = ["sampling"]

        # Run evaluations, that don't need predictions
        phases_pre_pred = phases.copy()
        try:
            phases_pre_pred.remove("ensemble_initialization")
        except ValueError:
            pass
        if EvaluationName.CHAIN_VAR in self.config.evaluations:
            for phase in phases_pre_pred:
                evaluators[EvaluationName.CHAIN_VAR].evaluate_and_save(
                    phase=phase,
                    sample_loader=sample_loader,
                )
        if EvaluationName.CRHAT in self.config.evaluations:
            for phase in phases_pre_pred:
                evaluators[EvaluationName.CRHAT].evaluate_and_save(
                    phase=phase,
                    train_plan=self.train_plan,
                    sample_loader=sample_loader,
                )
        if EvaluationName.ESS in self.config.evaluations:
            for phase in phases_pre_pred:
                evaluators[EvaluationName.ESS].evaluate_and_save(
                    phase=phase,
                    train_plan=self.train_plan,
                    sample_loader=sample_loader,
                )
        if EvaluationName.PRIOR_LOGLIK in self.config.evaluations:
            for phase in phases_pre_pred:
                evaluators[EvaluationName.PRIOR_LOGLIK].evaluate_and_save(
                    phase=phase,
                    train_plan=self.train_plan,
                    sample_loader=sample_loader,
                    sample_batch_size=self.config.evaluation_args["samples_batch_size"],
                    data_loader=self.loader,
                    data_batch_size=self.batch_size_s_test,
                    prob_model=self.prob_model,
                )

        # Make predictions
        if EvaluationName.PREDICT in self.config.evaluations:
            # Default value (TODO)
            try:
                datapoint_limit_pred = self.config.evaluation_args[
                    "datapoint_limit_pred"
                ]
            except KeyError:
                datapoint_limit_pred = None

            pred_start_time = time.time()

            # TODO: document this costly option that evtl. allows for better calibration
            predict_sampling_factor = self.config.evaluation_args.get(
                "pred_sampling_factor", 1
            )

            pred_dist = {
                phase: evaluators[EvaluationName.PREDICT].evaluate(
                    phase=phase,
                    train_plan=self.train_plan,
                    sample_loader=sample_loader,
                    sample_batch_size=self.config.evaluation_args["samples_batch_size"],
                    data_loader=self.loader,
                    data_batch_size=self.batch_size_s_test,
                    datapoint_limit=datapoint_limit_pred,
                    model=self.module,
                )
                for phase in phases
            }
            pred = {
                phase: sample_from_predictions(
                    predictions=pred_dist[phase],
                    task=self.task,
                    rng_key=self.key,
                    sampling_factor=(
                        predict_sampling_factor
                        * self.config.training.sampler.n_samples
                        // self.config.training.sampler.n_thinning
                        if phase == "ensemble_initialization"
                        else predict_sampling_factor
                    ),
                )
                for phase in phases
            }
            logger.info(
                f"Predictions completed in {time.time() - pred_start_time:.2f}s"
            )

        # Run evaluations, that need predictions
        if datapoint_limit_pred is None:
            max_len = len(self.loader.data_test[1]) // self.batch_size_s_test
        else:
            max_len = datapoint_limit_pred // self.batch_size_s_test

        if EvaluationName.CALIBRATION in self.config.evaluations:
            for phase in phases:
                evaluators[EvaluationName.CALIBRATION].evaluate_and_save(
                    phase=phase,
                    pred=pred[phase],
                    pred_dist=pred_dist[phase],
                    target=self.loader.data_test[1][: max_len * self.batch_size_s_test],
                    nominal_coverages=self.config.evaluation_args["nominal_coverages"],
                    kappa=self.config.evaluation_args["kappa"],
                )
        if EvaluationName.LPPD in self.config.evaluations:
            for phase in phases:
                evaluators[EvaluationName.LPPD].evaluate_and_save(
                    phase=phase,
                    pred_dist=pred_dist[phase],
                    target=self.loader.data_test[1][: max_len * self.batch_size_s_test],
                )
        if EvaluationName.LPPD_CUM in self.config.evaluations:
            for phase in phases:
                evaluators[EvaluationName.LPPD_CUM].evaluate_and_save(
                    phase=phase,
                    pred_dist=pred_dist[phase],
                    target=self.loader.data_test[1][: max_len * self.batch_size_s_test],
                    n_orderings=self.config.evaluation_args["n_orderings"],
                )
        if EvaluationName.PRED_PERF in self.config.evaluations:
            for phase in phases:
                evaluators[EvaluationName.PRED_PERF].evaluate_and_save(
                    phase=phase,
                    pred_dist=pred_dist[phase],
                    target=self.loader.data_test[1][: max_len * self.batch_size_s_test],
                )
        if EvaluationName.PRED_SAVER in self.config.evaluations:
            for phase in phases:
                evaluators[EvaluationName.PRED_SAVER].evaluate_and_save(
                    phase=phase,
                    pred_dist=pred_dist[phase],
                    target=self.loader.data_test[1][: max_len * self.batch_size_s_test],
                )

    def train_de_member(
        self,
        state: TrainState,
        chains: jax.Array,
    ) -> tuple[TrainState, MetricsStore]:
        """Train Deep Ensemble Members.

        Args:
            state: Initial Training State.
            chains: Which deep ensemble members to train in parallel across devices
                using jax.pmap (requires n_devices >= len(chains)).
        """
        match self.task:
            case Task.REGRESSION:
                step_func = single_step_regr
                pred_func = predict_regr
            case Task.MEAN_REGRESSION:
                step_func = single_step_mean_regr
                pred_func = predict_mean_regr
            case _:
                raise NotImplementedError(f"Task {self.task} not yet implemented.")

        n_parallel = len(chains)
        valid_losses = jnp.array([]).reshape(n_parallel, 0)
        _stop_n = jnp.repeat(False, n_parallel)
        metrics_train, metrics_valid = [], []

        # Initialize best state and best validation loss
        best_state = state
        best_valid_loss = [jnp.inf] * n_parallel

        for epoch in range(self.config_warmstart.max_epochs):
            if jnp.all(_stop_n):
                break  # Early stopping condition

            # Train Single Epoch
            for i, batch in enumerate(
                self.loader.iter(
                    split="train",
                    batch_size=self.batch_size_w,
                    chains=chains,
                    progress=True,
                )
            ):
                state, metrics = jax.pmap(step_func)(state, batch[0], batch[1], _stop_n)
                # if isinstance(metrics, ClassificationMetrics):
                #     logger.info(
                #         f"Epoch {epoch} "
                #         f"| Batch {i} | Training Loss: {metrics.cross_entropy} "
                #         f"| Accuracy: {metrics.accuracy}"
                #     )
                # if isinstance(metrics, RegressionMetrics):
                #     logger.info(
                #         f"Epoch {epoch} | Batch {i} | Training Loss: {metrics.nlll} "
                #         f"| RMSE: {metrics.rmse}"
                #     )
                metrics_train.append(metrics)

            for batch in self.loader.iter(split="valid", chains=chains):
                metrics = jax.pmap(pred_func, in_axes=(0, 0, 0, None))(
                    state, batch[0], batch[1], False
                )
                metrics_valid.append(metrics)
                if isinstance(metrics, RegressionMetrics):
                    logger.info(
                        f"Epoch {epoch} | Validation Loss: {metrics.nlll} "
                        f"| RMSE: {metrics.rmse}"
                    )
                    valid_losses = jnp.append(
                        valid_losses,
                        metrics.nlll[..., None],
                        axis=-1,
                    )

                # Update best state if current validation loss is better do this individually
                # for each chain
                for i, loss in enumerate(valid_losses):
                    if loss[-1] is not jnp.nan:
                        if loss[-1] < best_valid_loss[i]:
                            best_valid_loss[i] = loss[-1]
                            best_state = best_state.replace(
                                params=jax.tree_map(
                                    lambda p, bp: bp.at[i].set(p[i]),
                                    state.params,
                                    best_state.params,
                                ),
                                batch_stats=jax.tree_map(
                                    lambda b, bb: bb.at[i].set(b[i]),
                                    state.batch_stats,
                                    best_state.batch_stats,
                                ),
                            )

                _stop_n += earlystop(
                    losses=valid_losses, patience=self.config_warmstart.patience
                )
                logger.info(f"Early stopping status at epoch {epoch}: {_stop_n}")

        if jnp.any(_stop_n):
            state = best_state

        if isinstance(metrics, RegressionMetrics):
            metrics_store = MetricsStore(
                train=RegressionMetrics.cstack(metrics_train),
                valid=RegressionMetrics.cstack(metrics_valid),
                # test=RegressionMetrics.cstack(metrics_test),
            )
        else:
            raise ValueError("Metrics type not recognized.")

        return state, metrics_store

def single_step_regr(
    state: TrainState, x: jnp.ndarray, y: jnp.ndarray, early_stop: bool = False
) -> tuple[TrainState, RegressionMetrics]:
    """Perform a single training step for regression Task."""

    def loss_fn(params: ParamTree):
        logits = state.apply_fn({"params": params}, x=x)
        loss = bm_metrics.GaussianNLLLoss(
            y=y,
            mu=logits[..., 0],
            sigma=jnp.exp(logits[..., 1]).clip(min=1e-6, max=1e6),
        )
        metrics = compute_metrics_regr(logits, y, step=state.step)
        return loss.mean(), metrics

    def _single_step(state: TrainState):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    def _fallback(state: TrainState):
        metrics = RegressionMetrics(step=state.step, nlll=jnp.nan, rmse=jnp.nan)
        return state, metrics

    return jax.lax.cond(early_stop, _fallback, _single_step, state)

def single_step_mean_regr(
    state: TrainState, x: jnp.ndarray, y: jnp.ndarray, early_stop: bool = False
) -> tuple[TrainState, RegressionMetrics]:
    """Perform a single training step for a mean regression Task."""

    def loss_fn(params: ParamTree):
        logits = state.apply_fn({"params": params}, x=x)
        loss = bm_metrics.GaussianNLLLoss(y=y, mu=logits[..., 0], sigma=1.0)
        metrics = compute_metrics_mean_regr(logits, y, step=state.step)
        return loss.mean(), metrics

    def _single_step(state: TrainState):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    def _fallback(state: TrainState):
        metrics = RegressionMetrics(step=state.step, nlll=jnp.nan, rmse=jnp.nan)
        return state, metrics

    return jax.lax.cond(early_stop, _fallback, _single_step, state)

def predict_regr(
    state: TrainState, x: jnp.ndarray, y: jnp.ndarray, early_stop: bool = False
) -> RegressionMetrics:
    """Predict the model for regression Task."""

    def _pred(state: TrainState, x: jnp.ndarray, y: jnp.ndarray):
        logits = state.apply_fn({"params": state.params}, x=x)
        return compute_metrics_regr(logits, y, step=state.step)

    def _fallback(*args, **kwargs):
        metrics = RegressionMetrics(step=state.step, nlll=jnp.nan, rmse=jnp.nan)
        return metrics

    return jax.lax.cond(early_stop, _fallback, _pred, state, x, y)


def predict_mean_regr(
    state: TrainState, x: jnp.ndarray, y: jnp.ndarray, early_stop: bool = False
) -> RegressionMetrics:
    """Predict the model for regression Task."""

    def _pred(state: TrainState, x: jnp.ndarray, y: jnp.ndarray):
        logits = state.apply_fn({"params": state.params}, x=x)
        return compute_metrics_mean_regr(logits, y, step=state.step)

    def _fallback(*args, **kwargs):
        metrics = RegressionMetrics(step=state.step, nlll=jnp.nan, rmse=jnp.nan)
        return metrics

    return jax.lax.cond(early_stop, _fallback, _pred, state, x, y)

def compute_metrics_regr(
    logits: jnp.ndarray, y: jnp.ndarray, step: jnp.ndarray = jnp.nan
) -> RegressionMetrics:
    """Compute the metrics for regression Task."""
    loss = bm_metrics.GaussianNLLLoss(
        y=y, mu=logits[..., 0], sigma=jnp.exp(logits[..., 1]).clip(min=1e-6, max=1e6)
    )
    se = bm_metrics.SELoss(y=y, mu=logits[..., 0])
    metrics = RegressionMetrics(step=step, nlll=loss.mean(), rmse=jnp.sqrt(se.mean()))
    return metrics

def compute_metrics_mean_regr(
    logits: jnp.ndarray, y: jnp.ndarray, step: jnp.ndarray = jnp.nan
) -> RegressionMetrics:
    """Compute the metrics for Mean Regression Task."""
    loss = bm_metrics.GaussianNLLLoss(y=y, mu=logits[..., 0], sigma=1.0)
    se = bm_metrics.SELoss(y=y, mu=logits[..., 0])
    metrics = RegressionMetrics(step=step, nlll=loss.mean(), rmse=jnp.sqrt(se.mean()))
    return metrics


def get_initial_state(
    rng: PRNGKey,
    x: jnp.ndarray,
    module: nn.Module,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Get the initial training state."""
    variables = module.init(rng, x=x, train=False)
    if "batch_stats" in variables:
        return TrainState.create(
            apply_fn=module.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables["batch_stats"],
        )
    return TrainState.create(
        apply_fn=module.apply,
        params=variables["params"],
        tx=optimizer,
        batch_stats=None,
    )