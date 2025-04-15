"""Evaluation functions for DE and BDE models."""

import json
import logging
from enum import Enum
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
from dataserious.base import JsonSerializableDict

import src.sai.inference.metrics as metrics
from src.sai.bnns.logliks import ProbabilisticModel
from src.sai.config.data import Task
from src.sai.dataset.base import BaseLoader
from src.sai.inference.sample_loader import SampleLoader

logger = logging.getLogger(__name__)


class EvaluationName(str, Enum):
    """Evaluations to refere from the config file.

    Attributes:
        LPPD: Log pointwise predictive density. + Running LPPD.
        LPPD_CUM: Cumulative LPPD over multiple orderings.
        PRED_PERF: Predictive performance depending on the task.
        CRHAT: chainwise Rhat for local mixing.
        ESS: Effective sample size.
        CHAIN_VAR: Chain variances (between and within).
        CALIBRATION: Calibration of the model (coverage/ECE).
        PRED_SAVER: Save the predictions of the model.
    """

    LPPD = "Lppd"
    LPPD_CUM = "LppdCum"
    PRED_PERF = "PredPerf"
    CRHAT = "Crhat"
    ESS = "Ess"
    CHAIN_VAR = "ChainVar"
    CALIBRATION = "Calibration"
    PRED_SAVER = "PredSaver"
    PRIOR_LOGLIK = "PriorLogLik"
    PREDICT = "Predict"


class Evaluator:
    """Base class for evaluation functions."""

    name: EvaluationName
    post_predict: bool = False

    def __init__(self, task: Task, path: str):
        """Initialize the evaluator."""
        self.task = task
        self.path = path
        self.save_str = ""
        if not (Path(path) / "eval").exists():
            (Path(path) / "eval").mkdir()

    def log(self, phase: str):
        """Print that the evaluator is run (use in evaluate_and_save)."""
        logger.info(f"Running evaluator {self.name.value} for {phase} phase.")

    def evaluate_and_save(self, phase: str, *args, **kwargs) -> None:
        """Evaluate the model and save the results (if possible human readable)."""
        self.log(phase)
        raise NotImplementedError

    def save_results_as_json(self, results: JsonSerializableDict, phase: str) -> None:
        """Save the results of the evaluation as JSON."""
        path = f"{self.path}/eval/{self.name.value}_{phase}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=4)

    def save_results_as_npz(self, results: dict[str, jnp.ndarray], phase: str) -> None:
        """Save the results of the evaluation as npz if the results are large arrays.

        Args:
            results: Dictionary of the results (jnp arrays).
            phase: Which phase is evaluated.
        """
        path = f"{self.path}/eval/{self.name.value}_{phase}.npz"
        jnp.savez(path, **results)

    @classmethod
    def from_name(cls, name: EvaluationName, task: Task, path: str):
        """Return the evaluator class based on the name."""
        for subclass in cls.__subclasses__():
            if subclass.name == name:
                return subclass(task=task, path=path)
        raise ValueError(f"Evaluator for {name} not found.")


class Lppd(Evaluator):
    """Log pointwise predictive density evaluation."""

    name = EvaluationName.LPPD
    post_predict: bool = True

    def evaluate_and_save(
        self, phase: str, pred_dist: jnp.ndarray, target: jnp.ndarray
    ):
        """Log pointwise predictive density evaluation."""
        self.log(phase)
        results = {}
        lppd_pointwise = metrics.lppd_pointwise(
            pred_dist=pred_dist, y=target, task=self.task
        )
        results["lppd"] = metrics.lppd(lppd_pointwise=lppd_pointwise).item()
        chainwise_running_lppd = metrics.running_chainwise_lppd(
            lppd_pointwise=lppd_pointwise
        )
        running_lppd = metrics.running_lppd(lppd_pointwise=lppd_pointwise)

        self.save_results_as_json(results, phase)
        self.save_results_as_npz(
            {
                "chainwise_running_lppd": chainwise_running_lppd,
                "running_lppd": running_lppd,
            },
            phase,
        )

class LppdCum(Evaluator):
    """Multi cumulative log pointwise predictive density evaluation."""

    name = EvaluationName.LPPD_CUM
    post_predict: bool = True

    def evaluate_and_save(
        self, phase: str, pred_dist: jnp.ndarray, target: jnp.ndarray, n_orderings: int
    ):
        """Multi cumulative log pointwise predictive density evaluation."""
        self.log(phase)
        mean = jnp.zeros((pred_dist.shape[0],))
        m2 = jnp.zeros((pred_dist.shape[0],))  # for Welford's algorithm
        lppd_pointwise = metrics.lppd_pointwise(
            pred_dist=pred_dist, y=target, task=self.task
        )
        for i in range(1, n_orderings + 1):
            # randomly shuffle the first axis (chains) before calculating LPPD
            lppd_pointwise = jax.random.permutation(
                jax.random.key(i), lppd_pointwise, axis=0
            )
            current_lppd = metrics.running_seq_chain_lppd(lppd_pointwise=lppd_pointwise)
            delta = current_lppd - mean
            mean += delta / i
            m2 += delta * (current_lppd - mean)
        # calculate the final standard deviation
        sd = jnp.sqrt(m2 / n_orderings)
        self.save_results_as_npz(
            results={
                "mean_seq_lppd": mean,
                "std_seq_lppd": sd,
            },
            phase=phase,
        )

class PredPerf(Evaluator):
    """Predictive performance evaluation."""

    name = EvaluationName.PRED_PERF
    post_predict: bool = True

    def evaluate_and_save(
        self, phase: str, pred_dist: jnp.ndarray, target: jnp.ndarray
    ):
        """Predictive performance evaluation."""
        self.log(phase)
        results = {}
        match self.task:
            case Task.REGRESSION | Task.MEAN_REGRESSION:
                results["rmse"] = metrics.rmse(
                    pred=pred_dist[..., 0], target=target
                ).item()
        self.save_results_as_json(results, phase)


class PriorLogLik(Evaluator):
    """Prior value and log-likelihood of samples."""

    name = EvaluationName.PRIOR_LOGLIK
    post_predict: bool = False

    def evaluate_and_save(
        self,
        phase: str,
        train_plan: jax.Array,
        sample_loader: SampleLoader,
        sample_batch_size: int,
        data_loader: BaseLoader,
        data_batch_size: int,
        prob_model: ProbabilisticModel,
    ):
        """Predictive performance evaluation."""
        self.log(phase)
        v_log_prior = jax.vmap(prob_model.log_prior)
        v_log_likelihood = jax.vmap(prob_model.log_likelihood, in_axes=(0, None, None))

        ls_log_lik = []
        ls_log_prior = []
        for chains in train_plan:
            log_lik_chain = []
            log_prior_chain = []
            for samples in sample_loader.iter(
                batch_size=sample_batch_size,
                chains=chains,
                phase=phase,
                progress=True,
            ):
                log_lik_batch = []
                for x, y in data_loader.iter(
                    split="test",
                    batch_size=data_batch_size,
                    chains=chains,
                    shuffle=False,
                    progress=True,
                ):
                    log_lik_batch.append(jax.pmap(v_log_likelihood)(samples, x, y))

                log_lik_chain.append(sum(log_lik_batch))
                log_prior_chain.append(jax.pmap(v_log_prior)(samples))

            ls_log_lik.append(jnp.concatenate(log_lik_chain, axis=1))
            ls_log_prior.append(jnp.concatenate(log_prior_chain, axis=1))

        results = {
            "log_prior": jnp.concatenate(ls_log_prior, axis=0),
            "log_lik": jnp.concatenate(ls_log_lik, axis=0),
        }

        self.save_results_as_npz(results, phase)


class Predict(Evaluator):
    """Predict from samples."""

    name = EvaluationName.PREDICT
    post_predict: bool = False

    def evaluate(
        self,
        phase: str,
        train_plan: jax.Array,
        sample_loader: SampleLoader,
        sample_batch_size: int,
        data_loader: BaseLoader,
        data_batch_size: int,
        datapoint_limit: int,
        model: nn.Module,
        **kwargs,
    ):
        """Predict from samples."""
        self.log(phase)

        def predict_fn(params, x):
            """Forward pass for a single set of parameters."""
            return model.apply({"params": params}, x, **kwargs)

        def predict_fn_batch_stats(params, x):
            """Forward pass for models with batch normalization."""
            return model.apply(
                {"params": params["params"], "batch_stats": params["batch_stats"]},
                x,
                train=False,
                **kwargs,
            )

        def predict_from_samples(samples, x):
            """Perform predictions for a batch of samples."""
            pred_fn = predict_fn_batch_stats if "batch_stats" in samples else predict_fn
            return jax.vmap(lambda params: pred_fn(params, x))(samples)

        predictions = []
        for chains in train_plan:
            chain_predictions = []
            for samples in sample_loader.iter(
                batch_size=sample_batch_size,
                chains=chains,
                phase=phase,
                progress=True,
            ):
                batch_predictions = []

                if "sigma" in samples:
                    sigma = samples["sigma"]
                    samples = samples["params"]
                else:
                    sigma = None

                for x, _ in data_loader.iter(
                    split="test",
                    batch_size=data_batch_size,
                    chains=chains,
                    shuffle=False,
                    progress=True,
                    datapoint_limit=datapoint_limit,
                ):
                    preds_batch = jax.pmap(predict_from_samples)(samples, x)
                    if sigma is not None:
                        # Add sigma to the predictions
                        sigma_expanded = jnp.expand_dims(
                            jnp.repeat(sigma[:, None], preds_batch.shape[2], axis=1),
                            axis=-1,
                        )
                        sigma_expanded = jnp.swapaxes(sigma_expanded, 1, 2)
                        preds_batch = jnp.concatenate(
                            [preds_batch, sigma_expanded], axis=-1
                        )
                    batch_predictions.append(preds_batch)

                preds_chain = jnp.concatenate(batch_predictions, axis=2)
                chain_predictions.append(preds_chain)

            preds = jnp.concatenate(chain_predictions, axis=1)
            predictions.append(preds)

        return jnp.concatenate(predictions, axis=0)


class Crhat(Evaluator):
    """Chainwise Rhat evaluation."""

    name = EvaluationName.CRHAT
    post_predict: bool = False

    def evaluate_and_save(
        self,
        phase: str,
        train_plan: jax.Array,
        sample_loader: SampleLoader,
        n_splits: int = 4,
        rank_normalize: bool = True,
    ):
        """Chainwise Rhat evaluation.

        Args:
            phase: From which phase to take samples.
            train_plan: The BDETrainer.train_plan.
            sample_loader: A sample loader.
            n_splits: Number of splits for the chainwise Rhat.
            rank_normalize: Rank normalize the samples before calculating Rhat.
        """
        self.log(phase)
        r_hat_fun = jax.jit(
            partial(
                metrics.split_chain_r_hat,
                n_splits=n_splits,
                rank_normalize=rank_normalize,
            )
        )
        ls_rhat = []
        for chains in train_plan:
            for samples in sample_loader.iter(
                batch_size=None,  # TODO: extend split_chain_r_hat() to sample setting
                chains=chains,
                phase=phase,
                progress=False,
            ):
                chain_rhat = jax.tree.map(r_hat_fun, samples)
            ls_rhat.append(chain_rhat)

        rhat = jax.tree.map(lambda *leave: jnp.concatenate(leave, axis=0), *ls_rhat)
        self.save_results_as_npz({"rhat": rhat}, phase)


class Ess(Evaluator):
    """Effective sample size evaluation."""

    name = EvaluationName.ESS
    post_predict: bool = False

    def evaluate_and_save(
        self,
        phase: str,
        train_plan: jax.Array,
        sample_loader: SampleLoader,
    ):
        """Effective sample size evaluation."""
        self.log(phase)
        ess_fun = jax.jit(metrics.effective_sample_size)
        ls_ess = []
        for chains in train_plan:
            for samples in sample_loader.iter(
                batch_size=None,  # TODO: extend ess() to sample setting
                chains=chains,
                phase=phase,
                progress=False,
            ):
                chain_ess = jax.tree.map(ess_fun, samples)
            ls_ess.append(chain_ess)

        ess = jax.tree.map(lambda *leave: jnp.concatenate(leave, axis=0), *ls_ess)
        self.save_results_as_npz({"ess": ess}, phase)


class ChainVar(Evaluator):
    """Chain variance evaluation."""

    name = EvaluationName.CHAIN_VAR
    post_predict: bool = False

    def evaluate_and_save(
        self,
        phase: str,
        sample_loader: SampleLoader,
    ):
        """Chain variance evaluation (between and within)."""
        self.log(phase)
        between_fun = jax.jit(metrics.between_chain_var)
        within_fun = jax.jit(metrics.within_chain_var)
        for samples in sample_loader.iter(
            batch_size=None,  # TODO: extend chain_var to sample setting
            chains=None,  # TODO: extend chain_var to chain setting
            phase=phase,
            progress=False,
        ):
            between_chain_var = jax.tree.map(between_fun, samples)
            within_chain_var = jax.tree.map(within_fun, samples)
        self.save_results_as_npz(
            {
                "between_chain_var": between_chain_var,
                "within_chain_var": within_chain_var,
            },
            phase,
        )


class PredSaver(Evaluator):
    """Save the predictions of the model."""

    name = EvaluationName.PRED_SAVER
    post_predict: bool = True

    def evaluate_and_save(
        self, phase: str, pred_dist: jnp.ndarray, target: jnp.ndarray
    ):
        """Save the predictions."""
        self.log(phase)
        self.save_results_as_npz({"pred_dist": pred_dist, "target": target}, phase)


class Calibration(Evaluator):
    """Calibration evaluation for Bayesian Deep Ensembles."""

    name = EvaluationName.CALIBRATION
    post_predict: bool = True

    def evaluate_and_save(
        self,
        phase: str,
        pred: jnp.ndarray,
        pred_dist: jnp.ndarray,
        target: jnp.ndarray,
        nominal_coverages: list[float],
        kappa: float,
    ):
        """Coverage evaluation for Bayesian Deep Ensembles."""
        self.log(phase)
        if self.task == Task.CLASSIFICATION:
            calibration_gaps, bin_sizes = metrics.calibration_gap_classification(
                nominal_coverages=nominal_coverages,
                y=target,
                preds=pred,
                pred_dist=pred_dist,
            )

            coverage_weights = metrics.coverage_weighting(
                nominal_coverage=nominal_coverages,
                kappa=kappa,
            )
            assert bin_sizes.shape == coverage_weights.shape

            calibration_error = metrics.calibration_error_classification(
                calibration_gaps=calibration_gaps,
                bin_sizes=bin_sizes,
                weights=coverage_weights,
            )
            results = {
                "nominal_coverages": nominal_coverages,
                "bin_sizes": bin_sizes.tolist(),
                "calibration_gaps": calibration_gaps.tolist(),
                "calibration_error": calibration_error.item(),
            }
            self.save_results_as_json(results, phase)
        elif self.task in (Task.REGRESSION, Task.MEAN_REGRESSION):
            coverage_weights = metrics.coverage_weighting(
                nominal_coverage=nominal_coverages,
                kappa=kappa,
            )
            observed_coverages = metrics.calculate_coverage_regression(
                nominal_coverages=nominal_coverages,
                y=target,
                preds=pred,
            )
            calibration_error = metrics.calibration_error_regression(
                nominal_coverage=nominal_coverages,
                observed_coverage=observed_coverages,
                weights=coverage_weights,
            )
            calibration_gap = metrics.calibration_gap_regression(
                nominal_coverage=nominal_coverages,
                observed_coverage=observed_coverages,
                weights=coverage_weights,
            )

            results = {
                "nominal_coverages": nominal_coverages,
                "observed_coverages": observed_coverages.tolist(),
                "calibration_error": calibration_error.item(),
                "calibration_gap": calibration_gap.item(),
            }
            self.save_results_as_json(results, phase)

        else:
            raise NotImplementedError(f"Calibration for {self.task} not implemented.")
