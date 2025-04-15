"""Utility functions for generating figures."""
import jax
import jax.numpy as jnp
import numpy as np

from src.sai.config.core import Config
from src.sai.dataset.tabular import TabularLoader
from src.sai.inference.evaluation import EvaluationName, Evaluator
from src.sai.inference.sample_loader import SampleLoader


def load_config_and_key(path):
    config = Config.from_file(path)
    key = jax.random.key(config.rng)
    return config, key

def setup_loaders(config, key):
    sample_loader = SampleLoader(
        root_dir=config.experiment_dir,
        config=config,
        sample_warmup_dir=None,
    )
    samples = [s for s in sample_loader.iter()][0]
    data_loader = TabularLoader(config=config.data, rng_key=key, n_chains=config.n_chains)
    
    return sample_loader, samples, data_loader

def get_train_plan_and_batch_size(config, data_loader):
    train_plan = jnp.array_split(
        jnp.arange(config.n_chains), config.n_chains / jax.device_count()
    )
    batch_size_test = len(data_loader.data_test[0])
    return train_plan, batch_size_test

def setup_evaluators(config):
    if "Predict" not in config.evaluations:
        config.evaluations += ["Predict"]
    
    return {
        name: Evaluator.from_name(
            name=name,
            task=config.data.task,
            path=config.experiment_dir,
        )
        for name in config.evaluations
    }

def get_predictions(evaluators, train_plan, sample_loader, config, data_loader, batch_size):
    return {
        "sampling": evaluators[EvaluationName.PREDICT].evaluate(
            phase="sampling",
            train_plan=train_plan,
            sample_loader=sample_loader,
            sample_batch_size=config.evaluation_args["samples_batch_size"],
            data_loader=data_loader,
            data_batch_size=batch_size,
            datapoint_limit=None,
            model=config.get_flax_model(),
        )
    }