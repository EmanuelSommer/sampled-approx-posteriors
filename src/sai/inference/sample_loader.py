"""Implementation of a sample loader."""

import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from tqdm import tqdm

from src.sai.training.utils import count_chains_and_samples
from src.sai.types import PRNGKey

logger = logging.getLogger(__name__)
PHASES = ["ensemble_initialization", "warmup_sampling", "sampling"]


class SampleLoader:
    """Load samples."""

    def __init__(
        self,
        root_dir: str | Path,
        config,
        sample_warmup_dir: Optional[str | Path] = None,
    ):
        """Initialize the loader."""
        self._root_dir = Path(root_dir)
        if config.training.warmstart.warmstart_exp_dir is not None:
            self._root_dir_de = Path(config.training.warmstart.warmstart_exp_dir)
            logger.info(
                "Using chain initialization from experiment: "
                f"{str(self._root_dir_de).split('/')[-1]}"
            )
        else:
            self._root_dir_de = None  # type: ignore
        self._de_dir = config.training.warmstart._dir_name
        self._sample_warmup_dir = sample_warmup_dir
        self._sample_dir = config.training.sampler._dir_name
        try:
            self.evaluation_args = config.evaluation_args
        except AttributeError:
            self.evaluation_args = {}
        with open(self._root_dir / "tree", "rb") as f:
            self.tree = pickle.load(f)
        with open(self._root_dir / "tree_sampling", "rb") as f:
            self.tree_sampling = pickle.load(f)
        if (self._root_dir / "batch_stats_tree").exists():
            with open(self._root_dir / "batch_stats_tree", "rb") as f:
                self.tree_path_batch_stats = pickle.load(f)
        self.n_chains = count_chains_and_samples(directory=self.sample_dir(PHASES[2]))[
            0
        ]
        self.burn_in = config.training.sampler.burn_in

    def __str__(self):
        """Return informative string representation of the class."""
        return f"{'-'*50}\n\tSample Loader:\n" f" | Number of Chains: {self.n_chains}"

    def sample_dir(self, phase: str):
        """Path to the samples of the phase."""
        if phase == PHASES[0]:
            root_dir = self._root_dir_de or self._root_dir
            dir = self._de_dir
        elif phase == PHASES[1]:
            root_dir = self._root_dir
            dir = self._sample_warmup_dir
        elif phase == PHASES[2]:
            root_dir = self._root_dir
            dir = self._sample_dir

        if dir is not None:
            return root_dir / dir
        else:
            ValueError(f"Sample Laoder: Phase '{phase}' has no directory specified.")

    def ls_dir(self, phase: str):
        """List samples in directory for phase."""
        files = os.listdir(self.sample_dir(phase))
        files = [
            file
            for file in files
            if file.endswith(".npz")
            and (file.startswith("sample") or file.startswith("params"))
        ]
        files = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        return files

    def n_samples(self, phase: str):
        """Number of samples of the phase."""
        return count_chains_and_samples(directory=self.sample_dir(phase))[1]

    def _sample_parameters(
        self,
        param_dict: Dict[str, jnp.ndarray],
        sigma_dict: Dict[str, jnp.ndarray],
        key: PRNGKey,
    ) -> Dict[str, ArrayLike]:
        """Sample parameters from normal distributions using provided sigmas."""

        def add_noise(
            param_value: jnp.ndarray,
            sigma: jnp.ndarray,
            key: PRNGKey,
        ) -> jnp.ndarray:
            """Add scaled Gaussian noise to a parameter value."""
            noise = jax.random.normal(key, param_value.shape, param_value.dtype)
            return param_value + sigma * noise

        keys = jax.random.split(key, len(param_dict))
        key_dict = dict(zip(param_dict.keys(), keys))

        return jax.tree_util.tree_map(
            add_noise,
            param_dict,
            sigma_dict,
            key_dict,
        )

    def _load_sample(
        self,
        phase: str,
        file: str,
        chain: Optional[int] = None,
        sample_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load a sample, incorporating sigma-based sampling if available.

        Args:
            phase: One of 'ensemble_initialization', 'warmup_sampling', 'sampling'
            file: Name of the file to load
            chain: Chain index if applicable
            sample_idx: Sample index for PRNGKey generation in phase 0
        """
        phase_path = (
            self.sample_dir(phase)
            if chain is None
            else self.sample_dir(phase) / str(chain)
        )
        path = phase_path / file

        with jnp.load(path) as sample_npz:
            param_dict = dict(sample_npz)
            leaves = param_dict.values()

        sigma_path = phase_path / f"sigma_{file}"
        if phase == PHASES[0] and sigma_path.exists():
            with jnp.load(sigma_path) as sigma_npz:
                sigma_dict = dict(sigma_npz)

            # Use provided sample_idx for key generation
            key = jax.random.key(sample_idx if sample_idx is not None else 0)
            sampled_dict = self._sample_parameters(param_dict, sigma_dict, key)
            leaves = sampled_dict.values()

        sample_def = self.tree_sampling if phase != PHASES[0] else self.tree
        if isinstance(sample_def, tuple):
            if sample_def[1] == jax.tree_util.tree_structure(None):
                sample_def = sample_def[0]
        sample = jax.tree.unflatten(sample_def, leaves)

        batch_stats_file = phase_path / f"batch_stats_params_{sample_idx}.npz"
        if batch_stats_file.exists():
            with jnp.load(batch_stats_file) as batch_stats_npz:
                batch_stats_leaves = dict(batch_stats_npz).values()
            batch_stats = jax.tree.unflatten(
                self.tree_path_batch_stats, batch_stats_leaves
            )
            return {"params": sample, "batch_stats": batch_stats}

        return sample

    def iter(
        self,
        batch_size: Optional[int] = None,
        chains: Optional[jax.Array] = None,
        phase: str = PHASES[2],
        progress: bool = False,
    ):
        """Iterate over samples.

        Args:
            batch_size: Size of each batch. If `None`, yields entire chains.
            chains: Which chains to load.
            phase: One of 'ensemble_initialization', 'sample_warmup', 'sample'.
            progress: Whether to display a progress bar.

        Yields:
            Batches of samples with dim (n_chains, n_samples, ...)
        """
        if chains is None:
            chains = jnp.arange(self.n_chains)

        if phase == PHASES[0]:
            files = self.ls_dir(phase)
            sigma_path = self.sample_dir(phase) / f"sigma_{files[0]}"
            if "warmstart_param_samples" in self.evaluation_args:
                if not sigma_path.exists():
                    warnings.warn(
                        "You are about to draw multiple samples without having sigma values provided creating duplicate samples. If this behaviour is wanted you may ignore this warning."
                    )
                n_samples = self.evaluation_args["warmstart_param_samples"]
            else:
                n_samples = 1

            ls_chains = []
            for chain in chains:
                ls_samples = []
                for i in range(n_samples):
                    sample = self._load_sample(
                        phase=phase,
                        file=files[chain],
                        sample_idx=i,  # Pass sample index for PRNGKey generation
                    )
                    ls_samples.append(sample)
                ls_chains.append(jax.tree.map(lambda *s: jnp.stack(s), *ls_samples))

            yield jax.tree.map(lambda *s: jnp.stack(s), *ls_chains)
        else:
            files = os.listdir(self.sample_dir(phase) / "0")

            n_samples = self.n_samples(phase)
            if batch_size is None:
                batch_size = n_samples

            iter = range(
                self.burn_in if phase == PHASES[2] else 0,
                n_samples,
                batch_size,
            )
            if progress:
                iter = tqdm(iter, desc="Sample Batch")

            for j in iter:
                ls_chains = []
                for chain in chains:
                    ls_samples = []
                    for i in range(j, min(j + batch_size, n_samples)):
                        sample = self._load_sample(
                            phase=phase,
                            file=files[i],
                            chain=chain,
                        )
                        ls_samples.append(sample)

                    ls_chains.append(jax.tree.map(lambda *s: jnp.stack(s), *ls_samples))
                yield jax.tree.map(lambda *s: jnp.stack(s), *ls_chains)
