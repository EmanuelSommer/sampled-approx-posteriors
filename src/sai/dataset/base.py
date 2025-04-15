"""Implementation of Base DataLoader class."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, overload

import jax
import jax.numpy as jnp
from tqdm import tqdm

from src.sai.config.data import DataConfig
from src.sai.types import PRNGKey


class Split(str, Enum):
    """Splits of the data."""

    TRAIN = "train"
    TEST = "test"
    VALID = "valid"


class BaseLoader(ABC):
    """Base class for all loaders."""

    def __init__(self, config: DataConfig, rng_key: PRNGKey, n_chains: int):
        """Initialize the loader."""
        assert isinstance(config, DataConfig)
        self.config = config
        _path_parts = config.path.strip("/").split("/")
        self._suffix = _path_parts[-1].split(".")[-1]
        self._name = _path_parts[-1].split(".")[0]
        self._dir = "/".join(_path_parts[:-1])
        self.n_chains = n_chains
        self._key = rng_key
        self._key_chains = jax.random.split(self.key, n_chains)

        self.data = self.load_data()
        if not isinstance(self.data, tuple):
            raise TypeError(
                f"`load_data()` method must return a tuple, got {type(self.data)}"
            )
        if not all(isinstance(x, jnp.ndarray) for x in self.data):
            raise TypeError(
                "All elements of the tuple must be jnp.ndarray, got types: "
                f"{tuple(type(x) for x in self.data)}"
            )

        self.splits = self.get_splits()
        self._pretty_str = (
            f"{self._name}(train={self.splits[Split.TRAIN].size}, "
            f"valid={self.splits[Split.VALID].size}, test={self.splits[Split.TEST].size})"
        )

    @abstractmethod
    def load_data(self) -> tuple[jnp.ndarray, ...]:
        """TODO: DOCUMENT THIS in DETAIL."""

    @property
    def key(self):
        """Return the next rng key for the dataloader."""
        self._key, key = jax.random.split(self._key)
        return key

    def chainwise_key(
        self,
        chains: jax.Array,
    ) -> jax.Array:
        """Handle the RNG state for all chains."""
        keys = jax.vmap(jax.random.split)(self._key_chains[chains])
        self._key_chains = self._key_chains.at[chains].set(keys[..., 0])
        return keys[..., 1]

    def __str__(self):
        """Return informative string representation of the class."""
        return self._pretty_str

    def get_split(self, split: Split):
        """Return the data for the specified split."""
        return tuple(x[self.splits[split]] for x in self.data)

    @property
    def data_train(self):
        """Return the training data."""
        return tuple(x[self.splits[Split.TRAIN]] for x in self.data)

    @property
    def data_valid(self):
        """Return the validation data."""
        return tuple(x[self.splits[Split.VALID]] for x in self.data)

    @property
    def data_test(self):
        """Return the testing data."""
        return tuple(x[self.splits[Split.TEST]] for x in self.data)

    def iter(
        self,
        split: Split | str,
        batch_size: Optional[int] = None,
        chains: Optional[jax.Array] = None,
        shuffle: bool = True,
        progress: bool = False,
        datapoint_limit: Optional[int] = None,
    ):
        """Iterate over data with optional shuffling.

        Args:
            split: The data split to iterate over (e.g., train, valid, test).
            batch_size: Size of each batch. If `None`, yields the entire data.
            chains: For which chains to load the data.
            shuffle: Whether to shuffle the data before batching.

        Returns:
            Generator yielding batches of data.
        """
        match split:
            case Split.TRAIN:
                data = self.data_train
            case Split.VALID:
                data = self.data_valid
            case Split.TEST:
                data = self.data_test
            case _:
                raise ValueError(f"Invalid split: {split}")
        if datapoint_limit is not None:
            data = tuple(d[:datapoint_limit, ...] for d in data)
        return self._iter(
            data=data,
            batch_size=batch_size,
            chains=chains,
            shuffle=shuffle,
            progress=progress,
        )

    def get_splits(self):
        """Return the indices of the splits."""
        i_s = len(self.data[0])
        assert all(
            len(x) == i_s for x in self.data
        ), "All data arrays must have the same length."
        i_t = int(i_s * self.config.train_split)
        i_v = int(i_s * (self.config.train_split + self.config.valid_split))
        splits = jnp.split(jnp.arange(i_s), [i_t, i_v])
        if len(splits) == 2:
            splits.append(jnp.array([], dtype=jnp.int32))
        return {Split.TRAIN: splits[0], Split.VALID: splits[1], Split.TEST: splits[2]}

    def shuffle(self, split: Split | str = Split.TRAIN):
        """Shuffle the data for the next dataloder iteration."""
        self.splits[split] = jax.random.permutation(self.key, self.splits[split])

    @overload
    def shuffle_arrays(self, arrays: jnp.ndarray) -> jnp.ndarray: ...

    @overload
    def shuffle_arrays(self, *arrays: jnp.ndarray) -> tuple[jnp.ndarray, ...]: ...

    def shuffle_arrays(  # type: ignore[misc]
        self, *arrays: jnp.ndarray
    ) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
        """Shuffle array along the first axis."""
        if not arrays:
            raise ValueError("At least one array must be provided.")
        len_i = len(arrays[0])
        assert all(
            len(x) == len_i for x in arrays
        ), "All arrays must have the same length."

        i_s = jax.random.permutation(self.key, jnp.arange(len_i))
        shuffled = tuple(x[i_s] for x in arrays)
        return shuffled if len(shuffled) > 1 else shuffled[0]

    def _iter(
        self,
        data: tuple[jnp.ndarray, jnp.ndarray],
        batch_size: Optional[int] = None,
        chains: Optional[jax.Array] = None,
        shuffle: bool = True,
        progress: bool = False,
    ):
        """Iterate over the data helper.

        Args:
            data: Tuple of arrays representing the data split.
            batch_size: Size of each batch. If `None`, yields the entire data.
            chains: For which chains to load the data.
            shuffle: Whether to shuffle the data before batching.

        Yields:
            Batches of data split across devices.
        """
        if chains is None:
            chains = jnp.arange(self.n_chains)
        if (i_s := len(data[0])) == 0:
            return None
        if batch_size is None:
            yield tuple(jnp.repeat(x[None, ...], len(chains), axis=0) for x in data)
        else:
            n_batches = i_s // batch_size
            i_s = jnp.arange(n_batches * batch_size)  # Drops the last incomplete batch
            if shuffle:
                splits = [
                    jnp.array_split(jax.random.permutation(key, i_s), n_batches)
                    for key in self.chainwise_key(chains=chains)
                ]
            else:
                splits = [
                    jnp.array_split(i_s, n_batches) for _ in chains
                ]  # No shuffling across devices
            iterator = range(n_batches)
            if progress:
                iterator = tqdm(iterator, desc="Data Batch")
            for i in iterator:
                yield tuple(jnp.stack([x[split[i]] for split in splits]) for x in data)
