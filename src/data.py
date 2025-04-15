"""Dataloader for different types of data."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import overload, Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.config import DataConfig, DatasetType, Source, Task

class Split(str, Enum):
    """Splits of the data."""

    TRAIN = "train"
    TEST = "test"
    VALID = "valid"


class BaseLoader(ABC):
    """Base class for all loaders."""

    def __init__(self, config: DataConfig, rng_key, n_chains: int):
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

class ImageLoader():
    """DataLoader for Image data."""

    def __init__(self, config: DataConfig, rng):
        """__init__ method for the ImageLoader class."""
        assert config.data_type == DatasetType.IMAGE
        self.config = config
        self.key = rng

    def load_data(self, standard_splits=True):
        """Load the dataset from different sources."""
        if self.config.source == Source.TORCHVISION:
            if not standard_splits:
                data_x, data_y = self.shuffle_arrays(
                    *_get_torchvision_data(self._name, self._dir, standard_splits)
                )

                # Limit the number of datapoints
                data_x = data_x[: self.config.datapoint_limit]
                data_y = data_y[: self.config.datapoint_limit]

                # normalize
                if self.config.normalize:
                    data_x = data_x / 255.0

                return data_x, data_y
            
            else:
                d_train, d_test = _get_torchvision_data(self.config.path.split("/")[-1], self.config.path, standard_splits)

                if not isinstance(d_train.data, np.ndarray):
                    x_train = jnp.array(d_train.data.numpy())
                    y_train = jnp.array(d_train.targets.numpy())
                    x_test = jnp.array(d_test.data.numpy())
                    y_test = jnp.array(d_test.targets.numpy())
                
                else:
                    x_train = jnp.array(d_train.data)
                    y_train = jnp.array(d_train.targets)
                    x_test = jnp.array(d_test.data)
                    y_test = jnp.array(d_test.targets)

                # normalize
                if self.config.normalize:
                    x_train = x_train / 255.0
                    x_test = x_test / 255.0
                
                # create valid set as subset of train set
                if self.config.valid_split:
                    num_train = len(x_train)
                    len_val = int(num_train * self.config.valid_split)
                    indices = jax.random.permutation(self.key, jnp.arange(num_train))
                    val_indices = indices[:len_val]
                    train_indices = indices[len_val:]
                    x_val, y_val = x_train[val_indices], y_train[val_indices]
                    x_train, y_train = x_train[train_indices], y_train[train_indices]

                    self.data_valid_loaded = (x_val, y_val)

                self.data_train_loaded = (x_train, y_train)
                self.data_test_loaded = (x_test, y_test)

                return (x_train, y_train, x_val, y_val, x_test, y_test)

        else:
            raise NotImplementedError(
            f"Source {self.config.source} is not supported for image data."
            )
    
    def iter_image(
        self, split: Split | str, batch_size: int | None = None, n_devices: int = 1
    ):
        match split:
            case 'train':
                return self._iter_image(self.data_train_loaded, batch_size, n_devices=n_devices)
            case 'valid':
                return self._iter_image(self.data_valid_loaded, batch_size, n_devices=n_devices)
            case 'test':
                return self._iter_image(self.data_test_loaded, batch_size, n_devices=n_devices)
            case _:
                raise ValueError(f"Invalid split: {split}")
    
    def get_n_batches(self, split: str | str, batch_size: int | None = None):
        """Return the number of batches for the specified split."""
        if not batch_size:
            return 1
        return getattr(self, f'data_{split}_loaded')[0].shape[0] // batch_size
    
    def _iter_image(
        self,
        data: tuple[jnp.ndarray, jnp.ndarray],
        batch_size: int | None,
        n_devices: int = 1,
    ):
        """Iterate over the data helper."""
        if not (i_s := len(data[0])):
            return None
        if not batch_size:
            x, y = data
            if x.ndim == 3:  # if there is no channel dimension
                x = x[:, None, ...]  # add channel dimension at position 1
            yield tuple(jnp.repeat(x[None, ...], n_devices, axis=0) for x in (x, y))
        else:
            n_batches = i_s // batch_size
            i_s = jnp.arange(n_batches * batch_size)  # drops the last uncomplete batch
            splits = [
                jnp.array_split(jax.random.permutation(self.key, i_s), n_batches)
                for _ in range(n_devices)
            ]
            for i in range(n_batches):
                x, y = data
                if x.ndim == 3:  # if there is no channel dimension
                    x = x[:, None, ...]  # add channel dimension at position 1
                smallest_dim = jnp.argmin(jnp.array(x.shape[1:])).item() # assumes less channels than features
                x = jnp.moveaxis(x, smallest_dim + 1, 1)
                yield tuple(jnp.concatenate([x[split[i]] for split in splits], axis=0) for x in (x, y))

    


def _get_torchvision_data(name: str, dir: str, standard_splits:str):
    """Get torchvision datasets."""
    from torchvision import datasets, transforms

    match name:
        case "mnist":
            d_train = datasets.MNIST(
                dir, train=True, download=True, transform=transforms.ToTensor()
            )
            d_test = datasets.MNIST(
                dir, train=False, download=True, transform=transforms.ToTensor()
            )
        case "fashion_mnist":
            d_train = datasets.FashionMNIST(
                dir, train=True, download=True, transform=transforms.ToTensor()
            )
            d_test = datasets.FashionMNIST(
                dir, train=False, download=True, transform=transforms.ToTensor()
            )
        case "cifar10":
            d_train = datasets.CIFAR10(
                dir, train=True, download=True, transform=transforms.ToTensor()
            )
            d_test = datasets.CIFAR10(
                dir, train=False, download=True, transform=transforms.ToTensor()
            )
        case _:
            raise NotImplementedError(f"Dataset {name} is not supported.")

    if standard_splits:
        return d_train, d_test

    data_x = jnp.concatenate(
        [jnp.array(d_train.data.numpy()), jnp.array(d_test.data.numpy())],
        axis=0,
    )

    data_y = jnp.concatenate(
        [jnp.array(d_train.targets.numpy()), jnp.array(d_test.targets.numpy())],
        axis=0,
    )
    return data_x, data_y



class TabularLoader(BaseLoader):
    """DataLoader for tabular data."""

    def __init__(self, config: DataConfig, rng_key, n_chains: int):
        """__init__ method for the TabularLoader class."""
        assert config.data_type == DatasetType.TABULAR
        super().__init__(config=config, rng_key=rng_key, n_chains=n_chains)

    def load_data(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Load the dataset from the specified file."""
        match self._suffix:
            case "npy":
                data = np.load(self.config.path)
            case "csv":
                data = np.loadtxt(self.config.path, delimiter=",")
            case "data":
                data = np.genfromtxt(self.config.path, delimiter=" ")
            case _:
                raise NotImplementedError(
                    "Only .npy and .csv files are supported at this time."
                )

        data = self.shuffle_arrays(jnp.array(data))[: self.config.datapoint_limit]

        x, y = data[..., :-1], data[..., -1:]

        if self.config.normalize:
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            if self.config.task == Task.CLASSIFICATION:
                y = y.astype(jnp.int32)
            else:
                y = (y - y.mean(axis=0)) / y.std(axis=0)
        return x, y.squeeze()