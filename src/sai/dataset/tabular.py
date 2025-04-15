"""DataLoader Implementations for Tabular Data."""

import jax.numpy as jnp
import numpy as np

from src.sai.config.data import DataConfig, DatasetType, Task
from src.sai.dataset.base import BaseLoader
from src.sai.types import PRNGKey


class TabularLoader(BaseLoader):
    """DataLoader for tabular data."""

    def __init__(self, config: DataConfig, rng_key: PRNGKey, n_chains: int):
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
            y = (y - y.mean(axis=0)) / y.std(axis=0)
        return x, y.squeeze()
