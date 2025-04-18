"""Base Deep Learning Model Configuration Class."""
from dataclasses import dataclass, field

import jax.numpy as jnp
from flax import linen as nn
from abc import ABC, abstractmethod
from typing import Generator, Literal

from src.base_config import BaseConfig, BaseStrEnum


class FloatPrecision(BaseStrEnum):
    """Floating Point Precision."""

    FLOAT16 = 'float16'
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    BFLOAT16 = 'bfloat16'

    @property
    def flax_dtype(self) -> jnp.dtype:
        """Get the Flax Data Type."""
        return getattr(jnp, self.value)


class Activation(BaseStrEnum):
    """Activation Function."""

    SIGMOID = 'sigmoid'
    RELU = 'relu'
    GELU = 'gelu'
    TANH = 'tanh'
    SOFTMAX = 'softmax'
    LEAKY_RELU = 'leaky_relu'
    IDENTITY = "identity"

    @property
    def flax_activation(self):
        """Get the Flax Activation Function."""
        if self.value == "identity":
            return lambda x: x
        return getattr(nn, self.value)
    
"""Dataset Configuration."""
import importlib.util
import os
from dataclasses import dataclass, field

from src.base_config import BaseConfig, BaseStrEnum


class Source(BaseStrEnum):
    """Possible Sources to load the data from.

    Notes
    -----
    - Local: Data is stored in a local file.
    - URL: Data is stored on some URL.
    - HuggingFace: Data is loaded from the HuggingFace datasets.
    - TorchVision: Data is loaded from the TorchVision datasets.
    """

    LOCAL = 'local'
    URL = 'url'
    HUGGINGFACE = 'huggingface'
    TORCHVISION = 'torchvision'


class Task(BaseStrEnum):
    """Possible Tasks to perform on the data."""

    REGRESSION = 'regr'
    CLASSIFICATION = 'class'


class DatasetType(BaseStrEnum):
    """Data Type of the Dataset.

    Notes
    -----
    - Tabular: Data is in a tabular format.
    - Image: Data is in an image format.
    - Text: Data is in a text format.
    """

    TABULAR = 'tabular'
    IMAGE = 'image'
    TEXT = 'text'


@dataclass(frozen=True)
class DataConfig(BaseConfig):
    """Configuration for the Dataset related parameters.

    Notes
    -----
    - Handling the I/O pre-processing, post-processing, etc.
        of the data should be defined in the `module_sandbox.dataset` module.
    - `module_sandbox.config` module is only responsible for
        Serialization/Deserialization and simple basic Validation checks on the config.
    """

    path: str = field(
        metadata={
            'description': 'Path to the data can be a local path or a URL.',
            'searchable': True,
        }
    )
    source: Source = field(metadata={'description': 'Source of the data.'})
    data_type: DatasetType = field(metadata={'description': 'Type of the dataset.'})
    task: Task = field(metadata={'description': 'Task to be performed on the data.'})
    target_column: str | None = field(
        default=None, metadata={'description': 'Target Column Name or Index.'}
    )
    target_len: int = field(
        default=1, metadata={'description': 'Length of the Target Column.'}
    )
    features: list[str] | None = field(
        default=None,
        metadata={
            'description': 'names as list if None all columns except label are used.'
        },
    )
    datapoint_limit: int | None = field(
        default=None, metadata={'description': 'Maximum Number of Datapoints to Load.'}
    )
    normalize: bool = field(
        default=False, metadata={'description': 'Whether to Normalize the Data.'}
    )
    train_split: float = field(
        default=0.8, metadata={'description': 'Training Split Ratio.'}
    )
    valid_split: float = field(
        default=0.1, metadata={'description': 'Validation Split Ratio.'}
    )
    test_split: float = field(
        default=0.1, metadata={'description': 'Testing Split Ratio.'}
    )

    def __post_init__(self):
        """Perform additional checks on this particular class."""
        super().__post_init__()
        assert (
            self.train_split + self.valid_split + self.test_split - 1.0
        ) < 1e-6, 'Train, Validation, and Test Split should sum to 1.0'
        if self.source == Source.LOCAL:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f'File {self.path} not found.')
        elif self.source == Source.HUGGINGFACE:
            if not importlib.util.find_spec('datasets'):
                raise ImportError(
                    'Install the "datasets" module to use the HuggingFace datasets.'
                )
        elif self.source == Source.TORCHVISION:
            if not importlib.util.find_spec('torchvision'):
                raise ImportError(
                    'Install the "torchvision" module to use the TorchVision datasets.'
                )
        if self.data_type == DatasetType.TEXT:
            if self.features:
                assert len(self.features) == 1, 'Text must have only one feature.'


class BaseLoader(ABC):
    """Base class for all loaders."""

    def __init__(self, config: DataConfig):
        """Initialize the loader."""
        assert isinstance(config, DataConfig)
        self.config = config

    @property
    def dataset_name(self):
        return self.config.path.split('/')[-1].split('.')[0]

    def __str__(self):
        """Return informative string representation of the class."""
        return (
            f'{self.__class__.__name__}:\n'
            f' | Data: {self.dataset_name}\n'
            f' | Task: {self.config.task}'
        )

    @abstractmethod
    def iter(
        self, split: Literal['train', 'test', 'valid'], batch_size: int | None, **kwargs
    ) -> Generator[dict[str, jnp.ndarray], None, None]:
        """
        Return the next batch of data in dictionary format. e.g.

            {
                'feature': jnp.ndarray,
                'label': jnp.ndarray
            }
        containing batched (batch_size) features and labels.
        """
        pass

    @abstractmethod
    def shuffle(self, *args, **kwargs) -> None:
        """Shuffle the data for the next .iter() call."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the length of the data."""
        pass


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    """Base Model representing Deep Learning Model Configuration."""

    model: str = field(
        metadata={
            'description': (
                'The LiteralString must match actual class name'
                'implementing the model!'
            )
        },
    )

    @classmethod
    def get_name_mapping(cls):
        """Get the mapping of model names to the model classes."""
        return {c.model: c for c in cls.get_all_subclasses()}
