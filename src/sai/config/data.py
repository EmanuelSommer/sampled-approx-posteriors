"""Dataset Configuration."""

import importlib.util
import os
import warnings
from dataclasses import field
from enum import Enum

from dataserious import BaseConfig


class Source(str, Enum):
    """Possible Sources to load the data from.

    Attributes:
        LOCAL: Data is stored in a local file.
        URL: Data is stored on some URL.
        HuggingFace: Data is loaded from the HuggingFace datasets.
        TorchVision: Data is loaded from the TorchVision datasets.

    """

    LOCAL = "local"


class Task(str, Enum):
    """Possible Tasks to perform on the data.

    Attributes:
        REGRESSION: Supervised Regression Task.
        CLASSIFICATION: Supervised Classification Task.
        MEAN_REGRESSION: Mean Regression Task.

    """
    
    REGRESSION = "regr"
    MEAN_REGRESSION = "mean_regr"

class DatasetType(str, Enum):
    """Data Type of the Dataset.

    Attributes:
        TABULAR: Data is in a tabular format.
        IMAGE: Data is in an image format.
        TEXT: Data is in a text format

    """

    TABULAR = "tabular"


class DataConfig(BaseConfig):
    """Configuration for the Dataset related parameters."""

    path: str = field(
        metadata={
            "description": "Path to the data can be a local path or a URL.",
            "searchable": True,
        }
    )
    source: Source = field(metadata={"description": "Source of the data."})
    data_type: DatasetType = field(metadata={"description": "Type of the dataset."})
    task: Task = field(metadata={"description": "Task to be performed on the data."})
    target_column: str | None = field(
        default=None, metadata={"description": "Target Column Name or Index."}
    )
    target_len: int = field(
        default=1, metadata={"description": "Length of the Target Column."}
    )
    features: list[str] | None = field(
        default=None,
        metadata={
            "description": "names as list if None all columns except label are used."
        },
    )
    datapoint_limit: int | None = field(
        default=None,
        metadata={
            "description": "Maximum Number of Datapoints to Load.",
            "searchable": True,
        },
    )
    flatten: bool = field(
        default=False,
        metadata={"description": "Whether to make a vector out of the image."},
    )
    normalize: bool = field(
        default=False, metadata={"description": "Whether to Normalize the Data."}
    )
    train_split: float = field(
        default=0.8, metadata={"description": "Training Split Ratio."}
    )
    valid_split: float = field(
        default=0.1, metadata={"description": "Validation Split Ratio."}
    )
    test_split: float = field(
        default=0.1, metadata={"description": "Testing Split Ratio."}
    )

    def __post_init__(self):
        """Perform additional checks on this particular class."""
        super().__post_init__()
        assert (
            self.train_split + self.valid_split + self.test_split - 1.0
        ) < 1e-6, "Train, Validation, and Test Split should sum to 1.0"
        if self.source == Source.LOCAL:
            if not os.path.exists(self.path):
                warnings.warn(f"File {self.path} not found.")
        elif self.source == Source.HUGGINGFACE:
            if not importlib.util.find_spec("datasets"):
                raise ImportError(
                    'Install the "datasets" module to use the HuggingFace datasets.'
                )
        elif self.source == Source.TORCHVISION:
            if not importlib.util.find_spec("torchvision"):
                raise ImportError(
                    'Install the "torchvision" module to use the TorchVision datasets.'
                )
