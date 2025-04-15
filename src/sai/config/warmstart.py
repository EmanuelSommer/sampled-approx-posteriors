"""WarmStart configuration, employs Frequentist Pretraining for the model."""

from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any

from dataserious import BaseConfig


class Optimizer(str, Enum):
    """Optimizer Names."""

    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"

    def get_optimizer(self):
        """Get optimizer."""
        from src.sai.training.optimizers import OPTIMIZERS

        if self.value not in OPTIMIZERS:
            raise NotImplementedError(
                f"Optimizer for {self.value} is not yet implemented."
            )
        return OPTIMIZERS[self.value]


class OptimizerConfig(BaseConfig):
    """Optimizer Configuration Class."""

    name: Optimizer = field(
        default=Optimizer.ADAMW, metadata={"description": "Optimizer to Use."}
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"learning_rate": 1e-3},
        metadata={"description": "Optimizer Parameters, check optax documentation."},
    )

    def get_optimizer(self):
        """Get the optimizer."""
        return self.name.get_optimizer()(**self.parameters)


class WarmStartConfig(BaseConfig):
    """Warmstart Configuration Class, employs Frequentist Pretraining."""

    include: bool = field(
        default=True, metadata={"description": "Include Warmup in Training."}
    )
    optimizer_config: OptimizerConfig = field(
        metadata={"description": "Optimizer Configuration for Warms."},
        default_factory=OptimizerConfig,
    )
    warmstart_exp_dir: str | None = field(
        default=None, metadata={"description": "Path to Warmstart Experiment to use."}
    )
    max_epochs: int = field(
        default=1,
        metadata={"description": "Maximum Number of Epochs in warmup training."},
    )
    batch_size: int | None = field(
        default=None, metadata={"description": "Batch Size in Warmup Training."}
    )
    patience: int | None = field(
        default=None,
        metadata={"description": "Early Stopping Patience in warmup training."},
    )
    permutation_warmstart: bool = field(
        default=False,
        metadata={
            "description": "Whether to permute the warmstarts to start from symmetric solutions."
        },
    )

    def __post_init__(self):
        """Perform additional checks on this particular class."""
        super().__post_init__()
        if not self.include:
            return
        if self.warmstart_exp_dir:
            if not Path(self.warmstart_exp_dir).exists():
                raise FileNotFoundError(
                    f'Warmstart Path "{self.warmstart_exp_dir}" does not exist.'
                )
            pp = Path(self.warmstart_exp_dir).as_posix()
            self._modify_field(**{"warmstart_exp_dir": pp})

    @property
    def optimizer(self):
        """Get the optimizer from optax module."""
        return self.optimizer_config.get_optimizer()

    @property
    def _dir_name(self):
        """Get the directory name for the warmstart model."""
        return "warmstart"

    @property
    def _metrics_fname(self):
        """Get the metrics filename for the warmstart model."""
        return "metrics.pkl"
