"""Training Configuration."""

from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any

from dataserious import BaseConfig

from src.sai.config.sampler import SamplerConfig
from src.sai.config.warmstart import WarmStartConfig


class TrainingConfig(BaseConfig):
    """Training Configuration Class."""

    warmstart: WarmStartConfig = field(
        default_factory=WarmStartConfig,
        metadata={"description": "Configuration for Frequentist pretraining as warmup"},
    )
    sampler: SamplerConfig = field(
        default_factory=SamplerConfig,
        metadata={"description": "Sampler Configuration for Training."},
    )

    @property
    def optimizer(self):
        return self.warmstart.optimizer

    @property
    def prior(self):
        return self.sampler.prior

    @property
    def warmstart_path(self):
        if self.warmstart.warmstart_exp_dir:
            return Path(self.warmstart.warmstart_exp_dir)

    @property
    def has_warmstart(self):
        return self.warmstart.include
