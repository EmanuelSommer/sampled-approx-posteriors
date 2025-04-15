"""Sampler Configuration."""

import warnings
from dataclasses import field
from enum import Enum
from typing import Any, Optional

from dataserious import BaseConfig

from src.sai.bnns.priors import PriorDist
from src.sai.training.scheduler import (
    cosine_annealing_scheduler,
    linear_decay_scheduler,
)


class GetSampler(str, Enum):
    """Sampler Names."""

    NUTS = "nuts"
    MCLMC = "mclmc"
    HMC = "hmc"

    def get_kernel(self):
        """Get sampling kernel."""
        from src.sai.kernels import KERNELS

        if self.value not in KERNELS:
            raise NotImplementedError(
                f"Sampler for {self.value} is not yet implemented."
            )
        return KERNELS[self.value]

    def get_warmup_kernel(self):
        """Get warmup kernel."""
        from src.sai.kernels import WARMUP_KERNELS

        if self.value not in WARMUP_KERNELS:
            raise NotImplementedError(
                f"Warmup Kernel for {self.value} is not yet implemented."
            )
        return WARMUP_KERNELS[self.value]


class Scheduler(str, Enum):
    """Learning Rate Scheduler Names."""

    COSINE = "Cosine"
    LINEAR = "Linear"

    def get_scheduler(self):
        """Get the learning rate scheduler."""
        if self == Scheduler.COSINE:
            return cosine_annealing_scheduler
        if self == Scheduler.LINEAR:
            return linear_decay_scheduler
        raise NotImplementedError(
            f"Learning Rate Scheduler for {self.value} is not yet implemented."
        )


class PriorConfig(BaseConfig):
    """Configuration for the prior distribution on the model parameters.

    Note:
        The `name` should be a `PriorDist` enum value which defines the complete
        prior distribution, it can be a general distribution or a pre-defined one.
        To extend the possible priors, add a new value to the `PriorDist` enum.
        and extend the `get_prior` method accordingly. Through `parameters` field
        the user can pass as many keyword arguments from the configuration file
        as needed for the initialization of the prior distribution.
    """

    name: PriorDist = field(
        default=PriorDist.StandardNormal,
        metadata={"description": "Prior to Use", "searchable": True},
    )
    parameters: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Parameters for the prior distribution.",
            "searchable": True,
        },
    )

    def get_prior(self):
        """Get the prior distribution.

        Note:
            Get the prior by passing the parameters from the config to `get_prior`
            method of the `PriorDist` enum. See the `PriorDist` enum for more details.
        """
        return self.name.get_prior(**self.parameters)


class SchedulerConfig(BaseConfig):
    """Scheduler Configuration."""

    name: Optional[Scheduler] = field(
        default=None,
        metadata={"description": "Scheduler to Use.", "searchable": True},
    )
    exploration: float = field(
        default=0.25,
        metadata={"description": "Exploration Ratio.", "searchable": True},
    )
    target_lr: float = field(
        default=0.0,
        metadata={"description": "Target Learning Rate.", "searchable": True},
    )
    n_cycles: int = field(
        default=4,
        metadata={
            "description": "Number of Cycles [Cosine Scheduler].",
            "searchable": True,
        },
    )

    def __post_init__(self):
        """Post Initialization for the Scheduler Configuration."""
        super().__post_init__()
        n_cycles_default = self.__class__.__dataclass_fields__["n_cycles"].default
        if self.name == Scheduler.LINEAR and self.n_cycles != n_cycles_default:
            self._modify_field(**{"n_cycles": n_cycles_default})
            warnings.warn("Ignoring n_cycles in Linear Scheduler.", UserWarning)

    def get_scheduler(self, n_steps: int, init_lr: float):
        """Get the learning rate scheduler."""
        if self.name == Scheduler.COSINE:
            return self.name.get_scheduler()(
                n_steps=n_steps,
                n_cycles=self.n_cycles,
                init_lr=init_lr,
                target_lr=self.target_lr,
                exploration_ratio=self.exploration,
            )
        if self.name == Scheduler.LINEAR:
            return self.name.get_scheduler()(
                n_steps=n_steps,
                init_lr=init_lr,
                target_lr=self.target_lr,
                exploration_ratio=self.exploration,
            )


class SamplerConfig(BaseConfig):
    """Sampler Configuration."""

    name: GetSampler = field(
        default=GetSampler.NUTS, metadata={"description": "Sampler to Use."}
    )
    epoch_wise_sampling: bool = field(
        default=False,
        metadata={
            "description": "Perform epoch-wise or batch-wise in minibatch sampling."
        },
    )
    params_frozen: list[str] = field(
        default_factory=list,
        metadata={
            "description": (
                "Point delimited parameter names in pytree to freeze."
                "(not yet fully implemented)"
            )
        },
    )
    batch_size: int | None = field(
        default=None,
        metadata={"description": "Batch Size in SBI Training.", "searchable": True},
    )
    burn_in: int = field(
        default=0,
        metadata={
            "description": "Number of samples to discard from the main sampling phase.",
            "searchable": True,
        },
    )
    warmup_steps: int = field(
        default=50,
        metadata={"description": "Number of warmup steps.", "searchable": True},
    )
    n_chains: int = field(
        default=2,
        metadata={"description": "Number of chains to run.", "searchable": True},
    )
    n_samples: int = field(
        default=1000,
        metadata={"description": "Number of samples to draw.", "searchable": True},
    )
    use_warmup_as_init: bool = field(
        default=True,
        metadata={
            "description": "Use params resulting from warmup as initial for sampling."
        },
    )
    n_thinning: int = field(
        default=1, metadata={"description": "Thinning.", "searchable": True}
    )
    diagonal_preconditioning: bool = field(
        default=False,
        metadata={
            "description": "Use Diagonal Preconditioning (MCLMC).",
            "searchable": True,
        },
    )
    desired_energy_var_start: float = field(
        default=5e-4,
        metadata={
            "description": "Desired Energy Variance (MCLMC) at start of lin. decay.",
            "searchable": True,
        },
    )
    desired_energy_var_end: float = field(
        default=1e-4,
        metadata={
            "description": "Desired Energy Variance (MCLMC) at end of lin. decay.",
            "searchable": True,
        },
    )
    trust_in_estimate: float = field(
        default=1.5,
        metadata={"description": "Trust in Estimate (MCLMC).", "searchable": True},
    )
    num_effective_samples: int = field(
        default=100,
        metadata={
            "description": "Number of Effective Samples (MCLMC).",
            "searchable": True,
        },
    )
    step_size_init: float = field(
        default=0.005,
        metadata={"description": "Initial Step Size (MCLMC).", "searchable": True},
    )
    step_size: float = field(
        default=0.0001, metadata={"description": "Step Size.", "searchable": True}
    )
    mdecay: float = field(
        default=0.05, metadata={"description": "Momentum Decay.", "searchable": True}
    )
    n_integration_steps: int = field(
        default=1,
        metadata={"description": "Number of Integration Steps.", "searchable": True},
    )
    momentum_resampling: float = field(
        default=0.0,
        metadata={"description": "Momentum Resampling (adaSGHMC)", "searchable": True},
    )
    temperature: float = field(
        default=1.0, metadata={"description": "Temperature (SGLD)", "searchable": True}
    )
    running_avg_factor: float = field(
        default=0.0,
        metadata={
            "description": "Running average factor (rmsprop)",
            "searchable": True,
        },
    )

    keep_warmup: bool = field(
        default=False, metadata={"description": "Keep warmup samples."}
    )
    prior_config: PriorConfig = field(
        default_factory=PriorConfig,
        metadata={"description": "Prior configuration for the model."},
    )
    scheduler_config: SchedulerConfig = field(
        default_factory=SchedulerConfig,
        metadata={"description": "Learning Rate Scheduler Configuration"},
    )

    @property
    def scheduler(self):
        """Get the learning rate scheduler."""
        return self.scheduler_config.get_scheduler(
            n_steps=self.n_samples, init_lr=self.step_size
        )

    def __post_init__(self):
        """Post Initialization for the Sampler Configuration."""
        super().__post_init__()
        mini_batch_only = ["batch_size", "n_integration_steps", "mdecay", "step_size"]
        if self.name == GetSampler.NUTS:
            for fn in mini_batch_only:
                if getattr(self, fn) is not None:
                    default = self.__class__.__dataclass_fields__[fn].default
                    warnings.warn(f"Ignoring {fn} in NUTS Sampling.", UserWarning)
                    self._modify_field(**{fn: default})

    @property
    def prior(self):
        """Get the prior."""
        return self.prior_config.get_prior()

    def kernel(self, **kwargs):
        """Returns the kernel."""
        return self.name.get_kernel()(**self._sampler_kwargs, **kwargs)

    def warmup_kernel(self, **kwargs):
        """Returns the warmup kernel."""
        kernel = self.name.get_warmup_kernel()
        if kernel is None:
            return None
        else:
            return kernel(**self._warmup_kwargs, **kwargs)

    @property
    def _warmup_dir_name(self):
        """Return the directory name for saving warmup samples."""
        return "sampling_warmup"

    @property
    def _dir_name(self):
        """Return the directory name for saving samples."""
        return "samples"

    @property
    def _sampler_kwargs(self):
        """Sampler configs."""
        return {}

    @property
    def _warmup_kwargs(self):
        """Sampler configs."""
        if self.name.value in [GetSampler.ADASGHMC, GetSampler.SGHMC, GetSampler.SGLD]:
            return {
                "num_integration_steps": self.n_integration_steps,
                "mdecay": self.mdecay,
                "mresampling": self.momentum_resampling,
            }
        elif self.name.value == GetSampler.RMSPROP:
            return {
                "num_integration_steps": self.n_integration_steps,
                "mdecay": self.mdecay,
                "mresampling": self.momentum_resampling,
                "running_avg_factor": self.running_avg_factor,
            }
        else:
            return {}
