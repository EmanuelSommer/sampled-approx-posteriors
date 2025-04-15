"""Optimizers used for warmstarting in the module_sandbox."""

from typing import Callable

from optax import (
    adam,
    adamw,
    sgd,
)
from optax._src.base import GradientTransformation

OPTIMIZERS: dict[str, Callable[..., GradientTransformation]] = {
    "adam": adam,
    "adamw": adamw,
    "sgd": sgd,
}