"""What to load from training part."""

from typing import Callable, Optional, TypeAlias

from blackjax import (  # maybe wrap them in kernels.py for cleaner sampling
    hmc,
    mclmc,
    nuts,
)

__all__ = ["nuts", "hmc", "mclmc"]

KernelRegistry: TypeAlias = dict[str, Optional[Callable]]

KERNELS: KernelRegistry = {
    "nuts": nuts,
    "hmc": hmc,
    "mclmc": mclmc,
}

WARMUP_KERNELS: KernelRegistry = {
}
