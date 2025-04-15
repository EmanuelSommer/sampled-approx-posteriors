"""Utility functions."""

import inspect
import json
import logging
import operator
import os
import time
from contextlib import contextmanager
from functools import reduce
from json.encoder import JSONEncoder
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CustomJSONEncoder(JSONEncoder):
    """Less restrictive JSON encoder, used only for pretty printing."""

    def default(self, obj):  # noqa: D102
        if inspect.isclass(obj):
            return obj.__name__
        if hasattr(obj, "__class__"):
            return obj.__class__.__name__
        return JSONEncoder.default(self, obj)


def pretty_string_dict(d: dict, indent: int = 3):
    """Pretty print serializable dictionary."""
    return json.dumps(d, indent=indent, cls=CustomJSONEncoder)


@contextmanager
def measure_time(name: str):
    """Masure execution time."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{name} took {end - start:.2f} seconds.")


def get_by_path(tree: dict[str, Any], path: list[str]):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, path, tree)


def set_by_path(tree: dict[str, Any], path: list[str], value):
    """Set a value in a dict by path."""
    reduce(operator.getitem, path[:-1], tree)[path[-1]] = value
    return tree


def get_flattened_keys(d: dict[str, Any], sep: str = "."):
    """Recursively get `sep` delimited path to the leaves of a tree."""
    keys: list[str] = []
    for k, v in d.items():
        if isinstance(v, dict):
            keys.extend([f"{k}{sep}{kk}" for kk in get_flattened_keys(v, sep=sep)])
        else:
            keys.append(k)
    return keys


def configure_xla_flags(
    overrides: Optional[dict[str, str]] = None, replace: bool = False
) -> str:
    """Configures XLA_FLAGS environment variable for JAX.

    Args:
        overrides: Dictionary of flags to override or add.
        replace: If True, overrides the current flags with the provided ones.

    Returns:
        The new XLA_FLAGS string.
    """
    if overrides is None:
        overrides = {}

    current_flags = os.environ.get("XLA_FLAGS", "")
    flags_dict = {}

    for flag in current_flags.split():
        if "=" in flag:
            key, value = flag.lstrip("--").split("=", 1)
            flags_dict[key] = value
        else:
            flags_dict[flag.lstrip("--")] = ""

    if replace:
        flags_dict = overrides
    else:
        flags_dict.update(overrides)

    new_flags = " ".join(
        f"--{key}={value}" if value != "" else f"--{key}"
        for key, value in flags_dict.items()
    )
    # make sure there is no double space
    new_flags = " ".join(new_flags.split())
    os.environ["XLA_FLAGS"] = new_flags
    return new_flags
