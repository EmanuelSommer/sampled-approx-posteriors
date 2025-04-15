"""Scheduler functions for learning rate scheduling in mini-batch sampling."""

from typing import NamedTuple

import jax.numpy as jnp


class SchedulerState(NamedTuple):
    """State of the learning rate scheduler.

    Attributes:
        lr: float - Learning rate.
        explore: bool - Whether to explore or sample.
    """

    lr: float
    explore: bool


def cosine_annealing_scheduler(
    n_steps: int,
    n_cycles: int = 4,
    init_lr: float = 1e-3,
    target_lr: float = 0,
    exploration_ratio: float = 0.25,
):
    """Cosine annealing learning rate scheduler with restarts and exploration ratio.

    Args:
        n_steps: Total number of steps.
        n_cycles: Number of cycles.
        init_lr: Initial learning rate.
        target_lr: Target learning rate.
        exploration_ratio: Ratio of steps to explore \
        (These are steps with high learning rate).

    Returns:
        A function that takes the current step and returns `SchedulerState`.
    """
    cycle_length = n_steps // n_cycles
    assert target_lr <= init_lr, "target_lr should be less than or equal to init_lr"

    def step_fn(step_id: int) -> SchedulerState:
        multiplier = (step_id % cycle_length) / cycle_length
        cos_out = jnp.cos(jnp.pi * multiplier) + 1
        lr = jnp.minimum(0.5 * init_lr * cos_out, target_lr)
        return SchedulerState(
            lr=lr.item(), explore=(multiplier < exploration_ratio) or False
        )

    return step_fn


def linear_decay_scheduler(
    n_steps: int,
    init_lr: float = 1e-3,
    target_lr: float = 0,
    exploration_ratio: float = 0.25,
):
    """Linear decay learning rate scheduler with exploration ratio.

    Args:
        n_steps: Total number of steps.
        init_lr: Initial learning rate.
        target_lr: Target learning rate.
        exploration_ratio: Ratio of steps to explore \
        (These are steps with high learning rate).

    Returns:
        A function that takes the current step and returns `SchedulerState`.
    """
    assert target_lr <= init_lr, "target_lr should be less than or equal to init_lr"

    def step_fn(step_id: int) -> SchedulerState:
        lr = jnp.minimum(init_lr * (1 - step_id / n_steps), target_lr)
        return SchedulerState(
            lr=lr.item(), explore=(step_id < exploration_ratio * n_steps)
        )

    return step_fn
