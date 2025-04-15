"""Train Script for BDE Experiments."""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional

from src.sai.utils import configure_xla_flags

logger = logging.getLogger(__name__)

PHASES = ["chain_init", "sampling", "evaluation"]


def train_bde(
    config: "Config",  # type: ignore # noqa
    n_devices: int,
    phase: Optional[list[str]] = None,
    folder: Optional[Path] = None,
):
    """Train a BDE model for a given configuration."""
    _ = configure_xla_flags(
        {"xla_force_host_platform_device_count": str(min(n_devices, config.n_chains))}
    )

    from src.sai.training.trainer import BDETrainer  # noqa

    trainer = BDETrainer(config=config, folder=folder)

    training_steps = {
        PHASES[0]: lambda: trainer.train_warmstart(),
        PHASES[1]: lambda: trainer.start_sampling(),
        PHASES[2]: lambda: trainer.evaluate(),
    }

    logger.info(f"Running experiment '{trainer.config.experiment_name}'.")
    for k, v in training_steps.items():
        if k in (phase or PHASES):
            v()
    logger.info("Code exited successfully.")


def main(
    config: Path,
    devices: int = 1,
    outer_parallel: bool = False,
    folder: Optional[Path] = None,
    phase: Optional[list[str]] = None,
    device_limit: Optional[int] = None,
    search_tree: Optional[str] = None,
):
    """Run main."""
    from src.sai.config.core import Config  # don't slow down --help flag

    # assertions
    if phase is not None:
        for x in phase:
            assert x in PHASES, f"Phase {x} is not implemented."
    if phase is None and folder is not None:
        ValueError(
            "Re-running an entire experiment in the same folder is not recommended."
        )
    # device limit currently disabled
    device_limit = device_limit or devices

    # config
    if config.exists():
        if config.is_dir():  # Load all configs from directory
            if search_tree is not None:
                warnings.warn(
                    "Ignoring search tree file when loading directory of configs.",
                    category=UserWarning,
                )
            configs = Config.from_dir(config)
        else:  # Single config file with potential search tree
            cfg = Config.from_file(config)
            if search_tree is None:
                configs = [cfg]
            else:
                # Remove duplicates
                configs = list(set(cfg.get_configs_grid_from_path(search_tree)))
                # Enumerate experiments to make sure they are unique
                for i, cfg in enumerate(configs):
                    cfg._modify_field(experiment_name=f"{cfg.experiment_name}_{i}")
    else:
        raise FileNotFoundError(f"Configuration file or directory not found: {config}")
    logger.info(f"Loaded {len(configs)} Experiment(s)")

    _ = configure_xla_flags(
        {
            "xla_cpu_multi_thread_eigen": "false",
            "xla_gpu_strict_conv_algorithm_picker": "false",
        }
    )

    if outer_parallel:
        import multiprocessing

        logger.info(
            "Disabling stream logging in outer parallel mode, "
            "logs will only appear in log files."
        )
        for cfg in configs:
            cfg._modify_field(logging=False)

        with multiprocessing.Pool(devices) as p:
            p.starmap(train_bde, [(cfg, 1) for cfg in configs])
    else:
        for cfg in configs:
            train_bde(config=cfg, n_devices=devices, phase=phase, folder=folder)


help = f"""examples:
  %(prog)s --help
      show this help message and exit

  %(prog)s --config config.yaml
      run the experiment specified in config.yaml

  %(prog)s --config config.yaml --phase {','.join(PHASES)}
      run only selected phases of the experiment

  %(prog)s --config config.yaml --phase {PHASES[2]} --folder airfoil_20241109-173850/
      (re)run the evaluation of the experiment"""

parser = argparse.ArgumentParser(
    prog="python -m sai",
    description="sampling-based BNN inference",
    epilog=help,
    argument_default=argparse.SUPPRESS,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--config",
    "-c",
    type=Path,
    help="path to the configuration file or directory",
    metavar="",
)
parser.add_argument(
    "--phase",
    type=lambda t: [s.strip() for s in t.split(",")],
    help="(re)run only selected parts of the code (comma-separated)",
    metavar="",
)
parser.add_argument(
    "--folder",
    type=Path,
    help="which experiment to (re)run ('experiment_name' plus date)",
    metavar="",
)
parser.add_argument(
    "--search-tree",
    "-s",
    help="path to the search tree file",
    metavar="",
)
parser.add_argument(
    "--devices",
    "-d",
    type=int,
    help="split up available compute to n-devices",
    metavar="",
)
# TODO Either remove or implement this feature
parser.add_argument(
    "--device-limit",
    type=int,
    help="ammount of compute to use (default: all) - currently disabled",
    metavar="",
)
parser.add_argument(
    "--outer-parallel",
    action="store_true",
    help="run experiments in parallel",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))  # Rely on the usual way of defaults in python.
