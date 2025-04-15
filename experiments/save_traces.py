"""Save specific traces."""

import argparse
import os
from pathlib import Path

import jax.numpy as jnp
from tqdm import tqdm

from src.sai.config.core import Config
from src.sai.inference.sample_loader import SampleLoader


def parse_args():
    """Save specific traces."""
    parser = argparse.ArgumentParser(description="Save specific traces.")
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory containing the experiment"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of traces to save per layer",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer for which to save traces",
    )
    return parser.parse_args()


def main():
    """Save specific traces."""
    args = parse_args()
    DIR = Path(args.dir)

    # load config,yaml as dictionary
    config = Config.from_yaml(DIR / "config.yaml")

    n_layers = len(config["model"]["hidden_structure"])
    n_chains = config["training"]["sampler"]["n_chains"]

    os.makedirs(DIR / "traces", exist_ok=True)

    sample_loader = SampleLoader(DIR, config)

    layers_to_save = [args.layer] if args.layer >= 0 else range(n_layers)

    for layer in layers_to_save:
        print("Getting samples for layer", layer)  # noqa: T201
        samples_kernel = []
        samples_bias = []
        for samples_batch in tqdm(
            sample_loader.iter(
                batch_size=10,
                chains=[i for i in range(n_chains)],
                phase="sampling",
            )
        ):
            samples_kernel.append(
                samples_batch["fcn"][f"layer{layer}"]["kernel"][
                    :, :, : (args.n), 0
                ]
            )
            if samples_batch["fcn"][f"layer{layer}"]["bias"].shape[2] < args.n:
                samples_bias.append(samples_batch["fcn"][f"layer{layer}"]["bias"])
            else:
                samples_bias.append(
                    samples_batch["fcn"][f"layer{layer}"]["bias"][
                        :, :, : (args.n)
                    ]
                )
        samples_kernel = jnp.concatenate(samples_kernel, axis=1)
        samples_bias = jnp.concatenate(samples_bias, axis=1)
        for i in range(args.n):
            kernel = samples_kernel[:, :, i]
            bias = samples_bias[:, :, i]
            jnp.savez(DIR / f"traces/layer{layer}_kernel{i}.npz", kernel=kernel)
            jnp.savez(DIR / f"traces/layer{layer}_bias{i}.npz", bias=bias)


if __name__ == "__main__":
    main()
