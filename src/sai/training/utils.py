"""Training utilities."""

import pickle
from pathlib import Path
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import PyTreeDef

from src.sai.types import ParamTree, PRNGKey
from src.sai.utils import get_flattened_keys


def save_tree(path: str | Path, tree: ParamTree):
    """Save tree in .pkl format."""
    with open(path, "wb") as f:
        pickle.dump(tree, f)


def load_tree(path: str | Path) -> PyTreeDef:
    """Load tree in .pkl format."""
    with open(path, "rb") as f:
        tree = pickle.load(f)
    return tree


def load_params(path: str | Path, tree_path: str | Path) -> ParamTree:
    """Load model parameters from disk."""
    tree = load_tree(tree_path)
    with jnp.load(path) as param_zip:
        leaves = [jnp.array(param_zip[i]) for i in param_zip.files]
    return jax.tree.unflatten(treedef=tree, leaves=leaves)


def load_params_batch(paths: list[str | Path], tree_path: str | Path) -> ParamTree:
    """Load multiple model parameters and stack them for pmap compatibility."""
    paths_ = [Path(p) for p in paths]
    paths_ = sorted(paths_, key=lambda x: int(x.stem.split("_")[-1]))
    tree = load_tree(tree_path)
    params_list = []
    for param in paths_:
        with jnp.load(param) as param_zip:
            params_list.append([jnp.array(param_zip[i]) for i in param_zip.files])
    leaves = [
        jnp.stack([p[i] for p in params_list]) for i in range(len(params_list[0]))
    ]
    return jax.tree.unflatten(tree, leaves)


def save_params(
    directory: str | Path, params: ParamTree, idx: int | None = None, prefix: str = ""
):
    """Save model parameters to disk.

    Args:
        directory: directory to save the parameters to
        params: model parameters to save
        idx: index to append to the file name (default: None)
        prefix: prefix to append to the file name (default: "")
    """
    if not isinstance(directory, Path):
        directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True)
    leaves, tree = jax.tree.flatten(params)
    if not (directory.parent / f"{prefix}tree").exists():
        save_tree(directory.parent / f"{prefix}tree", tree)
    param_names = get_flattened_keys(params)
    name = f"{prefix}params_{idx}.npz" if idx is not None else "params.npz"
    np.savez_compressed(directory / name, **dict(zip(param_names, leaves)))


def count_chains_and_samples(directory: Union[str, Path]) -> tuple[int, int]:
    """Counts the number of chains and samples in the given directory.

    Args:
        directory: Path to the directory containing chains.

    Returns:
        A tuple (n_chains, n_samples)
        indicating the number of chains and samples per chain.
    """
    if not isinstance(directory, Path):
        directory = Path(directory)

    chain_dirs = sorted(
        [d for d in directory.iterdir() if d.is_dir()],
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    n_chains = len(chain_dirs)
    if n_chains == 0:
        raise ValueError("No chains found in the specified directory.")

    # Count samples in the first chain directory as a reference
    first_chain_samples = sorted(
        [p for p in chain_dirs[0].iterdir() if p.suffix == ".npz"],
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    n_samples = len(first_chain_samples)

    return n_chains, n_samples


def load_posterior_samples(
    directory: Union[str, Path],
    tree_path: Union[str, Path],
    chain_indices: Optional[list[int]] = None,
    sample_indices: Optional[list[int]] = None,
) -> ParamTree:
    """Loads a specified number of chains and samples from posterior samples on disk.

    Args:
        directory: Path to the directory containing chains.
        tree_path: Path to the module tree used to reconstruct parameters.
        num_chains: Number of chains to load (default loads all available).
        num_samples: Number of samples per chain to load (default loads all available).

    Returns:
        A ParamTree containing the loaded samples.
        (Array shapes: (n_chains, n_samples, ...))
    """
    if not isinstance(directory, Path):
        directory = Path(directory)

    # Load tree structure
    with open(tree_path, "rb") as f:
        tree = pickle.load(f)

    # Get list of chains in the directory and limit to num_chains if specified
    chain_dirs = sorted(
        [d for d in directory.iterdir() if d.is_dir()],
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    if chain_indices is not None:
        chain_dirs = [chain_dirs[i] for i in chain_indices]

    chain_stack = []

    for chain_dir in chain_dirs:
        # Get samples in this chain directory and limit to num_samples if specified
        samples = sorted(
            [p for p in chain_dir.iterdir() if p.suffix == ".npz"],
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        if sample_indices is not None:
            samples = [samples[i] for i in sample_indices]

        # Load samples for the current chain and stack them
        chain_params, _ = _load_chain_samples(samples)
        chain_stack.append(chain_params)

    # Stack all chains
    leaves = [
        jnp.stack([chain[i] for chain in chain_stack])
        for i in range(len(chain_stack[0]))
    ]
    return jax.tree_util.tree_unflatten(tree, leaves)


def _load_chain_samples(
    samples: list[Path],
) -> tuple[list[jnp.ndarray], list[str] | None]:
    """Helper function to load and stack samples for a single chain.

    Args:
        samples: List of Paths to sample files for a single chain.

    Returns:
        Tuple of stacked samples as list of jnp.ndarrays and parameter names.
    """
    params_list = []
    param_names = None

    for sample_path in samples:
        with jnp.load(sample_path) as sample_zip:
            if param_names is None:
                param_names = list(sample_zip.keys())
            params_list.append([jnp.array(sample_zip[k]) for k in param_names])

    # Stack samples for each parameter
    stacked_params = [
        jnp.stack([p[i] for p in params_list]) for i in range(len(params_list[0]))
    ]
    return stacked_params, param_names


def split_key_by_tree(key: PRNGKey, pytree: ParamTree):
    """Expand an RNG key to the structure of a Pytree."""
    treedef = jax.tree.structure(pytree)
    keys = jax.random.split(key, treedef.num_leaves)
    return jax.tree.unflatten(treedef, keys)


def earlystop(losses: jnp.ndarray, patience: int):
    """Check early stopping condition for losses shape (n_devices, N_steps).

    Args:
        losses: Losses array of shape (n_devices, N_steps).
        patience: Maximum number of steps to wait for improvement.

    Returns:
        Boolean array of shape (n_devices,) where True indicates to stop.
    """
    if losses.shape[-1] < patience:
        return jnp.repeat(False, len(losses))

    is_nan = jnp.isnan(losses)
    nan_window = is_nan[:, -(patience):]
    stop_on_nan = jnp.all(nan_window, axis=1)

    reference_loss = losses[:, -(patience + 1)]
    reference_loss = jnp.expand_dims(reference_loss, axis=-1)
    recent_losses = losses[:, -(patience):]
    stop_on_loss = jnp.all(recent_losses >= reference_loss, axis=1)

    return jnp.logical_or(stop_on_loss, stop_on_nan)


def get_nn_size(params: ParamTree):
    """Calculate the size of a single NN from a ParamTree."""
    return sum(
        jax.tree.leaves(
            jax.tree.map(lambda leaf: jnp.prod(jnp.array(leaf[0, ...].shape)), params)
        )
    )


def random_permutation(rng_key, n):
    """Return a random permutation of [0, 1, ..., n-1] as a jnp.array."""
    return jax.random.permutation(rng_key, jnp.arange(n))

def permute_linear_layer(params, perm_in=None, perm_out=None):
    """
    Permute a linear layer's kernel and bias.
    kernel: shape [in_features, out_features]
    bias:   shape [out_features]
    """
    kernel = params["kernel"]
    bias   = params["bias"]
    # Permute rows (input dimension) if needed
    if perm_in is not None:
        kernel = kernel[perm_in, :]
    # Permute columns (output dimension) + bias if needed
    if perm_out is not None:
        kernel = kernel[:, perm_out]
        bias   = bias[perm_out]
    return {"kernel": kernel, "bias": bias}

def get_sequential_layer_names(params):
    """
    Return the layer names in ascending order: layer0, layer1, layer2, ...
    Adapt this if your layer naming scheme differs.
    """
    layer_names = sorted(params.keys(), key=lambda s: int(s.replace("layer","")))
    return layer_names

def build_layer_permutations(layer_names, params_dict, rng):
    """
    Given an ordered list of layer_names and the sub-dict `params_dict`,
    build a list of (perm_in, perm_out) for each layer, ensuring
    the out perm of layer i is the in perm of layer i+1.
    
    We skip permuting the final layer's output dimension to preserve
    the final output labels/order.
    
    Returns:
      perms_for_layers: dict keyed by layer name, value = (perm_in, perm_out)
      new_rng: updated PRNGKey
    """
    perms_for_layers = {}
    
    # We start with no permutation on input for layer0
    current_in_perm = None
    
    rng_current = rng
    
    for i, lname in enumerate(layer_names):
        layer_params = params_dict[lname]
        
        # Detect shape: linear or conv
        k = layer_params["kernel"]
        if k.ndim == 2:
            # linear
            in_dim, out_dim = k.shape
        elif k.ndim == 4:
            # conv
            # shape [filter_height, filter_width, in_channels, out_channels]
            _, _, in_dim, out_dim = k.shape
        else:
            # Not recognized -> skip
            # (in practice, handle or raise an error)
            in_dim = None
            out_dim = None

        # If this is NOT the final layer, we create a random out_perm
        if i < len(layer_names) - 1:
            rng_current, key_out = jax.random.split(rng_current)
            out_perm = random_permutation(key_out, out_dim)
        else:
            # Final layer => skip permuting output dimension
            rng_current, key_out = jax.random.split(rng_current)
            out_perm = random_permutation(key_out, out_dim)

        # Store (perm_in = current_in_perm, perm_out = out_perm)
        perms_for_layers[lname] = (current_in_perm, out_perm)

        # The out_perm of this layer becomes the in_perm of next layer
        current_in_perm = out_perm

    return perms_for_layers, rng_current

def permute_network_once(model_params, layer_names, perms_for_layers):
    """
    Permute all layers in `model_params["fcn"]` according to
    the adjacency-based (perm_in, perm_out) in `perms_for_layers`.

    Returns a new parameter dict of the same structure, but permuted.
    """
    def is_permutable_leaf(leaf):
        return (
            isinstance(leaf, dict) and
            "kernel" in leaf and
            "bias" in leaf
        )
    
    new_model_params = {}
    for lname in layer_names:
        layer_dict = model_params[lname]
        (p_in, p_out) = perms_for_layers[lname]

        kernel = layer_dict["kernel"]
        # Distinguish linear vs conv
        if kernel.ndim == 2:
            new_layer = permute_linear_layer(layer_dict, p_in, p_out)
        else:
            new_layer = layer_dict  # skip
        new_model_params[lname] = new_layer

    return new_model_params

def permute_network_params(params, base_param, rng):
    """
    Produce a set of permuted parameters that yield
    the same input->output function (up to internal re-labeled units)
    for a *sequential* feedforward network.

    Args:
      params:    A pytree of parameters with leading dimension n_chains.
                 For each leaf, shape is (n_chains, ...) .
      base_param: Integer index selecting which chain’s parameters serve
                  as the base for generating permutations.
      rng:       PRNGKey for sampling permutations.

    Returns:
      A pytree of shape (n_chains, ...) with the permuted parameters.
    """
    base_params = jax.tree_map(lambda x: x[base_param], params)
    
    # We assume the relevant layers live under params["fcn"].
    # Adjust if your naming structure differs.
    layer_names = get_sequential_layer_names(base_params["fcn"])

    n_chains = jax.tree_leaves(params)[0].shape[0]
    permuted_chains = []

    for i in range(n_chains):
       
        perms_for_layers, rng = build_layer_permutations(
            layer_names, base_params["fcn"], rng
        )
        
        
        if base_param == None: # If no base_param is given, we permute all chains
            chain_params = jax.tree_map(lambda x: x[i], params)
            permuted_fcn = permute_network_once(
                chain_params["fcn"], layer_names, perms_for_layers
            )
        
        else: # If a base_param is given, we permute only the base_param chain
            chain_params = jax.tree_map(lambda x: x[base_param], params)
            permuted_fcn = permute_network_once(
                chain_params["fcn"], layer_names, perms_for_layers
            )
            
        new_chain_params = dict(chain_params)
        new_chain_params["fcn"] = permuted_fcn
        
        permuted_chains.append(new_chain_params)

    permuted = jax.tree_map(lambda *xs: jnp.stack(xs, axis=0), *permuted_chains)
    return permuted

def permute_warmstart(warm_path: Path, n_chains: int, base_param: int | None, key: PRNGKey):
    """Permute warmstart parameters to start from symmetric solutions."""

    # Load warmstart parameters
    file_names = [warm_path / f"params_{int(chain)}.npz" for chain in range(n_chains)]
    warm_params = load_params_batch(file_names, warm_path.parent / "tree")

    return permute_network_params(warm_params, base_param, key)
