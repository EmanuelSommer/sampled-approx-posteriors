import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# possibly change working directory to the root of the repository
# os.chdir(...)

import src.sai.inference.metrics as metrics

from experiments.figures.utils import (
    load_config_and_key,
    setup_loaders,
    get_train_plan_and_batch_size,
    setup_evaluators,
    get_predictions,
)

# Specify results config path
CONFIG_PATH_SMALL = ""
CONFIG_PATH_MID = ""
CONFIG_PATH_LARGE = ""

# Load configs
config_small, key_small = load_config_and_key(
    CONFIG_PATH_SMALL
)
config_mid, key_mid = load_config_and_key(
    CONFIG_PATH_MID
)
config_large, key_large = load_config_and_key(
    CONFIG_PATH_LARGE
)

# Setup loaders
sample_loader_small, samples_small, data_loader_small = setup_loaders(config_small, key_small)
sample_loader_mid, samples_mid, data_loader_mid = setup_loaders(config_mid, key_mid)
sample_loader_large, samples_large, data_loader_large = setup_loaders(config_large, key_large)

# Get train plans and batch sizes
train_plan_small, batch_size_s_test_small = get_train_plan_and_batch_size(config_small, data_loader_small)
train_plan_mid, batch_size_s_test_mid = get_train_plan_and_batch_size(config_mid, data_loader_mid)
train_plan_large, batch_size_s_test_large = get_train_plan_and_batch_size(config_large, data_loader_large)

# Setup evaluators
evaluators_small = setup_evaluators(config_small)
evaluators_mid = setup_evaluators(config_mid)
evaluators_large = setup_evaluators(config_large)

# Get predictions
pred_dist_small = get_predictions(evaluators_small, train_plan_small, sample_loader_small, 
                                 config_small, data_loader_small, batch_size_s_test_small)
pred_dist_mid = get_predictions(evaluators_mid, train_plan_mid, sample_loader_mid, 
                                config_mid, data_loader_mid, batch_size_s_test_mid)
pred_dist_large = get_predictions(evaluators_large, train_plan_large, sample_loader_large,
                                config_large, data_loader_large, batch_size_s_test_large)

samples_small = [s for s in sample_loader_small.iter()][0]
samples_mid = [s for s in sample_loader_mid.iter()][0]
samples_large = [s for s in sample_loader_large.iter()][0]

def plot_predictive_cis(pred_dist_small, pred_dist_mid, pred_dist_large,
                       data_loader_small, data_loader_mid, data_loader_large,
                       figsize=(10, 6), title_size=24, label_size=24, tick_size=22,
                       alpha=0.3, linewidth=2):
    """
    Plot 90% empirical confidence intervals for predictive distributions.
    
    Args:
        pred_dist_*: Dictionary containing predictions for each model
        data_loader_*: Data loaders for each model configuration
        figsize: Tuple specifying figure dimensions
        title_size: Font size for title
        label_size: Font size for axis labels
        tick_size: Font size for tick labels
        alpha: Transparency for CI regions
        linewidth: Width of the mean prediction lines
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for pred_dist, data_loader, label, color in [
        (pred_dist_small, data_loader_small, '1x8 (small)', '#2171b5'),
        (pred_dist_mid, data_loader_mid, '2x8 (mid)', '#6baed6'),
        (pred_dist_large, data_loader_large, '3x8 (large)', '#bdd7e7')
    ]:
        x = data_loader.data_test[0]
        predictions = pred_dist['sampling'][:, :, :, 0].reshape(-1, pred_dist['sampling'].shape[-2])
        mean_pred = np.mean(predictions, axis=0)
        lower = np.percentile(predictions, 5, axis=0)
        upper = np.percentile(predictions, 95, axis=0)
        
        # Plot the line with full opacity for legend
        ax.plot([], [], label=label, color=color, linewidth=linewidth)
        ax.scatter(data_loader.data_train[0], data_loader.data_train[1], color='black', alpha=0.5)
        ax.fill_between(x.squeeze(), lower, upper, alpha=alpha, color=color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=label_size)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.set_xticks([0,-2, -1, 0, 1, 2])
    ax.set_yticks([0, -2, 0, 2, 4])

    ax.set_title('90% Predictive CI', fontsize=title_size)
    ax.set_xlabel('x', fontsize=label_size)
    ax.set_ylabel('y', fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.legend(fontsize=label_size)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.2, 3.2)
    plt.tight_layout()
    return fig, ax

fig, ax = plot_predictive_cis(pred_dist_small, pred_dist_mid, pred_dist_large,
                                data_loader_small, data_loader_mid, data_loader_large)
plt.show()
plt.close()

def plot_wasserstein_histograms(predictions_small, predictions_mid, predictions_large, n_samples=50, n_chains=4, 
                               figsize=(12, 5), bins=50, alpha=0.7):
    """
    Create histograms of Wasserstein distances for both permuted and regular chains.
    
    Args:
        predictions_small: Predictions from permuted warmstart chains
        predictions_mid: Predictions from regular warmstart chains
        n_samples: Number of samples per chain to use
        n_chains: Number of chains to use
        figsize: Size of the figure
        bins: Number of histogram bins
        alpha: Transparency of histograms
    """
    # Transform output of second neuron into variance by exponentiating
    predictions_small = predictions_small.at[:,:,:,1].set(jnp.exp(predictions_small[:,:,:,1]))
    predictions_mid = predictions_mid.at[:,:,:,1].set(jnp.exp(predictions_mid[:,:,:,1]))
    predictions_large = predictions_large.at[:,:,:,1].set(jnp.exp(predictions_large[:,:,:,1]))

    def compute_pairwise_wasserstein_distances(flattened_preds):
        pairwise_wasserstein_distances = jax.vmap(
            jax.vmap(metrics.wasserstein_distance_gaussian,
                    in_axes=(None, 0)),
            in_axes=(0, None)
        )(flattened_preds, flattened_preds)
        return pairwise_wasserstein_distances

    # Prepare data
    wasserstein_selection_small = np.array(predictions_small)
    wasserstein_selection_small = wasserstein_selection_small[:n_chains, :n_samples, :, :].reshape(-1, wasserstein_selection_small.shape[2], wasserstein_selection_small.shape[3])

    wasserstein_selection_mid = np.array(predictions_mid)
    wasserstein_selection_mid = wasserstein_selection_mid[:n_chains, :n_samples, :, :].reshape(-1, wasserstein_selection_mid.shape[2], wasserstein_selection_mid.shape[3])

    wasserstein_selection_large = np.array(predictions_large)
    wasserstein_selection_large = wasserstein_selection_large[:n_chains, :n_samples, :, :].reshape(-1, wasserstein_selection_large.shape[2], wasserstein_selection_large.shape[3])

    # Compute distances
    div_matrix_small = compute_pairwise_wasserstein_distances(wasserstein_selection_small)
    div_matrix_mid = compute_pairwise_wasserstein_distances(wasserstein_selection_mid)
    div_matrix_large = compute_pairwise_wasserstein_distances(wasserstein_selection_large)

    # Convert to numpy and get upper triangular values (excluding diagonal)
    distances_small = np.triu(div_matrix_small, k=1).flatten()
    # filter out all exact zeros
    distances_small = distances_small[distances_small > 0]
    #distances_small = distances_small[np.where(np.triu_indices(len(wasserstein_selection_small), k=1)[0].reshape(-1) < len(distances_small))]

    distances_mid = np.triu(div_matrix_mid, k=1).flatten()
    #distances_mid = distances_mid[np.where(np.triu_indices(len(wasserstein_selection_mid), k=1)[0].reshape(-1) < len(distances_mid))]
    distances_mid = distances_mid[distances_mid > 0]

    distances_large = np.triu(div_matrix_large, k=1).flatten()
    #distances_large = distances_large[np.where(np.triu_indices(len(wasserstein_selection_large), k=1)[0].reshape(-1) < len(distances_large))]
    distances_large = distances_large[distances_large > 0]

    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    ax1.hist(distances_small, bins=bins, alpha=alpha, color='blue', label='Small')
    ax1.set_title('Small Network')
    ax1.set_xlabel('Wasserstein Distance')
    ax1.set_ylabel('Frequency')

    ax2.hist(distances_mid, bins=bins, alpha=alpha, color='green', label='Mid')
    ax2.set_title('Mid Network')
    ax2.set_xlabel('Wasserstein Distance')

    ax3.hist(distances_large, bins=bins, alpha=alpha, color='red', label='Large')
    ax3.set_title('Large Network')
    ax3.set_xlabel('Wasserstein Distance')

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    
    plt.tight_layout()

    # Add zero count annotations
    zero_count_small = np.sum(distances_small == 0)
    zero_count_mid = np.sum(distances_mid == 0)
    zero_count_large = np.sum(distances_large == 0)

    ax1.annotate(f'Zeros: {zero_count_small}', xy=(0.7, 0.95), xycoords='axes fraction')
    ax2.annotate(f'Zeros: {zero_count_mid}', xy=(0.7, 0.95), xycoords='axes fraction')
    ax3.annotate(f'Zeros: {zero_count_large}', xy=(0.7, 0.95), xycoords='axes fraction')

    return fig, (ax1, ax2, ax3), distances_small
    

# Create the histograms
fig, (ax1, ax2, ax3), distances_small = plot_wasserstein_histograms(
    pred_dist_small["sampling"],
    pred_dist_mid["sampling"],
    pred_dist_large["sampling"],
    n_samples=50,
    n_chains=4
)
plt.show()

plt.close()
