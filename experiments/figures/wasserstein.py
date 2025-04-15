import os
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd

# possibly change working directory to the root of the repository
# os.chdir(...)

from experiments.figures.utils import (
    load_config_and_key,
    setup_loaders,
    get_train_plan_and_batch_size,
    setup_evaluators,
    get_predictions,
)

pred_perf_reg = np.load("") # Path to sampling run predictions on test set

def plot_wasserstein_distances(predictions_reg, predictions_perm=None, n_samples=50, n_chains=1, 
                             text_size=20, axis_text_size=18, cmap='Blues',
                             save_path=None, combined_plot=False, significance_matrix=None, alpha_sig = 0.8):
    """
    Create figure with Wasserstein distance plot for regular warmstart chains.
    Optionally create combined plot with permuted warmstart chains.
    
    Args:
        predictions_reg: Predictions from regular warmstart chains
        predictions_perm: Optional predictions from permuted warmstart chains
        n_samples: Number of samples per chain to use
        n_chains: Number of chains to use
        text_size: Size of title and label text
        axis_text_size: Size of tick labels
        cmap: Colormap to use for the plots
        save_path: Path to save plot
        combined_plot: Whether to create combined plot with both regular and permuted
        significance_matrix: Optional matrix of non-significant values to overlay on plot
        alpha_sig: Alpha value for significance matrix overlay
    """
    predictions_reg = predictions_reg.at[:,:,:,1].set(jnp.square(jnp.exp(predictions_reg[:,:,:,1])))
    if predictions_perm is not None:
        predictions_perm = predictions_perm.at[:,:,:,1].set(jnp.square(np.exp(predictions_perm[:,:,:,1])))

    def compute_pairwise_wasserstein_distances(flattened_preds):
        pairwise_wasserstein_distances = jax.vmap(
            jax.vmap(metrics.wasserstein_distance_gaussian,
                    in_axes=(None, 0)),
            in_axes=(0, None)
        )(flattened_preds, flattened_preds)
        return pairwise_wasserstein_distances

    # Prepare regular data
    wasserstein_selection_reg = np.array(predictions_reg)
    wasserstein_selection_reg = wasserstein_selection_reg[jnp.array(n_chains), :n_samples, :, :].reshape(-1, wasserstein_selection_reg.shape[2], wasserstein_selection_reg.shape[3])
    div_matrix_reg = compute_pairwise_wasserstein_distances(wasserstein_selection_reg)
    result_np_reg = np.array(div_matrix_reg)

    if combined_plot and predictions_perm is not None:
        fig, (ax_reg, ax_perm) = plt.subplots(1, 2, figsize=(20, 10))
        wasserstein_selection_perm = np.array(predictions_perm)
        wasserstein_selection_perm = wasserstein_selection_perm[jnp.array(n_chains), :n_samples, :, :].reshape(-1, wasserstein_selection_perm.shape[2], wasserstein_selection_perm.shape[3])
        div_matrix_perm = compute_pairwise_wasserstein_distances(wasserstein_selection_perm)
        result_np_perm = np.array(div_matrix_perm)
        max_abs_val = max(np.max(np.abs(result_np_reg)), np.max(np.abs(result_np_perm)))
    else:
        fig, ax_reg = plt.subplots(figsize=(10, 10))
        max_abs_val = np.max(np.abs(result_np_reg))
        im_reg = ax_reg.imshow(result_np_reg, 
                              cmap=cmap,
                              interpolation='nearest',
                              norm=plt.Normalize(vmin=0.0, vmax=max_abs_val))

       
    im_reg = ax_reg.imshow(result_np_reg, 
                          cmap=cmap,
                          interpolation='nearest',
                          norm=plt.Normalize(vmin=0.0, vmax=max_abs_val))
    
    if significance_matrix is not None:
        ax_reg.imshow(significance_matrix, 
                        cmap=plt.matplotlib.colors.ListedColormap(['none', 'orange']),
                        alpha=alpha_sig,
                        interpolation='nearest')

    # Configure axes
    x_sections = np.arange(0, len(n_chains) * n_samples + 1, n_samples)
    y_sections = np.arange(0, len(n_chains) * n_samples + 1, n_samples)

    if isinstance(n_chains, list):
        x_labels = [f"Chain {c+1}: 0" for c in range(len(n_chains))] + [f"Chain {n_chains[-1]+1}: {n_samples}"]
        y_labels = [f"Chain {c+1}: 0" for c in range(len(n_chains))] + [f"Chain {n_chains[-1]+1}: {n_samples}"]
    else:
        x_labels = [f"Chain {i + 1}: 0" for i in range(len(x_sections))]
        y_labels = [f"Chain {i + 1}: 0" for i in range(len(y_sections))]

    ax_reg.set_xticks(x_sections - 0.5)
    ax_reg.set_xticklabels(x_labels)
    ax_reg.set_yticks(y_sections - 0.5)
    ax_reg.set_yticklabels(y_labels)
    ax_reg.spines['top'].set_visible(False)
    ax_reg.spines['right'].set_visible(False)
    ax_reg.spines['left'].set_visible(False)
    ax_reg.spines['bottom'].set_visible(False)

    plt.sca(ax_reg)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=1)
    if combined_plot:
        plt.xlabel("Posterior Samples", fontsize=text_size, labelpad=0)
        plt.ylabel("Posterior Samples", fontsize=text_size, labelpad=0)
    else:
        plt.xlabel("Posterior Samples", fontsize=text_size, labelpad=-60)
        plt.ylabel("Posterior Samples", fontsize=text_size, labelpad=-60)
    if combined_plot:
        plt.title("Regular Warmstarts", fontsize=text_size + 4, pad=20)
    ax_reg.tick_params(axis='both', which='major', labelsize=axis_text_size)

    if combined_plot and predictions_perm is not None:
        ax_perm.imshow(result_np_perm, 
                      cmap=cmap,
                      interpolation='nearest',
                      norm=plt.Normalize(vmin=0.0, vmax=max_abs_val))
        
        ax_perm.set_xticks(x_sections - 0.5)
        ax_perm.set_xticklabels(x_labels)
        ax_perm.set_yticks(y_sections - 0.5)
        ax_perm.set_yticklabels(y_labels)
        ax_perm.spines['top'].set_visible(False)
        ax_perm.spines['right'].set_visible(False)
        ax_perm.spines['left'].set_visible(False)
        ax_perm.spines['bottom'].set_visible(False)

        plt.sca(ax_perm)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=1)
        plt.xlabel("Posterior Samples", fontsize=text_size, labelpad=0)
        ax_perm.set_title("Permuted Warmstarts", fontsize=text_size + 4, pad=20)
        ax_perm.tick_params(axis='both', which='major', labelsize=axis_text_size, pad=10)

        plt.subplots_adjust(wspace=0.0)
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([1.0, 0.2, 0.03, 0.7])
    else:
        cbar_ax = fig.add_axes([1.0, 0.22, 0.06, 0.7])

    colorbar = fig.colorbar(im_reg, cax=cbar_ax)
    colorbar.ax.tick_params(labelsize=axis_text_size)
    colorbar.set_label('Squared Wasserstein Distance', fontsize=axis_text_size, labelpad=20)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig, ax_reg, div_matrix_reg


def compute_z_values(chain, pred_dist_array, n_samples=50):
    """
    Compute Z values comparing each sample with every other sample.
    
    Args:
        pred_dist_array: Array of predictions with shape (chains, samples, points, features)
        n_samples: Number of samples to compare (default=50)
        
    Returns:
        Array of Z values of shape (n_samples, n_samples)
    """
    z_matrix = np.zeros((n_samples, n_samples))
    
    # Compare each sample with every other sample
    for i in range(n_samples):
        pred_dist_i = pred_dist_array[chain, i, :, 0]
        pred_dist_sigma_i = np.square(np.exp(pred_dist_array[chain, i, :, 1]))
        
        for j in range(n_samples):
            if i != j: 
                pred_dist_j = pred_dist_array[chain, j, :, 0]
                pred_dist_sigma_j = np.square(np.exp(pred_dist_array[chain, j, :, 1]))
                
                rho = np.corrcoef(pred_dist_i, pred_dist_j)[0, 1]
                numerator = (pred_dist_i - pred_dist_j).mean()
                denominator = np.sqrt((pred_dist_sigma_i + pred_dist_sigma_j - 
                                    2 * rho * np.sqrt(pred_dist_sigma_i) * np.sqrt(pred_dist_sigma_j)).mean())
                
                z_matrix[i, j] = numerator * np.sqrt(pred_dist_array.shape[2]) / denominator
                
    return z_matrix

z_values = compute_z_values(0, pred_perf_reg['pred_dist'])

fig_reg, ax_reg, div_matrix_reg = plot_wasserstein_distances(
    predictions_reg=jnp.array(pred_perf_reg["pred_dist"]),
    n_samples=50,
    n_chains=[0],
    text_size=26,
    axis_text_size=24,
    cmap='Blues',
    save_path='plots/lppd_permutation/wasserstein_regular_second_chain_significance.pdf',
    combined_plot=False,
    significance_matrix=((abs(z_values) < 1.96).astype(int)),
    alpha_sig=0.9
)