import os
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
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

CONFIG_PATH_PERM = ""
CONFIG_PATH_REG = ""

# load data & setup
config_perm, key_perm = load_config_and_key(CONFIG_PATH_PERM)
config_reg, key_reg = load_config_and_key(CONFIG_PATH_REG)

loader_perm, batch_size_s_test = setup_loaders(config_perm, key_perm)
loader_reg, _ = setup_loaders(config_reg, key_reg) 

train_plan = get_train_plan_and_batch_size(config_perm)

sample_loader_reg, samples_reg, data_loader_reg = setup_loaders(config_reg, key_reg)
sample_loader_perm, samples_perm, data_loader_perm = setup_loaders(config_perm, key_perm)

evaluators_perm = setup_evaluators(config_perm)
evaluators_reg = setup_evaluators(config_reg)

pred_dist_perm = get_predictions(evaluators_perm, train_plan, sample_loader_perm, config_perm, loader_perm, batch_size_s_test)
pred_dist_reg = get_predictions(evaluators_reg, train_plan, sample_loader_reg, config_reg, loader_reg, batch_size_s_test)

# Compute metrics for both permuted and regular models
for model_suffix, loader, pred_dist in [
    ("perm", loader_perm, pred_dist_perm),
    ("reg", loader_reg, pred_dist_reg)
]:
    y_test = loader.data_test[1]
    lppd_pointwise = metrics.lppd_pointwise(
        pred_dist=pred_dist["sampling"], y=y_test, task=config_perm.data.task
    )
    
    globals()[f"lppd_pointwise_{model_suffix}"] = lppd_pointwise
    

# Plot cumulative LPPD for both permuted and regular models
def plot_lppd_comparison(lppd_pointwise_perm, lppd_pointwise_reg, n_samples,
                        axis_text_size=24, text_size=28, figure_width=14, figure_height=6,
                        save_path=None, selected_chains=None):
    """
    Plot LPPD comparison between permuted and regular chains.
    
    Args:
        lppd_pointwise_perm: Pointwise LPPD for permuted chains
        lppd_pointwise_reg: Pointwise LPPD for regular chains 
        n_samples: Number of MCMC samples
        selected_chains: If provided, only plot these specific chain numbers (0-based indices)
    """
    
    sample_indices = np.arange(1, n_samples + 1)
    plot_data = []

    # Calculate chain-wise and ensemble LPPD
    if selected_chains is not None:
        lppd_pointwise_perm = lppd_pointwise_perm[np.array(selected_chains)]
        lppd_pointwise_reg = lppd_pointwise_reg[np.array(selected_chains)]
    
    chainwise_running_lppd_perm = metrics.running_chainwise_lppd(lppd_pointwise=lppd_pointwise_perm)
    chainwise_running_lppd_reg = metrics.running_chainwise_lppd(lppd_pointwise=lppd_pointwise_reg)
    running_lppd_perm = metrics.running_lppd(lppd_pointwise=lppd_pointwise_perm)
    running_lppd_reg = metrics.running_lppd(lppd_pointwise=lppd_pointwise_reg)

    n_chains_perm = len(chainwise_running_lppd_perm)
    n_chains_reg = len(chainwise_running_lppd_reg)

    # Add permuted data
    for i in range(n_chains_perm):
        plot_data.extend([
            {'samples': float(x), 'lppd': float(y), 'chain_nr': f'chain{i+1}', 'Chain': 'Individual Chain', 'type': 'Permuted Warmstarts'} 
            for x, y in zip(sample_indices, chainwise_running_lppd_perm[i])
        ])
    
    # Add ensemble
    plot_data.extend([
        {'samples': float(x), 'lppd': float(y), 'chain_nr': 'ens', 'Chain': 'Ensemble', 'type': 'Permuted Warmstarts'}
        for x, y in zip(sample_indices, running_lppd_perm)
    ])

    # Add regular data 
    for i in range(n_chains_reg):
        plot_data.extend([
            {'samples': float(x), 'lppd': float(y), 'chain_nr': f'chain{i+1}', 'Chain': 'Individual Chain', 'type': 'Regular Warmstarts'} 
            for x, y in zip(sample_indices, chainwise_running_lppd_reg[i])
        ])
    
    # Add ensemble
    plot_data.extend([
        {'samples': float(x), 'lppd': float(y),'chain_nr': 'ens', 'Chain': 'Ensemble', 'type': 'Regular Warmstarts'}
        for x, y in zip(sample_indices, running_lppd_reg)
    ])
    
    df = pd.DataFrame(plot_data)

    # Create a manual color scale mapping using tab10 colors
    if selected_chains is None:
        chain_names = [f'chain{i+1}' for i in range(max(n_chains_perm, n_chains_reg))] + ['ens']
    else:
        chain_names = [f'chain{i+1}' for i in selected_chains] + ['ens']
    chain_color_mapping = {chain: '#348ABD' for chain in chain_names[:-1]}
    chain_color_mapping['ens'] = '#A60628'
    
    # Create plot
    plot = (p9.ggplot(df, p9.aes(x='samples', y='lppd', group='chain_nr', color='Chain'))
        + p9.geom_line(size=2.0, alpha=0.7)  # Added transparency for better visibility
        + p9.facet_wrap('~type')
        + p9.theme_minimal()
        + p9.scale_color_manual(values={'Individual Chain': '#348ABD', 'Ensemble': '#A60628'})
        + p9.scale_y_continuous(limits=[0.17, 0.37])
        + p9.theme(
            panel_background=p9.element_rect(fill="white"),
            text=p9.element_text(size=text_size),
            axis_text=p9.element_text(size=axis_text_size),
            figure_size=(figure_width, figure_height))
        + p9.labs(x='#Samples', y='Cumulative LPPD')
        + p9.theme(legend_position='top', legend_title=p9.element_blank(), legend_text=p9.element_text(size=axis_text_size + 2))
    )
    
    if save_path:
        plot.save(save_path)
    
    return plot

plot_all = plot_lppd_comparison(
    lppd_pointwise_perm, lppd_pointwise_reg,
    n_samples=int(config_perm.training.sampler.n_samples / config_perm.training.sampler.n_thinning)
)

def plot_wasserstein_distances(predictions_perm, predictions_reg, n_samples=50, n_chains=4, 
                             text_size=20, axis_text_size=18, cmap='Blues',
                             save_path_reg=None, save_path_perm=None):
    """
    Create Wasserstein distance plots for both permuted and regular warmstart chains.
    
    Args:
        predictions_perm: Predictions from permuted warmstart chains
        predictions_reg: Predictions from regular warmstart chains
        n_samples: Number of samples per chain to use
        n_chains: Number of chains to use
        text_size: Size of title and label text
        axis_text_size: Size of tick labels
        cmap: Colormap to use for the plots
        save_path_reg: Path to save regular warmstart plot
        save_path_perm: Path to save permuted warmstart plot
    """

    predictions_perm = predictions_perm.at[:,:,:,1].set(jnp.exp(predictions_perm[:,:,:,1]))
    predictions_reg = predictions_reg.at[:,:,:,1].set(jnp.exp(predictions_reg[:,:,:,1]))

    def compute_pairwise_wasserstein_distances(flattened_preds):
        pairwise_wasserstein_distances = jax.vmap(
            jax.vmap(metrics.wasserstein_distance_gaussian,
                    in_axes=(None, 0)),
            in_axes=(0, None)
        )(flattened_preds, flattened_preds)
        return pairwise_wasserstein_distances

    # Prepare data
    wasserstein_selection_perm = np.array(predictions_perm)
    wasserstein_selection_perm = wasserstein_selection_perm[:n_chains, :n_samples, :, :].reshape(-1, wasserstein_selection_perm.shape[2], wasserstein_selection_perm.shape[3])

    wasserstein_selection_reg = np.array(predictions_reg)
    wasserstein_selection_reg = wasserstein_selection_reg[:n_chains, :n_samples, :, :].reshape(-1, wasserstein_selection_reg.shape[2], wasserstein_selection_reg.shape[3])

    div_matrix_perm = compute_pairwise_wasserstein_distances(wasserstein_selection_perm)
    div_matrix_reg = compute_pairwise_wasserstein_distances(wasserstein_selection_reg)

    result_np_perm = np.array(div_matrix_perm)
    result_np_reg = np.array(div_matrix_reg)

    def create_single_plot(result_np, title, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        max_abs_val = np.max(np.abs(result_np))
        im = ax.imshow(result_np, 
                      cmap=cmap,
                      interpolation='nearest',
                      norm=plt.Normalize(vmin=0.0, vmax=max_abs_val))

        x_sections = np.arange(0, n_chains * n_samples + 1, n_samples)
        y_sections = np.arange(0, n_chains * n_samples + 1, n_samples)

        x_labels = [f"chain{i + 1}: 0" for i in range(len(x_sections) - 1)] + [f"chain{n_chains}: {n_samples}"]
        y_labels = [f"chain{i + 1}: 0" for i in range(len(y_sections) - 1)] + [f"chain{n_chains}: {n_samples}"]

        # Shift ticks to align with grid boundaries
        ax.set_xticks(x_sections - 0.5)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_sections - 0.5)
        ax.set_yticklabels(y_labels)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.xticks(rotation=45)
        colorbar = plt.colorbar(im, ax=ax, label='Squared Wasserstein Distance', fraction=0.04)
        colorbar.ax.tick_params(labelsize=axis_text_size)
        colorbar.set_label('Squared Wasserstein Distance', fontsize=axis_text_size)

        plt.yticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=1)

        plt.xlabel("Posterior Samples", fontsize=text_size)
        plt.ylabel("Posterior Samples", fontsize=text_size)
        title_lines = title.split(" between ")
        plt.title(title_lines[0] + "\nbetween " + title_lines[1], fontsize=text_size)

        # Set tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=axis_text_size)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi = 300)
            
        return fig, ax

    # Create both plots
    fig_reg, ax_reg = create_single_plot(
        result_np_reg,
        "Wasserstein Dist. of Pred. Distr. between Chains from Regular Warmstarts",
        save_path_reg
    )

    fig_perm, ax_perm = create_single_plot(
        result_np_perm,
        "Wasserstein Dist. of Pred. Distr. between Chains from Permuted Warmstarts",
        save_path_perm
    )

    return (fig_reg, ax_reg), (fig_perm, ax_perm)


(fig_reg, ax_reg), (fig_perm, ax_perm) = plot_wasserstein_distances(
    pred_dist_perm["sampling"],
    pred_dist_reg["sampling"],
    n_samples=50,
    n_chains=4,
    text_size=20,
    axis_text_size=18,
    cmap='Oranges'
)