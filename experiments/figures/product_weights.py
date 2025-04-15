import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# possibly change working directory to the root of the repository
# os.chdir(...)

from experiments.figures.utils import (
    load_config_and_key,
    setup_loaders,
    get_train_plan_and_batch_size,
    setup_evaluators,
    get_predictions,
)

# Specify results config path
CONFIG_PATH = ""

# Load configs
config, key = load_config_and_key(
    CONFIG_PATH
)

# Setup loaders
sample_loader, samples, data_loader = setup_loaders(config, key)

# Get train plans and batch sizes
train_plan, batch_size_s_test = get_train_plan_and_batch_size(config, data_loader)

# Setup evaluators
evaluators = setup_evaluators(config)

# Get predictions
pred_dist = get_predictions(evaluators, train_plan, sample_loader, 
                                config, data_loader, batch_size_s_test)

samples = [s for s in sample_loader.iter()][0]

# %%
# Function that calculates new weights u_i that are defined as W_{1i} * w_{2i} over all neurons i

def multiply_weights(samples: dict):
    """    
    Args:
      W1 shape = (n_chains, n_samples, d, m)
      W2 shape = (n_chains, n_samples, m, 1)
    Returns:
      A concatenated array of shape (n_chains, n_samples, d*m)
    """
    W1 = samples["params"]["fcn"]["layer0"]["kernel"]
    W2 = samples["params"]["fcn"]["layer1"]["kernel"]
    
    # Number of columns m from W1's last dimension
    m = W1.shape[-1]
    
    # Multiply slices W1[..., i] and W2[..., i, :] and concatenate along the last axis
    multiplied_slices = [W1[:, :, :, i] * W2[:, :, i, :] for i in range(m)]
    return jnp.concatenate(multiplied_slices, axis=-1) # shape = (n_chains, n_samples, d*m), d data dim, m hidden neurons


# %%
def plot_kde(u, dims, chain_selection:list, save_path="product_kde", save=False):
    """
    [Previous docstring remains the same]
    """
    # Convert `u` to a NumPy array if it is a JAX array
    if not isinstance(u, np.ndarray):
        u = np.array(u)

    # Subset to selected chains
    u = u[np.array(chain_selection)]

    n_chains = u.shape[0]
    n_dims = len(dims)

    # --- Univariate case ---
    if n_dims == 1:
        # Create a single DataFrame for all chains
        all_data = []
        for chain_idx, chain_id in enumerate(chain_selection):
            chain_values = u[chain_idx, :, dims[0]]  # All samples for this chain, single dimension
            # Build a small DataFrame for each chain
            df_tmp = pd.DataFrame({
                "value": chain_values,
                "chain": np.repeat(chain_id + 1, len(chain_values))
            })
            all_data.append(df_tmp)
        df = pd.concat(all_data, ignore_index=True)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        ax.set_xticks([0, -4, -2, 2, 4])
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(0, 1.1)

        # Single call with hue
        sns.kdeplot(
            data=df,
            x="value",
            hue="chain",
            palette="tab10",
            linewidth=3.0,
            ax=ax
        )

        ax.set_xlabel('$u$', fontsize=22)
        ax.set_ylabel('Density', fontsize=22)
        ax.legend(title="Chain", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=False)
    # --- Bivariate case ---
    else:
        # Create figure with a gridspec layout
        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(n_dims - 1, n_dims - 1)

        axes = []
        # Create only the subplots we need (lower triangle)
        for i in range(1, n_dims):
            row_axes = []
            for j in range(i):
                ax = fig.add_subplot(gs[i-1, j])
                row_axes.append(ax)
            axes.append(row_axes)

        # Loop over every pair (i, j) for bivariate subplots
        for i in range(1, n_dims):
            for j in range(i):
                ax = axes[i-1][j]
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
                ax.set_xticks([0, -4, -2, 2, 4])
                ax.set_yticks([0, -4, -2, 2, 4])

                # Build DataFrame for this dimension pair
                all_data = []
                for chain_idx, chain_id in enumerate(chain_selection):
                    xvals = u[chain_idx, :, dims[j]]
                    yvals = u[chain_idx, :, dims[i]]
                    chain_col = np.repeat(chain_id + 1, len(xvals))
                    df_tmp = pd.DataFrame({
                        "Chain": chain_col,
                        "x": xvals,
                        "y": yvals
                    })
                    all_data.append(df_tmp)
                df_pair = pd.concat(all_data, ignore_index=True)
                df_pair["Chain"] = df_pair["Chain"].apply(lambda x: f"Chain {x:02d}").astype("category")

                # Plot the KDE
                sns.kdeplot(
                    data=df_pair,
                    x="x",
                    y="y",
                    hue="Chain",
                    palette="tab10",
                    linewidths=2.0,
                    multiple="layer",
                    ax=ax,
                )
                
                ax.set_xlabel(f'$u_{{{dims[j]+1}}}$', fontsize=14)
                ax.set_ylabel(f'$u_{{{dims[i]+1}}}$', fontsize=14)
                ax.set_xlim(-6, 6)
                ax.set_ylim(-6, 6)

                # Remove individual legends
                ax.get_legend().remove()

        # Add a single legend in the upper right position
        legend_ax = fig.add_subplot(gs[0, -1])
        legend_ax.axis('off')
        
        # Get unique chain labels for the legend
        chain_labels = [f"Chain {i+1:02d}" for i in chain_selection]
        # Create proxy artists for the legend
        legend_handles = [plt.Line2D([0], [0], color=plt.cm.tab10(i/10), lw=2) for i in range(len(chain_selection))]
        legend = legend_ax.legend(legend_handles, chain_labels, loc='center', fontsize=12, frameon=True, title='Chains', title_fontsize=12)
        legend_ax.add_artist(legend)

        plt.tight_layout()
        file_name = f"{save_path}_{str(dims)}.pdf"
        if save:
            plt.savefig(file_name, dpi=300)
        else:
            plt.show()
        plt.close()