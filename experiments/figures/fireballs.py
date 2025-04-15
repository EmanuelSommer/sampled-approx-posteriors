# %%
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import jax.numpy as jnp 
import numpy as np

color_start = "#7d7580"
color_mid = "#050df5"
color_end = "#ff00aa"
linear_cmap = LinearSegmentedColormap.from_list("custom_cmap", [color_start, color_mid])

def plot_bivar_density(
    w1, w2,
    xlab="w1", ylab="w2",
    title="", textsize=16,
    group=None,
    kde=False,
    kde_bandwidth=0.25,
    plot_wh=None,
    limiter=4,
    cmap="plasma",
    save_path="bivar_density.png",
):
    """
    Plots bivariate density for large vectors using datashader.
    """
    # Prepare the data
    df = pd.DataFrame({'W1': w1, 'W2': w2})
    if group is not None:
        df['group'] = pd.Categorical(group)
    else:
        df['group'] = None

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    if group is not None:
        # Categorical data
        if not kde:
            dsartist = dsshow(
                df,
                ds.Point("W1", "W2"),
                ds.count_cat("group"),
                shade_hook=tf.dynspread,
                width_scale=0.8,
                height_scale=0.8,
                norm="eq_hist",
                aspect="equal",
                ax=ax,
            )
        else:
            sns.kdeplot(
                data=df, x="W1", y="W2", hue="group", fill=False, bw_adjust=kde_bandwidth,
                common_norm=False, palette="plasma", ax=ax, legend=False
            )
    else:
        if not kde:
            dsartist = dsshow(
                df,
                ds.Point("W1", "W2"),
                ds.count(),
                #shade_hook=tf.dynspread,
                width_scale=3,
                height_scale=3,
                plot_height=plot_wh,
                plot_width=plot_wh,
                norm="eq_hist",
                aspect="equal",
                ax=ax,
                # blues as the colormap
                cmap="plasma" if cmap == "plasma" else linear_cmap,
                # cmap=linear_cmap,
            )
        else:
            sns.kdeplot(
                data=df, x="W1", y="W2", fill=True, bw_adjust=kde_bandwidth,
                common_norm=False, cmap="plasma", ax=ax, legend=False)

    ax.set_xlabel(xlab, fontsize=textsize)
    ax.set_ylabel(ylab, fontsize=textsize)
    ax.set_title(title, fontsize=textsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
    
    if limiter is not None:
        ax.set_xticks([0, -2, 2])
        ax.set_yticks([0, -2, 2])
        ax.set_xlim(-limiter, limiter)
        ax.set_ylim(-limiter, limiter)

    ax.tick_params(axis='both', which='major', labelsize=textsize - 2)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# %%

w1 = jnp.load("../../results/fireball/mile_air/traces/layer1_kernel0.npz")["kernel"]
w2 = jnp.load("../../results/fireball/mile_air/traces/layer1_kernel1.npz")["kernel"]
filter_groups = 10
filter_samples = 1000
group = False
if filter_groups is not None:
    w1 = w1[:filter_groups, :filter_samples]
    w2 = w2[:filter_groups, :filter_samples]
if not group:
    # flatten the arrays
    w1 = w1.flatten()
    w2 = w2.flatten()
    group = None
else:
    group = jnp.repeat(jnp.arange(w1.shape[0]), w1.shape[1])
    # make group a numpy array
    group = np.array(group)
    w1 = w1.flatten()
    w2 = w2.flatten()


# %%
# Call the function
plot_bivar_density(
    w1, w2, 
    xlab="", 
    ylab="", 
    group=group,
    textsize=28,
    title=f"",
    kde=True,
    kde_bandwidth=0.5,
    plot_wh=100,
    save_path="bivar_density.pdf",
)

# %%
# now I want to construct a grid of fireballs that compares the marginal distributions both within and across layers
# first generate the grid of plots then combine them into a single figure do one grid for biases and one for kernels

layers = [i for i in range(5)]
layer_types = ["kernel", "bias"]
sampler = "mile"
save_id_base = "firegrid_air"
subset_chains = 10

# generate the grid of plots
for layer1 in layers:
    for layer2 in layers:
        for layer_type in layer_types:
            # load data
            w1 = jnp.load(f"../../results/fireball/{sampler}_air/traces/layer{layer1}_{layer_type}0.npz")[layer_type]
            w2 = jnp.load(f"../../results/fireball/{sampler}_air/traces/layer{layer2}_{layer_type}1.npz")[layer_type]
            nchains = w1.shape[0]
            w1 = w1[:, :subset_chains]
            w2 = w2[:, :subset_chains]

            # first plot the full fireball
            group = None
            plot_bivar_density(
                w1.flatten(), w2.flatten(), 
                xlab="", 
                ylab="", 
                title="",
                group=group,
                textsize=28,
                plot_wh=700,
                kde=False,
                save_path=f"../../results/figs/fireballs/{save_id_base}_{sampler}_layer{layer1}_{layer2}_{layer_type}.pdf",
            )

# %%
