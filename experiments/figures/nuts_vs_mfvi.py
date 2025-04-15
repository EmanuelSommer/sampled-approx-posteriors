"""Python file to reproduce the NUTS vs MFVI comparison figure."""
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

# Run experiments and specify paths
CONFIG_PATH = ""
CONFIG_PATH_SCALE01 = ""
PATH_MFVI = ""
PATH_MFVI_SCALE01 = ""

# Load configs
config, key = load_config_and_key(
    CONFIG_PATH
)

config_scale01, key_scale01 = load_config_and_key(
    CONFIG_PATH_SCALE01
)

# Setup loaders
sample_loader, samples, data_loader = setup_loaders(config, key)
sample_loader_scale01, samples_scale01, data_loader_scale01 = setup_loaders(config_scale01, key_scale01)

# Get train plans and batch sizes
train_plan, batch_size_s_test = get_train_plan_and_batch_size(config, data_loader)
train_plan_scale01, batch_size_s_test_scale01 = get_train_plan_and_batch_size(config_scale01, data_loader_scale01)

# Setup evaluators
evaluators = setup_evaluators(config)
evaluators_scale01 = setup_evaluators(config_scale01)

# Get predictions
pred_dist = get_predictions(evaluators, train_plan, sample_loader, 
                                config, data_loader, batch_size_s_test)
pred_dist_scale01 = get_predictions(evaluators_scale01, train_plan_scale01, sample_loader_scale01, 
                                config_scale01, data_loader_scale01, batch_size_s_test_scale01)

samples = [s for s in sample_loader.iter()][0]
samples_scale01 = [s for s in sample_loader_scale01.iter()][0]

samples_mfvi = pd.read_pickle(
    PATH_MFVI
    )

samples_mfvi_scale01 = pd.read_pickle(
    PATH_MFVI_SCALE01
    )

color_start = "#348ABD"
color_mid = "#467821"
color_end = "#A60628"

textsize = 20
scattersize = 40

# Create figure with 2x2 subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot samples in the first subplot
# Create KDE plot for NUTS samples
sns.kdeplot(
    x=samples['params']['fcn']['layer0']['kernel'][:, :, :, :].flatten(),
    y=samples['params']['fcn']['layer1']['kernel'][:, :, :, :].flatten(),
    ax=axs[0], 
    levels=6,
    color=color_start,
    alpha=0.8,
    label='NUTS',
    bw_method=0.05,
    fill=True,
)

# Create KDE plot for MFVI samples 
sns.kdeplot(
    x=samples_mfvi['Dense_0']['kernel'][:, :, :, :].flatten(),
    y=samples_mfvi['Dense_1']['kernel'][:, :, :, :].flatten(),
    ax=axs[0],
    levels=10, 
    color=color_end,
    alpha=0.8,
    label='MFVI'
)

scatter1 = axs[0].scatter([],[], c=color_start, alpha=0.6, s=scattersize)
scatter2 = axs[0].scatter([],[], c=color_end, alpha=0.6, s=scattersize)
fig.legend(handles=[
    plt.scatter([], [], alpha=0.6, s=scattersize + 20, c=scatter1.get_facecolor()[0], label='NUTS samples'),
    plt.scatter([], [], alpha=0.6, s=scattersize + 20, c=scatter2.get_facecolor()[0], label='MFVI samples')
], fontsize=textsize -2, bbox_to_anchor=(0.5, 1.05), loc='upper center', ncol=1)
axs[0].set_xlabel('First Layer Weight', fontsize=textsize)
axs[0].set_ylabel('Second Layer Weight', fontsize=textsize)
axs[0].set_title(r'$\tau^2 = 1$', fontsize=textsize)

axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
axs[0].grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
axs[0].tick_params(axis='both', which='major', labelsize=textsize - 2)
axs[0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
axs[0].set_xticks([0, -4, -2, 2, 4])
axs[0].set_yticks([0, -4, -2, 2, 4])

# Add contour line for points that multiply to 1
x = np.linspace(-4, 4, 1000)
y = 1/x
# set y=0 to nan to avoid division by zero and remove values close to zero
y[abs(y) > 1e2] = np.nan
axs[0].plot(x, y, 'r--', alpha=0.7, linewidth=2, c = color_mid)
axs[0].legend(fontsize=14)

axs[0].set_xlim(-2.8, 2.8)
axs[0].set_ylim(-2.8, 2.8)

axs[0].legend().set_visible(False)  # Disable legends for each plot

# Different prior scale second subplot
sns.kdeplot(
    x=samples_scale01['params']['fcn']['layer0']['kernel'][:, :, :, :].flatten(),
    y=samples_scale01['params']['fcn']['layer1']['kernel'][:, :, :, :].flatten(),
    ax=axs[1],
    levels=6,
    color=color_start,
    alpha=0.8,
    label='NUTS',
    bw_method=0.1,
    fill=True,
)

sns.kdeplot(
    x=samples_mfvi_scale01['Dense_0']['kernel'][:, :, :, :].flatten(),
    y=samples_mfvi_scale01['Dense_1']['kernel'][:, :, :, :].flatten(),
    ax=axs[1],
    levels=10,
    color=color_end,
    alpha=0.8,
    label='MFVI'
)
axs[1].set_xlabel('First Layer Weight', fontsize=textsize)
axs[1].set_ylabel('Second Layer Weight', fontsize=textsize)
axs[1].set_title(r'$\tau^2 = 0.1$', fontsize=textsize)

axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_visible(False)
axs[1].spines['bottom'].set_visible(False)
axs[1].grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
axs[1].tick_params(axis='both', which='major', labelsize=textsize - 2)
axs[1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
axs[1].set_xticks([0, -4, -2, 2, 4])
axs[1].set_yticks([0, -4, -2, 2, 4])

x = np.linspace(-4, 4, 1000)
y = 1/x
# set y=0 to nan to avoid division by zero and remove values close to zero
y[abs(y) > 1e2] = np.nan
axs[1].plot(x, y, 'r--', alpha=0.7, linewidth=2, c = color_mid)
axs[1].legend(fontsize=14)

axs[1].set_xlim(-3.1, 3.1)
axs[1].set_ylim(-3.1, 3.1)

axs[1].legend().set_visible(False)

plt.tight_layout()