# %%
import plotnine as p9
import pandas as pd
import jax.numpy as jnp

import pandas as pd
import plotnine as p9

def plot_cumulative_lppd(
    running_lppd: list,
    sd_running_lppd: list,
    iterations: int,
    output_path: str,
    x_label: str = "#Chains",
    y_label: str = "Cumulative LPPD",
    text_size: int = 20,
    axis_text_size: int = 16,
    multiple: int = 1,
):
    """Plot and save the cumulative LPPD over iterations.

    Args:
        running_lppd (list): Values of the running LPPD.
        sd_running_lppd (list): Values of the standard deviation of the running LPPD
        iterations (int): Number of iterations.
        output_path (str): Path to save the plot.
        x_label (str): Label for the x-axis. Default is '#Chains'.
        y_label (str): Label for the y-axis. Default is 'Cumulative LPPD'.
        text_size (int): Font size for text elements. Default is 20.
        axis_text_size (int): Font size for axis text. Default is 16.
        multiple (int): Multiple of the iterations. Default is 1.
    """
    df = pd.DataFrame({
        "running_lppd": running_lppd,
        "sd_running_lppd": sd_running_lppd,
        "iteration": jnp.array(range(1, iterations + 1)) * multiple,
    })

    p = (
        p9.ggplot(df) + 
        p9.geom_line(p9.aes(x="iteration", y="running_lppd"), size=0.9) +
        p9.geom_ribbon(
            p9.aes(x="iteration", ymin="running_lppd - sd_running_lppd", ymax="running_lppd + sd_running_lppd"),
            alpha=0.3,
        ) +
        p9.scale_x_log10() +
        p9.labs(x=x_label, y=y_label) +
        p9.theme_minimal() +
        p9.theme(
            panel_background=p9.element_rect(fill="white"),
            text=p9.element_text(size=text_size),
            axis_text=p9.element_text(size=axis_text_size),
        )
    )
    if iterations == 10000:
        p = p + p9.scale_x_log10(
            breaks=[1, 10, 100, 1000, 10000], 
            labels=["1", "10", "100", "1k", "10k"],
        )

    p.save(output_path)

# %%

# %%
# cumulative lppd
cumulative_lppd_miles = jnp.load("../../data/fireball/mile_air/LppdCum_sampling.npz")
running_lppd = cumulative_lppd_miles["mean_seq_lppd"]
sd_running_lppd = cumulative_lppd_miles["std_seq_lppd"]
plot_cumulative_lppd(
    running_lppd=running_lppd,
    sd_running_lppd=sd_running_lppd,
    iterations=running_lppd.shape[0],
    output_path="../../results/cumulative_lppd_fireballs/cumulative_lppd_miles_air.pdf",
    multiple=1,
)
# %%

cumulative_lppd_fmnist = jnp.load("../../data/fireball/LppdCum_sampling_fmnist.npz")
running_lppd = cumulative_lppd_fmnist["mean_seq_lppd"]
sd_running_lppd = cumulative_lppd_fmnist["std_seq_lppd"]
plot_cumulative_lppd(
    running_lppd=running_lppd,
    sd_running_lppd=sd_running_lppd,
    iterations=running_lppd.shape[0],
    output_path="../../results/cumulative_lppd_fireballs/cumulative_lppd_fmnist.pdf",
    multiple=1,
)
# %%
cumulative_lppd_fmnist = jnp.load("../../data/fireball/LppdCum_sampling_ionosphere.npz")
running_lppd = cumulative_lppd_fmnist["mean_seq_lppd"]
sd_running_lppd = cumulative_lppd_fmnist["std_seq_lppd"]
plot_cumulative_lppd(
    running_lppd=running_lppd,
    sd_running_lppd=sd_running_lppd,
    iterations=running_lppd.shape[0],
    output_path="../../results/cumulative_lppd_fireballs/cumulative_lppd_ionosphere.pdf",
    multiple=1,
)
# %%
cumulative_lppd_fmnist = jnp.load("../../data/fireball/LppdCum_sampling_bikesharing.npz")
running_lppd = cumulative_lppd_fmnist["mean_seq_lppd"]
sd_running_lppd = cumulative_lppd_fmnist["std_seq_lppd"]
plot_cumulative_lppd(
    running_lppd=running_lppd,
    sd_running_lppd=sd_running_lppd,
    iterations=running_lppd.shape[0],
    output_path="../../results/cumulative_lppd_fireballs/cumulative_lppd_bikesharing.pdf",
    multiple=1,
)
# %%
