# %% [markdown]
# ### Import libraries

# %%
# change of working directory to src
import os

os.chdir("/cluster/home/krum/github/VT2_airspace_complexity/src")

import pandas as pd
from traffic.core import Traffic
from complexity.airspace import airspace
from traffic.data import nm_airspaces
from utils import viz as viz

# %% [markdown]
# ### Defining airspace

# %%
# Deinition of airspace
ldzo = airspace(id="LDZO", volume=nm_airspaces["LDZOACCX"])
# ldzo.plot()

# %% [markdown]
# ### Processing ADS-B data

# %%
# Fetching of ADS-B data
ldzo.data_fetch(
    start_date="2019-01-01",
    end_date="2020-01-01",
)

# %%
# Preprocessing ADS-B data
ldzo.data_preprocess()
# Visualisation of airspace and trajectories before pre-processing
fig1 = ldzo.plot(traj_sample=True, traj_num=1000, reduced=False)
fig1.show()
# Visualisation of airspace and trajectories after pre-processing
fig2 = ldzo.plot(traj_sample=True, traj_num=1000, reduced=True)
fig2.show()

# %% [markdown]
# ### Determine hourly traffic load

# %%
# Generation of dataframe containing hourly entry counts
ldzo.hourly_generate_df()
# Plotting of heatmap of hourly count
fig = ldzo.hourly_heatmap()
fig.show()

# %% [markdown]
# ### Reduction to low traffic

# %%
# Generation of heatmap-like plot
fig1 = ldzo.hourly_heatmap_low(reference_type="max_perc", reference_value=0.45)
fig1.show()
# Generation of multiple boxplots
fig2 = ldzo.hourly_boxplots(reference_type="max_perc", reference_value=0.45)
fig2.show()
# Generation of Cumulative distribution function
fig3 = ldzo.hourly_cdf(reference_type="max_perc", reference_value=0.45)
fig3.show()

# %% [markdown]
# ### Simulation runs

# %% [markdown]
# #### identical intervals

# %%
# 120 seconds intervals
ldzo.simulation_monte_carlo_run(
    duration=24, interval=120, runs=10000, max_process=120, max_reads=15
)
fig = ldzo.simulation_monte_carlo_plot_histogram(
    duration=24, interval=120, ci=0.9
)
fig.show()

# %%
# 100 seconds intervals
ldzo.simulation_monte_carlo_run(
    duration=24, interval=100, runs=10000, max_process=120, max_reads=15
)
fig = ldzo.simulation_monte_carlo_plot_histogram(
    duration=24, interval=100, ci=0.9
)
fig.show()

# %%
# 80 seconds intervals
ldzo.simulation_monte_carlo_run(
    duration=24, interval=80, runs=10000, max_process=120, max_reads=15
)
fig = ldzo.simulation_monte_carlo_plot_histogram(
    duration=24, interval=80, ci=0.9
)
fig.show()

# %%
# 60 seconds intervals
ldzo.simulation_monte_carlo_run(
    duration=24, interval=60, runs=10000, max_process=120, max_reads=15
)
fig = ldzo.simulation_monte_carlo_plot_histogram(
    duration=24, interval=60, ci=0.9
)
fig.show()

# %%
# 40 seconds intervals
ldzo.simulation_monte_carlo_run(
    duration=24, interval=40, runs=10000, max_process=120, max_reads=15
)
fig = ldzo.simulation_monte_carlo_plot_histogram(
    duration=24, interval=40, ci=0.9
)
fig.show()

# %%
# 20 seconds intervals
ldzo.simulation_monte_carlo_run(
    duration=24, interval=20, runs=10000, max_process=120, max_reads=15
)
fig = ldzo.simulation_monte_carlo_plot_histogram(
    duration=24, interval=20, ci=0.9
)
fig.show()

# %% [markdown]
# #### Traffic volume dependant intervals

# %%
# determination of intervals corresponding to peak and average traffic
hourly_df = pd.read_parquet("../data/LDZO/04_hourly/hourly_df.parquet")
all_traf = Traffic.from_file(
    "../data/LDZO/03_preprocessed/preprocessed_all_red.parquet"
)
int_peak = (60 * 60) / hourly_df.ac_count.max()
int_avg = (365 * 24 * 60 * 60) / len(all_traf)

# average interval
int_avg = round((365 * 24 * 60 * 60) / len(all_traf))
print(int_avg)

# peak interval
int_peak = round((60 * 60) / hourly_df.ac_count.max())
print(int_peak)

# %%
# average intervals
ldzo.simulation_monte_carlo_run(
    duration=24, interval=48, runs=10000, max_process=120, max_reads=13
)
fig = ldzo.simulation_monte_carlo_plot_histogram(
    duration=24, interval=int_avg, ci=0.9
)
fig.show()

# %%
# peak intervals
ldzo.simulation_monte_carlo_run(
    duration=24, interval=16, runs=10000, max_process=120, max_reads=13
)
fig = ldzo.simulation_monte_carlo_plot_histogram(
    duration=24, interval=int_peak, ci=0.9
)
fig.show()

# %% [markdown]
# ### Visualisation of layers

# %%
import pickle

with open(
    "../data/LDZO/06_monte_carlo/24_20/cube_counts_aggregated.pkl", "rb"
) as handle:
    b = pickle.load(handle)

result_dict = {}
for key, values in b.items():
    average = sum(values) / len(values)
    result_dict[key] = average

sorted_keys = sorted(result_dict, key=result_dict.get, reverse=True)

for key in sorted_keys:
    print(key, result_dict[key])

# %%
fig = ldzo.simulation_monte_carlo_plot_map(
    duration=24, interval=20, alt_low=35500, zoom=6.5
)
fig.show()

# %%
fig = ldzo.simulation_monte_carlo_plot_map(
    duration=24, interval=20, alt_low=36500, zoom=6.5
)
fig.show()
