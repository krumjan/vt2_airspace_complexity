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
lmmm = airspace(id="LMMM", volume=nm_airspaces["LMMMALL"])
lmmm.plot()

# %% [markdown]
# ### Processing ADS-B data

# %%
# Fetching of ADS-B data
lmmm.data_fetch(
    start_date="2019-01-01",
    end_date="2020-01-01",
)

# %%
# Preprocessing ADS-B data
lmmm.data_preprocess()
# Visualisation of airspace and trajectories before pre-processing
fig1 = lmmm.plot(traj_sample=True, traj_num=1000, reduced=False)
fig1.show()
# Visualisation of airspace and trajectories after pre-processing
fig2 = lmmm.plot(traj_sample=True, traj_num=1000, reduced=True)
fig2.show()

# %% [markdown]
# ### Determine hourly traffic load

# %%
# Generation of dataframe containing hourly entry counts
lmmm.hourly_generate_df()
# Plotting of heatmap of hourly count
fig = lmmm.hourly_heatmap()
fig.show()

# %% [markdown]
# ### Reduction to low traffic

# %%
# Generation of heatmap-like plot
fig1 = lmmm.hourly_heatmap_low(reference_type="max_perc", reference_value=0.45)
fig1.show()
# Generation of multiple boxplots
fig2 = lmmm.hourly_boxplots(reference_type="max_perc", reference_value=0.45)
fig2.show()
# Generation of Cumulative distribution function
fig3 = lmmm.hourly_cdf(reference_type="max_perc", reference_value=0.45)
fig3.show()

# %% [markdown]
# ### Simulation runs

# %% [markdown]
# #### identical intervals

# %%
# 120 seconds intervals
lmmm.simulation_monte_carlo_run(
    duration=24, interval=120, runs=10000, max_process=120, max_reads=80
)
fig = lmmm.simulation_monte_carlo_plot_histogram(
    duration=24, interval=120, ci=0.9
)
fig.show()

# %%
# 100 seconds intervals
lmmm.simulation_monte_carlo_run(
    duration=24, interval=100, runs=10000, max_process=120, max_reads=80
)
fig = lmmm.simulation_monte_carlo_plot_histogram(
    duration=24, interval=100, ci=0.9
)
fig.show()

# %%
# 80 seconds intervals
lmmm.simulation_monte_carlo_run(
    duration=24, interval=80, runs=10000, max_process=120, max_reads=80
)
fig = lmmm.simulation_monte_carlo_plot_histogram(
    duration=24, interval=80, ci=0.9
)
fig.show()

# %%
# 60 seconds intervals
lmmm.simulation_monte_carlo_run(
    duration=24, interval=60, runs=10000, max_process=120, max_reads=80
)
fig = lmmm.simulation_monte_carlo_plot_histogram(
    duration=24, interval=60, ci=0.9
)
fig.show()

# %%
# 40 seconds intervals
lmmm.simulation_monte_carlo_run(
    duration=24, interval=40, runs=10000, max_process=120, max_reads=80
)
fig = lmmm.simulation_monte_carlo_plot_histogram(
    duration=24, interval=40, ci=0.9
)
fig.show()

# %%
# 20 seconds intervals
lmmm.simulation_monte_carlo_run(
    duration=24, interval=20, runs=10000, max_process=120, max_reads=80
)
fig = lmmm.simulation_monte_carlo_plot_histogram(
    duration=24, interval=20, ci=0.9
)
fig.show()

# %% [markdown]
# #### Traffic volume dependant intervals

# %%
# determination of intervals corresponding to peak and average traffic
hourly_df = pd.read_parquet("../data/LMMM/04_hourly/hourly_df.parquet")
all_traf = Traffic.from_file(
    "../data/LMMM/03_preprocessed/preprocessed_all_red.parquet"
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
lmmm.simulation_monte_carlo_run(
    duration=24, interval=int_avg, runs=10000, max_process=120, max_reads=80
)
fig = lmmm.simulation_monte_carlo_plot_histogram(
    duration=24, interval=262, ci=0.9
)
fig.show()

# %%
# peak intervals
lmmm.simulation_monte_carlo_run(
    duration=24, interval=int_peak, runs=10000, max_process=120, max_reads=80
)
fig = lmmm.simulation_monte_carlo_plot_histogram(
    duration=24, interval=77, ci=0.9
)
fig.show()

# %% [markdown]
# ### Visualisation of complexity maps

# %%
import pickle

with open(
    "../data/LMMM/06_monte_carlo/24_20/cube_counts_aggregated.pkl", "rb"
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
fig = lmmm.simulation_monte_carlo_plot_map(
    duration=24, interval=120, alt_low=36500, zoom=5.7
)
fig.show()

# %%
fig = lmmm.simulation_monte_carlo_plot_map(
    duration=24, interval=120, alt_low=38500, zoom=5.7
)
fig.show()
