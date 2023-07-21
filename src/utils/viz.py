import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import shapely
import plotly
import plotly.express as px
import plotly.graph_objects as go


def yearly_heatmap(
    hourly_df: pd.DataFrame,
) -> plotly.graph_objs._figure.Figure:
    """
    Generate a heatmap-like plot of the aircraft count over time. The function returns a
    plotly figure showing the aircraft count per hour of the day and day of the year,
    color-coding the count.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Dataframe containing the hourly aircraft count. Requires the columns
        'day_of_year', 'hour_of_day' and 'ac_count'.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Plotly figure containing the yearly heatmap.
    """

    # Generate a plotly heatmap plot of the variable 'ac_count'
    fig = px.density_heatmap(
        hourly_df,
        x="day_of_year",
        y="hour_of_day",
        z="ac_count",
        histfunc="avg",
        nbinsx=365,
        nbinsy=24,
        color_continuous_scale="jet",
    )

    # Update labels
    fig.update_xaxes(title_text="Day of the year")
    fig.update_yaxes(title_text="Hour of the day")
    fig.update_layout(title_text="Hourly aircraft entry count")
    fig.update_layout(coloraxis_colorbar=dict(title="entry count"))
    fig.update_layout(margin=dict(l=110, r=50, b=0, t=35), title_x=0.5)

    # Update ticks
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=10)
    fig.update_yaxes(tickmode="linear", tick0=0, dtick=1)

    # Return the figure
    return fig


def heatmap_low_hour(
    hourly_df: pd.DataFrame, reference_type: str, reference_value: float = 0.5
) -> plotly.graph_objs._figure.Figure:
    """
    Generate a plotly figure showing the distribution of low traffic hours. The function
    returns a plotly figure representing the distribution of low traffic hours per hour
    of the day and day of the year. The threshold for low traffic hours can be set by
    trough the reference_type and reference_value parameters.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Dataframe containing the hourly aircraft count. Requires the columns
        'day_of_year', 'hour_of_day' and 'ac_count'.
    reference_type : str
        Reference type to use for the threshold. Can be 'mean', 'median', 'quantile' or
        'max_perc'.
    reference_value : float, optional
        For type quantile the quantile to use and for type max_perc the percentage of
        the max observed hourly count to use as threshold, by default 0.5

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Plotly figure showing the low hours.

    Raises
    ------
    ValueError
        If the reference type is not recognized.
    """

    # Determine the threshold
    if reference_type == "mean":
        threshold = hourly_df["ac_count"].mean()
    elif reference_type == "median":
        threshold = hourly_df["ac_count"].median()
    elif reference_type == "quantile":
        threshold = hourly_df["ac_count"].quantile(reference_value)
    elif reference_type == "max_perc":
        threshold = hourly_df["ac_count"].max() * reference_value
    else:
        raise ValueError(
            f"Reference type {reference_type} not recognized. "
            "Please use 'mean', 'median', 'quantile' or 'max_perc'."
        )

    # Create a new column with a 1 if the hourly count is below the threshold
    hourly_df["below_th"] = (hourly_df["ac_count"] < threshold).astype(int)

    # Generate a plotly density plot of the variable 'below_th'
    my_colorsc = [[0, "salmon"], [1, "lightgreen"]]
    fig = px.density_heatmap(
        hourly_df,
        x="day_of_year",
        y="hour_of_day",
        z="below_th",
        histfunc="avg",
        nbinsx=365,
        nbinsy=24,
        color_continuous_scale=my_colorsc,
    )
    fig.update_coloraxes(showscale=False)

    # Adjust the legend
    fig.add_trace(
        go.Scatter(
            x=[-5],
            y=[0],
            mode="markers",
            marker_symbol="square",
            marker=dict(color="lightgreen", size=10),
            name="Yes",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[-5],
            y=[0],
            mode="markers",
            marker_symbol="square",
            marker=dict(color="salmon", size=10),
            name="No",
        )
    )

    # Improve readability
    fig.update_xaxes(range=[0.5, 365.5])
    fig.update_yaxes(range=[0.5, 24.5])
    fig.update_layout(legend={"itemsizing": "constant"})
    fig.update_layout(legend_title_text="Low-traffic hour")
    fig.update_layout(
        title_text="   Distribution of low traffic hours", title_x=0.5
    )
    fig.update_xaxes(title_text="Day of the year")
    fig.update_yaxes(title_text="Hour of the day")
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=10)
    fig.update_yaxes(tickmode="linear", tick0=0, dtick=1)
    fig.update_layout(margin=dict(l=110, r=50, b=0, t=35))

    # Return the figure
    return fig


def hourly_boxplots(
    hourly_df: pd.DataFrame,
    reference_type: str = None,
    reference_value: float = 0.5,
) -> plotly.graph_objs._figure.Figure:
    """
    Generate a plotly figure showing the distribution of hourly aircraft count for the
    data separated by hour of the day, day of the week, day of the month and month. The
    returned figure contains four subplots, each showing the distribution of the hourly
    aircraft count for the data separated by one of the four variables as a multiple
    boxplot. The threshold for low traffic hours, which is also visualised as a line in
    all four subplots, can be adjusted by the reference_type and reference_value
    parameters.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Dataframe containing the hourly aircraft count. Requires the columns
        'day_of_year', 'hour_of_day' and 'ac_count'.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Plotly figure showing the seasonality of the aircraft count.
    """

    # Determine the threshold
    if reference_type == None:
        pass
    elif reference_type == "mean":
        threshold = hourly_df["ac_count"].mean()
    elif reference_type == "median":
        threshold = hourly_df["ac_count"].median()
    elif reference_type == "quantile":
        threshold = hourly_df["ac_count"].quantile(reference_value)
    elif reference_type == "max_perc":
        threshold = hourly_df["ac_count"].max() * reference_value
    else:
        raise ValueError(
            f"Reference type {reference_type} not recognized. "
            "Please use 'mean', 'median', 'quantile' or 'max_perc'."
        )

    # Generate figure with four subplots
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(2000 / 96, 1200 / 96)

    # Plot the distribution ot the hourly aircraft count per hour of the day
    sns.boxplot(
        ax=axes[0, 0],
        x="hour_of_day",
        y="ac_count",
        data=hourly_df,
        showfliers=True,
        color="lightblue",
        notch=True,
    )
    axes[0, 0].set_title(
        "Distribution of hourly aircraft count by hour of the day"
    )
    axes[0, 0].set_xlabel("Hour of day")
    axes[0, 0].set_ylabel("Hourly aircraft count")
    axes[0, 0].grid(
        axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.5
    )

    # Plot the distribution ot the hourly aircraft count per day of the week
    hourly_df["weekday"] = hourly_df["weekday"].str[0:3]
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.boxplot(
        ax=axes[0, 1],
        x="weekday",
        y="ac_count",
        data=hourly_df,
        order=weekday_order,
        showfliers=True,
        color="lightblue",
        notch=True,
    )
    axes[0, 1].set_title(
        "Distribution of hourly aircraft count by day of the week"
    )
    axes[0, 1].set_xlabel("Day of the week")
    axes[0, 1].set_ylabel("Hourly aircraft count")
    axes[0, 1].grid(
        axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.5
    )

    # Plot the distribution ot the hourly aircraft count per day of the month
    sns.boxplot(
        ax=axes[1, 0],
        x="day_of_month",
        y="ac_count",
        data=hourly_df,
        showfliers=True,
        color="lightblue",
        notch=True,
    )
    axes[1, 0].set_title(
        "Distribution of hourly aircraft count by day of the month"
    )
    axes[1, 0].set_xlabel("Day of the month")
    axes[1, 0].set_ylabel("Hourly aircraft count")
    axes[1, 0].grid(
        axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.5
    )

    # Plot the distribution ot the hourly aircraft count per month
    sns.boxplot(
        ax=axes[1, 1],
        x="month",
        y="ac_count",
        data=hourly_df,
        showfliers=True,
        color="lightblue",
        notch=True,
    )

    # Add a horizontal line to indicate the threshold to each subplot
    if reference_type is not None:
        axes[0, 0].axhline(threshold, ls="--", color="red", linewidth=1)
        axes[0, 1].axhline(threshold, ls="--", color="red", linewidth=1)
        axes[1, 0].axhline(threshold, ls="--", color="red", linewidth=1)
        axes[1, 1].axhline(threshold, ls="--", color="red", linewidth=1)

    # Set the figure layout
    axes[1, 1].set_title("Distribution of hourly aircraft count by month")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Hourly aircraft count")
    axes[1, 1].grid(
        axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.5
    )

    # Return the figure
    return fig


def cumulative_distribution(
    hourly_df: pd.DataFrame,
    reference_type: str = None,
    reference_value: float = 0.5,
) -> plotly.graph_objs._figure.Figure:
    """
    Returns a cumulative distribution function plot for the hourly entry counts of
    the airspace with additional lines showing a set threshold. The threshold can be
    set using the reference type and value and is also indicated in the plot.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Dataframe containing the hourly aircraft count. Requires the columns
        'day_of_year', 'hour_of_day' and 'ac_count'.
    reference_type : str
        Reference type to use for the threshold. Can be 'mean', 'median', 'quantile' or
        'max_perc'.
    reference_value : float, optional
        For type quantile the quantile to use and for type max_perc the percentage of
        the max observed hourly count to use as threshold, by default 0.5

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A plot of the cumulative distribution of the hourly entry counts with lines
        representing the threshold value.

    Raises
    ------
    ValueError
        If the reference type is not recognized.
    """
    # Determine the threshold
    if reference_type == "mean":
        threshold = hourly_df["ac_count"].mean()
    elif reference_type == "median":
        threshold = hourly_df["ac_count"].median()
    elif reference_type == "quantile":
        threshold = hourly_df["ac_count"].quantile(reference_value)
    elif reference_type == "max_perc":
        threshold = hourly_df["ac_count"].max() * reference_value
    else:
        raise ValueError(
            f"Reference type {reference_type} not recognized. "
            "Please use 'mean', 'median', 'quantile' or 'max_perc'."
        )

    # Create figure and plot the cumulative distribution
    x = np.sort(hourly_df["ac_count"])
    y = np.arange(1, len(x) + 1) / len(x)
    fig = px.line(
        x=x,
        y=y,
        title="Cumulative distribution function of hourly entry counts",
    )

    # Calculate quantile corresponding to threshold
    quantile = (
        stats.percentileofscore(hourly_df["ac_count"], threshold)
    ) / 100

    # Add lines representing the threshold and matching quantile
    fig.add_vline(
        x=threshold,
        line_width=1.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold value",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=quantile,
        line_width=1.5,
        line_dash="dash",
        line_color="red",
    )

    # Improve readability
    fig.update_layout(margin=dict(l=0, r=50, b=0, t=35))
    fig.update_xaxes(title_text="Hourly aircraft entry count")
    fig.update_yaxes(title_text="Proportion of total hours")
    fig.update_layout(title_x=0.5)

    # Return the figure
    return fig


def occurence_histogram(occ_list: list, ci: float) -> matplotlib.figure.Figure:
    """
    Plots a histogram of the number of occurences for each run conducted as part of the
    Monte Carlo simulation and adds a line for the mean and 90% confidence interval of
    the mean.

    Parameters
    ----------
    occ_list : list
        List containing the total occurences for each run conducted as part of the Monte
        Carlo simulation
    ci : float
        Confidence interval to use for the confidence interval of the mean (e.g. 0.9 for
        a 90% confidence interval)

    Returns
    -------
    _type_
        Histogram plot of the number of occurences for each run conducted as part of
        the Monte Carlo simulation with a line for the mean and 90% confidence interval
        of the mean.
    """

    # Create histogram plot
    ax = sns.histplot(data=occ_list, bins=100)
    fig = ax.get_figure()
    fig.set_size_inches(18, 6)

    # Add labels
    ax.set_title(
        "Histogram of number of total occurences for Monte Carlo runs",
        pad=20,
    )
    ax.set_xlabel("Number of occurences")
    ax.set_ylabel("Count")

    # Compute mean with confidence interval
    mean = np.mean(occ_list)
    # lower_ci, upper_ci = stats.norm.interval(
    #     ci, loc=np.mean(occ_list), scale=np.std(occ_list)
    # )
    # lower_ci = max(lower_ci, 0)
    df = pd.DataFrame({"occurences": occ_list})
    lower_ci = df["occurences"].quantile((1 - ci) / 2)
    upper_ci = df["occurences"].quantile(1 - (1 - ci) / 2)

    # Plot lines for mean, lower and upper confidence intervals
    ax.axvline(mean, color="red", linestyle="dashed", linewidth=1)
    ax.text(
        mean,
        ax.get_ylim()[1] * 1,
        f"{mean:.2f}",
        horizontalalignment="center",
        verticalalignment="bottom",
        color="red",
    )
    ax.axvline(lower_ci, color="red", linestyle="dashed", linewidth=1)
    ax.text(
        lower_ci,
        ax.get_ylim()[1] * 1,
        f"{lower_ci:.2f}",
        horizontalalignment="center",
        verticalalignment="bottom",
        color="red",
    )
    ax.axvline(upper_ci, color="red", linestyle="dashed", linewidth=1)
    ax.text(
        upper_ci,
        ax.get_ylim()[1] * 1,
        f"{upper_ci:.2f}",
        horizontalalignment="center",
        verticalalignment="bottom",
        color="red",
    )

    # Return histogram as figure
    return fig


def occurence_heatmap(
    df: pd.DataFrame,
    airspace: shapely.geometry.polygon.Polygon,
    zoom: int = 10,
) -> matplotlib.figure.Figure:
    """
    Generates a heatmap showing the number of occurences for each grid cell in the
    airspace for the given dataframe which should contain the average occurence count
    along with information about the grid cell's location for one altitude layer.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the average occurence count for each grid cell on one
        altitude layer along with information about the grid cell's location. The
        dataframe should have the following columns: 'lat_min', 'lat_max', 'lon_min',
        'lon_max', 'alt_min', 'alt_max', 'count'.
    airspace : shapely.geometry.polygon.Polygon
        Shapely polygon representing the airspace for which the heatmap should be
        generated.
    zoom : int, optional
        Zoom level to use for the heatmap, by default 10

    Returns
    -------
    matplotlib.figure.Figure
        Heatmap showing the number of occurences for each grid cell in the airspace for
        the given dataframe.
    """

    # Generate heatmap plot
    fig = go.Figure(go.Scattermapbox())
    fig.update_layout(
        mapbox_style="mapbox://styles/jakrum/clgqc6e8u00it01qzgtb4gg1z",
        mapbox_accesstoken=(
            "pk.eyJ1IjoiamFrcnVtIiwiYSI6ImNsZ3FjM3BiMzA3dzYzZHMzNHRkZnFtb3EifQ."
            "ydDFlmylEcRCkRLWXqL1Cg"
        ),
        showlegend=False,
        height=1000,
        width=1300,
        margin={"l": 0, "b": 0, "t": 0, "r": 0},
        mapbox_center_lat=airspace.centroid.xy[1][0],
        mapbox_center_lon=airspace.centroid.xy[0][0],
        mapbox_zoom=zoom,
    )

    # Define colorscale
    cmap = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(
            vmin=df["count"].min(), vmax=df["count"].max()
        ),
        cmap="plasma",
    )

    # Function to convert value to color from colorscale
    def colorscale(value):
        color = cmap.to_rgba(value)
        color = [int(c * 255) for c in color]
        color_hex = "#{:02x}{:02x}{:02x}".format(*color[:3])
        return color_hex

    # For each rectangle in the dataframe, add a rectangle to the plot, color accoring
    # to the count value and add the count as text at the center of the rectangle
    for row in df.iterrows():
        fig.add_trace(
            go.Scattermapbox(
                lat=[row[1][0], row[1][0], row[1][1], row[1][1], row[1][0]],
                lon=[row[1][2], row[1][3], row[1][3], row[1][2], row[1][2]],
                mode="lines",
                line=dict(width=2, color="black"),
                fill="toself",
                fillcolor=colorscale(row[1][6]),
                # text=f"count: {row[1][6]}",
                opacity=0.2,
                name="Rectangle",
            )
        )
        # Calculate center point of rectangle
        center_lat = (row[1][0] + row[1][1]) / 2
        center_lon = (row[1][2] + row[1][3]) / 2
        # Add count as text at center point
        fig.add_trace(
            go.Scattermapbox(
                lat=[center_lat],
                lon=[center_lon],
                mode="text",
                text=[round(row[1][6], 2)],
                textfont=dict(size=8, color="grey"),
                textposition="middle center",
                showlegend=False,
            )
        )

    # Add airspace shape to the plot
    lons, lats = airspace.exterior.xy
    trace = go.Scattermapbox(
        mode="lines",
        lat=list(lats),
        lon=list(lons),
        line=dict(width=2, color="red"),
    )
    fig.add_trace(trace)

    # Manually add colorbar to the plot
    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale="plasma",
            showscale=True,
            cmin=df["count"].min(),
            cmax=df["count"].max(),
            colorbar=dict(thickness=20, outlinewidth=0, title="occurences"),
        ),
        hoverinfo="none",
    )
    fig["layout"]["showlegend"] = False
    fig.add_trace(colorbar_trace)

    # Improve layout of plot and add title
    alt_low = df.alt_min.min()
    fig.update_layout(
        title_text=f"Altitude level {alt_low}ft - {alt_low+1000}ft",
        title_x=0.5,
        margin={"l": 0, "b": 0, "t": 40, "r": 0},
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Return plot
    # fig.write_image("figure.png") # Alternative to directly save the figure
    return fig
