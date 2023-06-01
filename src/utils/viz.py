import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


def hourly_overview(
    hourly_df: pd.DataFrame,
    reference_type: str = None,
    reference_value: float = 0.5,
) -> plotly.graph_objs._figure.Figure:
    """
    Generate a plotly figure showing the distribution of hourly aircraft count .

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

    # print(threshold)

    # generate subplots containing four plots
    fig, axes = plt.subplots(2, 2)

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
    axes[1, 1].set_title("Distribution of hourly aircraft count by month")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Hourly aircraft count")
    axes[1, 1].grid(
        axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.5
    )

    if reference_type is not None:
        axes[0, 0].axhline(threshold, ls="--", color="red", linewidth=1)
        axes[0, 1].axhline(threshold, ls="--", color="red", linewidth=1)
        axes[1, 0].axhline(threshold, ls="--", color="red", linewidth=1)
        axes[1, 1].axhline(threshold, ls="--", color="red", linewidth=1)

    # Return the figure
    return fig


def yearly_heatmap(
    hourly_df: pd.DataFrame,
) -> plotly.graph_objs._figure.Figure:
    """
    Generate a yearly heatmap of the aircraft count. The function returns a plotly
    figure representing the aircraft count per hour of the day and day of the year.

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

    # Generate a plotly density plot of the variable 'ac_count'
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

    # update labels
    fig.update_xaxes(title_text="Day of the year")
    fig.update_yaxes(title_text="Hour of the day")
    fig.update_layout(title_text="Hourly aircraft entry count")
    fig.update_layout(coloraxis_colorbar=dict(title="AC count"))

    # update ticks
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=10)
    fig.update_yaxes(tickmode="linear", tick0=0, dtick=1)

    return fig


def heatmap_low_hour(
    hourly_df: pd.DataFrame, reference_type: str, reference_value: float = 0.5
) -> plotly.graph_objs._figure.Figure:
    """_summary_

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

    # Adjust axis ranges, ticks and labels
    fig.update_xaxes(range=[0.5, 365.5])
    fig.update_yaxes(range=[0.5, 24.5])
    fig.update_layout(legend={"itemsizing": "constant"})
    fig.update_layout(legend_title_text="Low-traffic hour")
    fig.update_layout(title_text="Distribution of low traffic hours")
    fig.update_xaxes(title_text="Day of the year")
    fig.update_yaxes(title_text="Hour of the day")
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=10)
    fig.update_yaxes(tickmode="linear", tick0=0, dtick=1)

    return fig
