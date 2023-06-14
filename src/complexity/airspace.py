import glob
import math
import os
import pickle
import random
import multiprocessing as mp
from pathlib import Path
from typing import Union

import matplotlib
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import shapely
from shapely.geometry import Point
from tqdm.auto import tqdm

import traffic
from traffic.core import Traffic

from utils import adsb as util_adsb
from utils import geo as util_geo
from utils import general as util_general
from utils import viz as viz


class airspace:
    def __init__(
        self,
        id: str,
        volume: Union[
            traffic.core.airspace.Airspace,
            tuple[shapely.geometry.polygon.Polygon, float, float],
        ],
    ) -> None:
        """
        Class for the definition of an airspace. The airspace is defined by an id and a
        volume. For the volume it is possible ot either pass a traffic airspace object
        or a tuple containing a shapely polygon and two floats (lower altitude  limit of
        airspace volumne, upper altitude limit of airspace volume).

        Parameters
        ----------
        id : str
            identifier for the airspace (e.g. 'LSAZ' or 'MUAC')
        volume : Union[traffic.core.airspace.Airspace,
                       tuple[shapely.geometry.polygon.Polygon, float, float]]
            Tree dimensional volume definition of the airspace. Can be defined either
            with a traffic.core.airspace.Airspace object or by a tuple of a
            (shapely.geometry.polygon.Polygon, lower altitude, upper altitude). For the
            latter the altitudes are given in feet.
        """

        # If volume is a traffic airspace object, extract the relevant information and
        # initialise the airspace instance
        if type(volume) == traffic.core.airspace.Airspace:
            self.id = id
            self.shape = volume.shape
            self.lat_max = volume.shape.bounds[3]
            self.lat_min = volume.shape.bounds[1]
            self.lon_max = volume.shape.bounds[2]
            self.lon_min = volume.shape.bounds[0]
            self.lat_cen = volume.shape.centroid.y
            self.lon_cen = volume.shape.centroid.x
            self.alt_min = int(volume.elements[0].lower * 100)
            self.alt_max = int(min(volume.elements[-1].upper * 100, 45000))

        # If volume is a tuple of a shapely polygon and two floats defining the upper
        # and lower altitude bounds, extract the relevant information and initialise
        # the airspace instance
        else:
            self.id = id
            self.shape = volume[0]
            alt_lo = volume[1]
            alt_up = volume[2]
            self.lat_max = self.shape.bounds[3]
            self.lat_min = self.shape.bounds[1]
            self.lon_max = self.shape.bounds[2]
            self.lon_min = self.shape.bounds[0]
            self.lat_cen = self.shape.centroid.y
            self.lon_cen = self.shape.centroid.x
            self.alt_min = alt_lo
            self.alt_max = min(alt_up, 45000)

    def plot(
        self,
        traj_sample: bool = False,
        traj_num: int = 200,
        reduced=True,
    ) -> go.Figure:
        """
        Generates a map plot of the airspace outline. If the parameter 'traj_sample' is
        set to True, a sample of the trajectories is added to the plot. The amount of
        trajectories is defined by the parameter 'traj_num'. If the parameter 'reduced'
        is set to True, the plotted sample trajectories are the ones that are reduced to
        the actual TMA extent. Otherwise, the sample trajectories span the entire
        rectangular download-boundary. To plot a sample of non-reduced trajectories, a
        prerequisite is that the function 'data_fetch' has been executed before. To plot
        a sample of reduced trajectories, the function 'data_preprocess' additionally
        needs to have been executed before.

        Parameters
        ----------
        traj_sample : bool, optional
            If set to True, a random sample of trajectories will be added to the plot,
            by default False
        traj_num : int, optional
            Amount of trajectories which are plotted if traj_sample is set to True, by
            default 100
        reduced : bool, optional
            If set to True, the plot will include sample trajectories that are reduced
            to the acutal TMA extent and otherwise sample trajectories that span the
            entire rectangular donwload-boundary, by default True

        Returns
        -------
        go.Figure
            Figure object containing the map plot of the airspace
        """

        # Create mapbox figure
        fig = go.Figure(go.Scattermapbox())
        fig.update_layout(
            mapbox_style="mapbox://styles/jakrum/clgqc6e8u00it01qzgtb4gg1z",
            mapbox_accesstoken="pk.eyJ1IjoiamFrcnVtIiwiYSI6ImNsZ3FjM3BiMzA3dzYzZHMzNHR"
            "kZnFtb3EifQ.ydDFlmylEcRCkRLWXqL1Cg",
            showlegend=False,
            height=400,
            width=800,
            margin={"l": 0, "b": 0, "t": 0, "r": 0},
            mapbox_center_lat=self.lat_cen,
            mapbox_center_lon=self.lon_cen,
            mapbox_zoom=7,
        )

        # When 'traj_sample' is True, add a sample of trajectories to the plot
        if traj_sample:
            # Define home path and import data
            home_path = util_general.get_project_root()
            # Load either reduced or non-reduced trajectories depending on the "reduced"
            # parameter
            if reduced:
                trajs = Traffic.from_file(
                    f"{home_path}/data/{self.id}/"
                    "03_preprocessed/preprocessed_all_tma.parquet"
                )
            else:
                trajs = Traffic.from_file(
                    f"{home_path}/data/{self.id}/"
                    "03_preprocessed/preprocessed_all_rec.parquet"
                )
            # Add the random sample of trajectories as lines to the plot
            ids = random.sample(
                trajs.data["flight_id"].unique().tolist(), traj_num
            )
            for traj in trajs[ids]:
                fig.add_trace(
                    go.Scattermapbox(
                        mode="lines",
                        lat=traj.data["latitude"],
                        lon=traj.data["longitude"],
                        line=dict(width=1, color="blue"),
                    )
                )

        # Add the airspace outline to the plot
        lons, lats = self.shape.exterior.xy
        trace = go.Scattermapbox(
            mode="lines",
            lat=list(lats),
            lon=list(lons),
            line=dict(width=4, color="red"),
        )
        fig.add_trace(trace)

        # Return the mapbox figure
        return fig

    def data_fetch(
        self,
        start_date: str,
        end_date: str,
    ) -> None:
        """
        Fetches all available ADS-B data for the airspace volume during the specified
        time period from OpenSky Networks historical database. The data is downloaded in
        daily chunks using eight parallel processes and subsequently saved as daily
        Traffic files in the folder 'data/cellspace_id/01_raw'. The data is additionally
        aggregated into monthly Traffic objects which are saved under
        'data/cellspace_id/02_combined'.

        Parameters
        ----------
        start_date : str
            start date for data collection period in the format 'YYYY-MM-DD'
        end_date : str
            end date for data collection period in the format 'YYYY-MM-DD'
        """

        # Define home path
        home_path = util_general.get_project_root()

        # Parallel data fetching and saving of daily chunks
        print("Fetching data...")
        util_adsb.download_adsb_para(
            start=start_date,
            stop=end_date,
            folder=f"{home_path}/data/{self.id}/01_raw",
            bounds=self.shape,
            lower=self.alt_min,
            upper=self.alt_max,
        )

        # Combining data to monthly files which are also saved
        print("Combining data...")
        util_adsb.combine_adsb(
            path_raw=f"{home_path}/data/{self.id}/01_raw",
            path_combined=f"{home_path}/data/{self.id}/02_combined",
        )

    def data_preprocess(self) -> None:
        """
        Preprocesses the montly data chuncks resulting from 'fetch_data' and saves them
        again as preprocessed monthly Traffic objects under 'data/cellspace_id/
        03_preprocessed/monthly'. The preprocessing includes the following steps:
            - Assigning a unique id to each trajectory
            - Removing invalid trajectories
            - Applying filtering to the trajectories in order to remove outliers
            - Resampling the trajectories to 5s intervals

        Afterwards the monthly Traffic objects are also combined to one aggregated
        Traffic object which is saved as 'data/cellspace_id/03_preprocessed/
        preprocessed_all_rec.parquet'. Since the data is downloaded in a rectangular
        shape extent beyond the actual airspace boundary, the aggregated Traffic object
        is also reduced to the actual airspace boundary extent and also saved under
        it under 'data/cellspace_id/03_preprocessed/preprocessed_all_red.parquet'.
        """

        # Define home path
        home_path = util_general.get_project_root()

        # Preprocess data in monthly chunks
        print("Preprocessing data in monthly chuks...")
        util_adsb.preprocess_adsb(
            path_get=f"{home_path}/data/{self.id}/02_combined",
            path_save=f"{home_path}/data/{self.id}/03_preprocessed/monthly/",
        )

        # Combine preprocessed data to one file
        print("Combining preprocessed data...")
        # Check whether file already exists
        check_file = Path(
            f"{home_path}/data/{self.id}/03_preprocessed/preprocessed_all_rec.parquet"
        )
        # If file does not exist, combine data
        if check_file.is_file() is False:
            # Get all monthly files
            files = glob.glob(
                f"{home_path}/data/{self.id}/03_preprocessed/monthly/*.parquet",
                recursive=True,
            )
            # Read all files and combine them to one dataframe
            dfs = [Traffic.from_file(file).data for file in tqdm(files)]
            all_df = pd.concat(dfs)
            # Save combined dataframe as Traffic object
            Traffic(all_df).to_parquet(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_rec.parquet"
            )

        # Reduce to actual airspace extent
        print("Reducing to airspace extent...")
        check_file = Path(
            f"{home_path}/data/{self.id}/03_preprocessed/preprocessed_all_red.parquet"
        )

        # If file does not exist, crop data to tma extent
        if check_file.is_file() is False:
            trajs = Traffic.from_file(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_rec.parquet"
            )
            trajs = trajs.clip(self.shape).eval(
                desc="clipping", max_workers=20
            )
            trajs.to_parquet(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_red.parquet"
            )

    def hourly_generate_df(self, return_df: bool = False) -> pd.DataFrame:
        """
        Generates a dataframe containing hourly aggregated traffic information of
        for the airspace. This information includes the entry count of aircraft into the
        airspace for the discrete one hour time intervals as well as a list of the
        corresponding flight_ids. The dataframe is saved as a parquet file under
        'data/cellspace_id/04_hourly' and can also directly be returned as a pandas
        dataframe by the function if 'return_df' is set to True.

        Parametersd
        ----------
        return_df : bool, optional
            If True, the dataframe will be returend by the function, by default False

        Returns
        -------
        pd.DataFrame
            Dataframe containing the hourly aggregated traffic information. Only
            returned if 'return_df' is set to True.
        """

        # Define home path
        home_path = util_general.get_project_root()

        # Check if file already exists
        check_file = Path(
            f"{home_path}/data/{self.id}/04_hourly/hourly_df.parquet"
        )
        # If file does not exist, run process to generate it
        if check_file.is_file() is False:
            # Load trajectory data
            trajs = Traffic.from_file(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_tma.parquet"
            )
            # Aggregate data by flight_id, keeping the minimum and maximum timestamp
            # which correspond to the airspace entry and exit time of the flight
            df = trajs.data
            df = (
                df.groupby("flight_id")["timestamp"]
                .agg(["min", "max"])
                .reset_index()
            )
            df = df.rename({"min": "in", "max": "out"}, axis=1)
            df["timestamp_entered_h"] = df["in"].dt.floor("h")
            df = df.drop(["in", "out"], axis=1)
            # Aggregate data by hour of entry, keeping the amount of flights and a
            # list of flight_ids
            hourly_users = df.groupby(["timestamp_entered_h"])[
                "flight_id"
            ].count()
            hourly_users.name = "ac_count"
            hourly_ids = df.groupby(["timestamp_entered_h"])[
                "flight_id"
            ].apply(list)
            hourly_ids.name = "flight_ids"
            hourly_df = pd.concat(
                [hourly_users, hourly_ids],
                axis=1,
            )
            hourly_df.reset_index(inplace=True)
            hourly_df = hourly_df.rename(
                {"timestamp_entered_h": "hour"}, axis=1
            )
            # Fill missing hours with count 0
            hourly_df = (
                hourly_df.set_index("hour")
                .resample("H")
                .asfreq()
                .fillna(0)
                .reset_index()
            )
            # Add additional columns containing information about the hour
            hourly_df["weekday"] = hourly_df["hour"].dt.day_name()
            hourly_df["month"] = hourly_df["hour"].dt.month
            hourly_df["hour_of_day"] = hourly_df["hour"].dt.hour + 1
            hourly_df["day_of_year"] = hourly_df["hour"].dt.dayofyear
            hourly_df["day_of_month"] = hourly_df["hour"].dt.day
            # rearange columns
            hourly_df = hourly_df[
                [
                    "hour",
                    "hour_of_day",
                    "weekday",
                    "day_of_month",
                    "month",
                    "day_of_year",
                    "ac_count",
                    "flight_ids",
                ]
            ]
            # Convert flight_ids to list (for cases with no flights in the hour)
            hourly_df["flight_ids"] = hourly_df["flight_ids"].apply(
                lambda x: [] if x == 0 else x
            )
            # Save hourly dataframe as parquet file
            if not os.path.exists(f"{home_path}/data/{self.id}/04_hourly/"):
                os.makedirs(f"{home_path}/data/{self.id}/04_hourly/")
            hourly_df.to_parquet(check_file)
            # Return dataframe if 'return_df' is set to True
            if return_df:
                hourly_df = pd.read_parquet(check_file)
                return hourly_df

        # If file aready existr, return it if 'return_df' is set to True
        else:
            if return_df:
                hourly_df = pd.read_parquet(check_file)
                return hourly_df

    def hourly_heatmap(self) -> go.Figure:
        """
        Generates a heatmap-like plot of the hourly aircraft entry count into the
        airspace. Prerequisite to run this function is the existence of a dataframe
        containing hourly aggregated data as created by the function 'get_hourly_df'.

        Returns
        -------
        go.Figure
            Plotly figure object containing the heatmap
        """

        # Define home path
        home_path = util_general.get_project_root()

        # Load hourly dataframe from parquet file
        hourly_df = pd.read_parquet(
            f"{home_path}/data/{self.id}/04_hourly/hourly_df.parquet"
        )

        # Return heatmap-like plot
        return viz.yearly_heatmap(hourly_df)

    def hourly_heatmap_low(
        self, reference_type: str = "max_perc", reference_value=0.4
    ) -> go.Figure:
        """
        Generates a heatmap-like plot showing the hours classified as low traffic hours.
        Prerequisite to run this function is the existence of a dataframe containing
        hourly aggregated which is created by the function 'get_hourly_df'. The
        threshold for the classification of low traffic hours can be set by the user but
        is by default set to 40% of the maximum traffic volume.

        Parameters
        ----------
        reference_type : str, optional
            Defines how the low-traffic threshold is defined. Can be either mean,
            median, quantile or max_perc, by default "max_perc"
        reference_value : float, optional
            For reference types quantile or max_perc this value defines the the value,
            ex. 0.4 for the 40th percentile or 40% of max traffic volume, by default 0.4

        Returns
        -------
        go.Figure
            heatmpa-like plot showing the hours classified as low traffic hours
        """
        # Define home path
        home_path = util_general.get_project_root()

        # Load pandas dataframe from parquet file
        hourly_df = pd.read_parquet(
            f"{home_path}/data/{self.id}/04_hourly/hourly_df.parquet"
        )

        # Create plot and return it
        return viz.heatmap_low_hour(hourly_df, reference_type, reference_value)

    def hourly_boxplots(
        self, reference_type: str = "max_perc", reference_value=0.4
    ) -> go.Figure:
        """
        Generates a plot containing four multiple boxplots showing the distribution of
        hourly traffic volume grouped by hour of day, day of week, day of month and
        month. In each plot a horizontal line is drawn to indicate the threshold for
        low-traffic hours. Prerequisite to run this function is the existence of a
        dataframe containing hourly aggregated which is created by the function
        'get_hourly_df'. The threshold for the classification of low traffic hours can
        be set by the user but is by default set to 40% of the maximum traffic volume.

        Parameters
        ----------
        reference_type : str, optional
            Defines how the low-traffic threshold is defined. Can be either mean,
            median, quantile or max_perc, by default "max_perc"
        reference_value : float, optional
            For reference types quantile or max_perc this value defines the the value,
            ex. 0.4 for the 40th percentile or 40% of max traffic volume, by default 0.4

        Returns
        -------
        go.Figure
            Plotly figure object containing the four multiple boxplots
        """
        # Define home path
        home_path = util_general.get_project_root()

        # Load pandas dataframe from parquet file
        hourly_df = pd.read_parquet(
            f"{home_path}/data/{self.id}/04_hourly/hourly_df.parquet"
        )

        # Create plot and return it
        return viz.hourly_boxplots(hourly_df, reference_type, reference_value)

    def hourly_cdf(
        self, reference_type: str = "max_perc", reference_value: float = 0.4
    ) -> plotly.graph_objs._figure.Figure:
        """
        returns a cumulative distribution plot of the hourly entry counts of the
        airspace instance. The function returns a plotly figure. The threshold can be
        set using the reference type and value and is also indicated in the plot.

        Parameters
        ----------
        reference_type : str
            Reference type to use for the threshold. Can be 'mean', 'median', 'quantile'
            or 'max_perc'.
        reference_value : float, optional
            For type quantile the quantile to use and for type max_perc the percentage
            of the max observed hourly count to use as threshold, by default 0.5

        Returns
        -------
        plotly.graph_objs._figure.Figure
            A plot of the cumulative distribution of the hourly entry counts with lines
            representing the threshold value.
        """

        # Define home path
        home_path = util_general.get_project_root()

        # Load pandas dataframe from parquet file
        hourly_df = pd.read_parquet(
            f"{home_path}/data/{self.id}/04_hourly/hourly_df.parquet"
        )

        # Create heatmap and return it
        return viz.cumulative_distribution(hourly_df, "max_perc", 0.4)

    def reduce_low_traffic(self, reference_type: str, reference_value: float):
        """
        Generates a set of trajectories that only contains the low traffic hours. The
        threshold for the classification of low traffic hours can be set by the user
        trough the reference type and value. Prerequisite to run this function is the
        existence of a dataframe containing hourly aggregated which is created by the
        function 'get_hourly_df'.

        Parameters
        ----------
        reference_type : str
            reference type to use for the threshold. Can be 'mean', 'median', 'quantile'
            or 'max_perc'.
        reference_value : float
            For type quantile the quantile to use and for type max_perc the percentage
            of the max observed hourly count to use as threshold

        Raises
        ------
        Exception
            Raised if the required trajectory file does not exits
        Exception
            Raised if the required hourly dataframe does not exits
        ValueError
            Raised if the reference type is not one of the allowed types
        """
        # Define home path
        home_path = util_general.get_project_root()

        # Only do the steps if the required file does not exist yet
        if not os.path.exists(
            f"{home_path}/data/{self.id}/"
            "05_low_traffic/trajs_tma_low.parquet"
        ):
            # If any of the required files does not exist, raise an exception
            if not os.path.exists(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_rec.parquet"
            ) or not os.path.exists(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_tma.parquet"
            ):
                raise Exception(
                    "Preprocessed trajectories have not been created yet. "
                    "Please run 'preprocess_data()' first."
                )
            if not os.path.exists(
                f"{home_path}/data/{self.id}/04_hourly/hourly_df.parquet"
            ):
                raise Exception(
                    "Hourly aggregated traffic information has not been created yet. "
                    "Please run 'get_hourly_df()' first."
                )

            # Load trajectory data and hourly aggregated traffic information
            hourly_df = pd.read_parquet(
                f"{home_path}/data/{self.id}/04_hourly/hourly_df.parquet"
            )
            trajs_rec = Traffic.from_file(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_rec.parquet"
            )
            trajs_tma = Traffic.from_file(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_tma.parquet"
            )

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

            # Determine hours below threshold
            hourly_df["below_th"] = hourly_df["ac_count"] < threshold

            # Get id of all flights during these low traffic hours
            low_ids = hourly_df[hourly_df.below_th == True].flight_ids
            low_ids_list = []
            for x in low_ids:
                if isinstance(x, list):
                    low_ids_list.append(x)
                else:
                    low_ids_list.append([x])
            low_ids = [item for sublist in low_ids_list for item in sublist]

            # Reduce trajectory data to low traffic hours
            trajs_rec_low = trajs_rec[low_ids]
            trajs_tma_low = trajs_tma[low_ids]

            # Save reduced trajectory data
            if not os.path.exists(
                f"{home_path}/data/{self.id}/05_low_traffic/"
            ):
                os.makedirs(f"{home_path}/data/{self.id}/05_low_traffic/")
            trajs_rec_low.to_parquet(
                f"{home_path}/data/{self.id}/05_low_traffic/trajs_rec_low.parquet"
            )
            trajs_tma_low.to_parquet(
                f"{home_path}/data/{self.id}/05_low_traffic/trajs_tma_low.parquet"
            )

    def cells_generate(self, dim: int = 5, alt_diff: int = 1000) -> None:
        """
        Generates a grid of cells for the airspace. The cells are saved as a list of
        tuples (lat, lon, alt_low, alt_high) in the attribute 'grid'.
        Parameters
        ----------
        dim : int, optional
            horizontal cell size [dim x dim] in nautical miles, by default 20
        alt_diff : int, optional
            height of the cells in feet, by default 3000
        """

        # Generate grid
        lats = [self.lat_max]
        lons = [self.lon_min]
        self.grid = []
        lat = self.lat_max
        lon = self.lon_min
        while lat > self.lat_min:
            lat = util_geo.new_pos_dist((lat, lon), dim, 180)[0]
            lats.append(lat)
        while lon < self.lon_max:
            lon = util_geo.new_pos_dist((lat, lon), dim, 90)[1]
            lons.append(lon)
        for i in range(len(lats) - 1):
            for j in range(len(lons) - 1):
                center_lat = (lats[i] + lats[i + 1]) / 2
                center_lon = (lons[j] + lons[j + 1]) / 2
                grid_pos = Point(center_lon, center_lat)
                if self.shape.contains(grid_pos):
                    self.grid.append(
                        (
                            lats[i],
                            lats[i + 1],
                            lons[j],
                            lons[j + 1],
                            center_lat,
                            center_lon,
                        )
                    )

        # Generate altitude intervals
        # Determine starting altitude (set to closest ...500 below alt_min)
        hundreds = int(str(self.alt_min)[-3:])
        if hundreds > 500:
            start_alt = self.alt_min - (hundreds - 500)
        elif hundreds < 500:
            start_alt = self.alt_min - (hundreds + 500)
        else:
            start_alt = self.alt_min

        # generate intervals based on starting altitude and altitude difference
        count = np.array(
            [*range(math.ceil((self.alt_max - start_alt) / alt_diff))]
        )
        alts_low = start_alt + alt_diff * count
        alts_high = start_alt + alt_diff * (count + 1)
        alts = np.array(list(zip(alts_low, alts_high)))
        self.levels = alts

        # Generate cubes from grid and altitude intervals
        self.cubes = []
        for level in self.levels:
            for idx, grid in enumerate(self.grid):
                self.cubes.append(
                    cube(
                        id=f"range_{level[0]}_grid_{idx}",
                        lat_max=grid[0],
                        lat_min=grid[1],
                        lon_max=grid[3],
                        lon_min=grid[2],
                        alt_low=level[0],
                        alt_high=level[1],
                    )
                )

    def cells_visualise(self) -> None:
        """
        Visualises the airspace grid and altitude levels. The function requires the
        attribute 'grid' to exist. If it does not exist, an error is raised.
        Raises
        ------
        ValueError
            Error is raised if the attribute 'grid' does not exist.
        """
        # Raise error if grid does not exist
        if not hasattr(self, "grid"):
            raise ValueError(
                "Grid does not exist. Please execute function "
                "generate_cells() first."
            )

        # Plot grid
        print("Grid:")
        print("-----------------------------")
        fig = go.Figure(go.Scattermapbox())
        fig.update_layout(
            mapbox_style="mapbox://styles/jakrum/clgqc6e8u00it01qzgtb4gg1z",
            mapbox_accesstoken="pk.eyJ1IjoiamFrcnVtIiwiYSI6ImNsZ3FjM3BiMzA3dzYzZHMzNHR"
            "kZnFtb3EifQ.ydDFlmylEcRCkRLWXqL1Cg",
            showlegend=False,
            height=800,
            width=800,
            margin={"l": 0, "b": 0, "t": 0, "r": 0},
            mapbox_center_lat=self.lat_cen,
            mapbox_center_lon=self.lon_cen,
            mapbox_zoom=6,
        )
        for pos in self.grid:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[pos[0], pos[0], pos[1], pos[1], pos[0]],
                    lon=[pos[2], pos[3], pos[3], pos[2], pos[2]],
                    mode="lines",
                    line=dict(width=2, color="blue"),
                    fill="toself",
                    fillcolor="rgba(0, 0, 255, 0.3)",
                    opacity=0.2,
                    name="Rectangle",
                )
            )

        # Add airspace shape to the plot
        lons, lats = self.shape.exterior.xy
        trace = go.Scattermapbox(
            mode="lines",
            lat=list(lats),
            lon=list(lons),
            line=dict(width=2, color="red"),
        )
        fig.add_trace(trace)
        fig.show()

        # Print vertical ranges
        print("Vertical ranges:")
        print("-----------------------------")
        for idx, level in enumerate(self.levels[::-1]):
            print(
                "Range "
                + "{:02}".format(len(self.levels) - idx)
                + f" -> {level[0]}ft - {level[1]}ft"
            )

    def simulation_generate_trajs(
        self,
        duration: int = 24,
        interval: int = 60,
        start_time: str = "2000-01-01 00:00:00",
    ) -> pd.DataFrame:
        """
        Simulates a single run of the airspace simulation and returns a dataframe
        containing a set of randomly sampled trajectories and simulation timestamps
        according to the specified parameters.

        Parameters
        ----------
        duration : int, optional
            duration of the simulation in hours, by default 24
        interval : int, optional
            interval in seconds between flights entering the airspace, by default 60
        start_time : str, optional
            starting time of the simulation. Used to generate and assign simulation
            timestamps, by default "2000-01-01 00:00:00"

        Returns
        -------
        pd.DataFrame
            A dataframe containing a set of randomly sampled trajectories and simulation
            timestamps according to the specified parameters.
        """

        # Define home path
        home_path = util_general.get_project_root()

        # Load trajectory data and generate flight id list and dataframe
        trajs_low = Traffic.from_file(
            f"{home_path}/data/LSAGUAC/05_low_traffic/trajs_tma_low.parquet"
        )
        ids = trajs_low.flight_ids
        trajs_low_data = trajs_low.data
        grouped = trajs_low_data.groupby("flight_id")

        # Determine simulation duration in seconds and amount of sim-trajectories
        totalseconds = duration * 60 * 60
        amount_deploys = int(totalseconds / interval)

        # Generate list of injection timestamps
        timelist = []
        timer = 0
        for i in range(int(amount_deploys)):
            timelist.append(
                pd.Timestamp(start_time) + pd.Timedelta(seconds=timer)
            )
            timer = timer + interval
        times = np.array(timelist)

        # Generate list of random flight ids
        indices = np.random.default_rng().choice(
            len(ids), len(timelist), replace=True
        )
        ids = np.array(ids)[indices]

        # Generate list of simulated trajectories
        df_all = []
        for id, tm in zip(ids, times):
            temp = grouped.get_group(id)
            timedelta = temp["timestamp"] - temp["timestamp"].iloc[0]
            new_timestamp = tm + timedelta
            timestring = str(new_timestamp.iloc[0].time())
            flight_id = temp["flight_id"].iloc[0]
            temp = temp[["latitude", "longitude", "altitude", "icao24"]]
            temp.insert(0, "timestamp", new_timestamp)
            temp.insert(1, "flight_id", flight_id + "_" + timestring)
            df_all.append(temp)

        # Concatenate trajectories and sort by timestamp
        df_traf = (
            pd.concat(df_all, axis=0)
            .sort_values(by=["timestamp"])
            .reset_index()
        )

        # Crop trajectories longer than the simulation duration
        df_traf = df_traf[
            df_traf.timestamp
            <= (pd.Timestamp(start_time) + pd.Timedelta(hours=duration))
        ]

        # Return trajectories
        return df_traf

    def simulation_single_run(
        self,
        args: tuple,
    ) -> None:
        """_summary_

        Parameters
        ----------
        args : tuple
            Tuple containing the following simulation parameters as arguments:
            - num: int
                Identifier for the simulation run. Mainly used for the parallel
                execution of this function to save results in separate identifiable
                files.
            - duration: int
                Simulation duration in hours.
            - interval: int
                Injection interval (time interval between flights entering the airspace)
                in seconds.
        """
        # Unpack arguments
        (
            num,
            duration,
            interval,
        ) = args

        # Define home path
        home_path = util_general.get_project_root()

        # Generate set of simulation trajectories
        df_traf = self.simulation_generate_trajs(duration, interval)

        # If grid for airspace does not exist, generate it
        if not hasattr(self, "grid"):
            self.cells_generate(dim=5, alt_diff=1000)

        # Initialise results dictionary and counter
        results = {}
        total_count = 0

        # Iterate over cubes of airspace grid
        for cube in self.cubes:
            # Subset dataframe to only include flights within the cube
            subset = df_traf.loc[
                (df_traf["latitude"] >= cube.lat_min)
                & (df_traf["latitude"] <= cube.lat_max)
                & (df_traf["longitude"] >= cube.lon_min)
                & (df_traf["longitude"] <= cube.lon_max)
                & (df_traf["altitude"] >= cube.alt_low)
                & (df_traf["altitude"] <= cube.alt_high)
            ]

            # Group flights by flight ID and find entry and exit timestamps
            in_out = (
                subset.groupby("flight_id")["timestamp"]
                .agg(["min", "max"])
                .reset_index()
                .sort_values(by="min")
            )

            # Create a dictionary containing the exit timestamp as key and a list of
            # the corresponding flight IDs exiting at that timestamp as value
            flight_dict = {
                flight.max: [flight.flight_id]
                for flight in in_out.itertuples()
            }

            # Set counter for cube to zero
            count = 0
            # Iterate over flights and find overlaps of time spent in the cube
            for flight in in_out.itertuples():
                matches = []
                # Using the dict, find flights that that are within the cube at the same
                # time
                for other_flight in flight_dict.get(flight.min, []):
                    if (
                        flight.max
                        > in_out.loc[
                            in_out["flight_id"] == other_flight, "min"
                        ].values[0]
                    ):
                        # for each flight in the cube at the same time, append to list
                        # and increase counter
                        matches.append(other_flight)
                        count += 1

            # Update results (count for cube and total count)
            results[cube.id] = count
            total_count += count

        # Save results for each cube of the gird packed in a dictionary. First check if
        # directory exists, if not create it
        if not os.path.exists(
            f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
            f"/runs_cube_counts/"
        ):
            os.makedirs(
                f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
                f"/runs_cube_counts/"
            )
        with open(
            f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
            f"/runs_cube_counts/{num}_results.pkl",
            "wb",
        ) as fp:
            pickle.dump(results, fp)

        # Save the total count of occurences also as a pickle file. First check if
        # directory exists, if not create it
        if not os.path.exists(
            f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
            f"/runs_total_counts/"
        ):
            os.makedirs(
                f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
                f"/runs_total_counts/"
            )
        with open(
            f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
            f"/runs_total_counts/{num}_total_count.pkl",
            "wb",
        ) as fp:
            pickle.dump(total_count, fp)

    def simulation_monte_carlo_run(
        self,
        duration: int,
        interval: int,
        runs: int,
        max_process: int,
        start_num: int = 0,
    ):
        """
        Runs a defined amount of simulation runs (monte carlo style), using parallel
        processes. The simulation duration, injection interval and number of runs as
        well as the max amount of parallel processes can be defined as parameters.

        Parameters
        ----------
        duration : int
            Simulation duration in hours
        interval : int
            Injection interval in seconds. Time between two injections of aircraft into
            the simulation.
        runs : int
            Number of runs to be performed.
        max_process : int
            Maximum number of parallel processes to be used.
        start_num : int, optional
            Id number of the first simulation run to be performed. Only relevant for the
            folder name, by default 0
        """

        # Parallelisation of the simulation runs
        with mp.Pool(max_process) as pool:
            list(
                # tqdm to show progress
                tqdm(
                    pool.imap(
                        # run the simulation function in parallel processes accordint to
                        # set parameters
                        self.simulation_single_run,
                        [
                            (i, duration, interval)
                            for i in range(start_num, start_num + runs)
                        ],
                    ),
                    total=runs,
                )
            )

        # Generate aggregated results for all simulation runs

        # define home path
        home_path = util_general.get_project_root()

        # 1. List of all total counts
        # Get list of all total run counts
        folder_path = (
            f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
            f"/runs_total_counts"
        )
        file_list = os.listdir(folder_path)

        # Read values from all files and aggregate in list
        total_counts = []
        for file in file_list:
            with open(folder_path + "/" + file, "rb") as f:
                total_counts.append(pickle.load(f))

        # Save aggregated list as pickle file
        with open(
            f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
            f"/total_counts_aggregated.pkl",
            "wb",
        ) as fp:
            pickle.dump(total_counts, fp)

        # 2. Dictionary with list of counts for each cube
        # Get list of all dictionaries with counts for each cube for each run
        folder_path = (
            f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
            f"/runs_cube_counts"
        )
        file_list = os.listdir(folder_path)

        # Read dictionaries from all files and aggregate in list
        dictionaries = []
        for file in file_list:
            with open(folder_path + "/" + file, "rb") as f:
                dictionaries.append(pickle.load(f))

        # Aggregate dictionaries in one dictionary
        aggregated_dict = {}
        for dictionary in dictionaries:
            for key, value in dictionary.items():
                if key in aggregated_dict:
                    aggregated_dict[key].append(value)
                else:
                    aggregated_dict[key] = [value]

        # Save aggregated dictionary as pickle file
        with open(
            f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
            f"/cube_counts_aggregated.pkl",
            "wb",
        ) as fp:
            pickle.dump(aggregated_dict, fp)

    def simulation_plot_monte_carlo_histogram(
        self, duration: int, interval: int, ci: float = 0.9
    ) -> matplotlib.figure.Figure:
        """
        generates a histogram of the number of occurences for each run conducted as part
        of the Monte Carlo simulation with a line for the mean and 90% confidence
        interval of the mean. The simulation duration and injection interval define
        which monte carlo simulation is to be plotted. The confidence interval that is
        to be plotted can be defined as a parameter.

        Parameters
        ----------
        duration : int
            Simulation duration in hours, serves as identifier for the monte carlo
            results to be plotted.
        interval : int
            Injection interval in seconds, serves as identifier for the monte carlo
            results to be plotted.
        ci : float, optional
            Confidence interval of the mean to be plotted, by default 0.9

        Returns
        -------
        matplotlib.figure.Figure
            Histogram plot of the number of occurences for each run conducted as part of
            the Monte Carlo simulation with a line for the mean and 90% confidence
            interval of the mean.
        """

        # define home path
        home_path = util_general.get_project_root()

        # Read aggregated list of total counts from pickle file
        file_path = (
            f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}"
            f"/total_counts_aggregated.pkl"
        )
        with open(file_path, "rb") as f:
            total_occurences_list = pickle.load(f)

        # Return histogram plot
        return viz.plot_occurence_histogram(
            occ_list=total_occurences_list, ci=ci
        )

    def simulation_plot_monte_carlo_heatmap(
        self,
        duration: int,
        interval: int,
        alt_low: int,
    ) -> matplotlib.figure.Figure:
        """
        Generates a heatmap showing the number of occurences for each grid cell in the
        airspace for an altitude layer defined by the lower altitude of the layer.

        Parameters
        ----------
        duration : int
            Simulation duration in hours, serves as identifier for the monte carlo
            results to be plotted.
        interval : int
            Injection interval in seconds, serves as identifier for the monte carlo
            results to be plotted.
        alt_low : int
            lower bound of altitude layer to show in feet

        Returns
        -------
        matplotlib.figure.Figure
            Heatmap showing the number of occurences for each grid cell in the airspace
            for the given altitude layer.
        """

        # define home path
        home_path = util_general.get_project_root()

        # Read aggregated dictionary of cube counts from pickle file
        file_path = (
            f"{home_path}/data/{self.id}/08_monte_carlo/{duration}_{interval}/"
            f"cube_counts_aggregated.pkl"
        )
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # If grid for airspace does not exist, generate it
        if not hasattr(self, "grid"):
            self.cells_generate(dim=5, alt_diff=1000)

        # generate lists and put information about cube position and count in them
        lat_min = []
        lat_max = []
        lon_min = []
        lon_max = []
        alt_min = []
        alt_max = []
        count = []
        for cube in self.cubes:
            lat_max.append(cube.lat_max)
            lat_min.append(cube.lat_min)
            lon_max.append(cube.lon_max)
            lon_min.append(cube.lon_min)
            alt_max.append(cube.alt_high)
            alt_min.append(cube.alt_low)
            count.append(np.mean(data[cube.id]))

        # Combine lists into dataframe
        df = pd.DataFrame(
            list(
                zip(
                    lat_min, lat_max, lon_min, lon_max, alt_min, alt_max, count
                )
            ),
            columns=[
                "lat_min",
                "lat_max",
                "lon_min",
                "lon_max",
                "alt_min",
                "alt_max",
                "count",
            ],
        )

        # reduce dataframe to the altitude level to visualize
        df = df[df.alt_min == alt_low]

        # Return heatmap plot
        return viz.plot_occurence_heatmap(df, self.shape)


class cube:
    def __init__(
        self,
        id: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        alt_low: int,
        alt_high: int,
    ) -> None:
        """
        Class for a cube in the airspace grid. The cube is defined by its id, the
        minimum and maximum latitude, longitude and altitude.
        Parameters
        ----------
        id : str
            id of the cube in the format 'range_altitude_grid_increasing_number'
        lat_min : float
            minimum latitude of the cube
        lat_max : float
            maximum latitude of the cube
        lon_min : float
            minimum longitude of the cube
        lon_max : float
            maximum longitude of the cube
        alt_low : int
            lower altitude bound of the cube
        alt_high : int
            higher altitude bound of the cube
        """
        self.id = id
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.alt_low = alt_low
        self.alt_high = alt_high
        self.lat_cen = (self.lat_min + self.lat_max) / 2
        self.lon_cen = (self.lon_min + self.lon_max) / 2
        # self.alt_cen = (self.alt_low + self.alt_high) / 2

    def visualise(self) -> None:
        """
        Visualises the cube. The location of the cube is shown on a map and its upper
        and lower altitude bounds (vertical range) are printed out.
        """
        # Plot grid
        print("Location:")
        print("-----------------------------")
        fig = go.Figure(go.Scattermapbox())
        fig.update_layout(
            mapbox_style="carto-positron",
            showlegend=False,
            height=800,
            width=800,
            margin={"l": 0, "b": 0, "t": 0, "r": 0},
            mapbox_center_lat=self.lat_cen,
            mapbox_center_lon=self.lon_cen,
            mapbox_zoom=6,
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=[
                    self.lat_min,
                    self.lat_min,
                    self.lat_max,
                    self.lat_max,
                    self.lat_min,
                ],
                lon=[
                    self.lon_min,
                    self.lon_max,
                    self.lon_max,
                    self.lon_min,
                    self.lon_min,
                ],
                mode="lines",
                line=dict(width=2, color="red"),
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.3)",
                opacity=0.2,
                name="Rectangle",
            )
        )
        fig.show()

        # Print vertical ranges
        print("Vertical range:")
        print("-----------------------------")
        print(f"{self.alt_low}ft - {self.alt_high}ft")
        print("-----------------------------")
