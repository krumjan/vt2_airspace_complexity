import glob
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point
from tqdm.auto import tqdm

import traffic
from traffic.core import Traffic

from utils import adsb as util_adsb
from utils import geo as util_geo
from utils import general as util_general


class airspace:
    def __init__(
        self, id: str, airspace: traffic.core.airspace.Airspace
    ) -> None:
        """
        Class for the definition of an airspace. The airspace is defined by a tuple of
        floats (lat_min, lat_max, lon_min, lon_max, alt_min, alt_max).

        Parameters
        ----------
        id : str
            id of the cellspace (e.g. 'rectangle_Switzerland')
        volume : tuple[float, float, float, float, int, int]
            Definition of the cellspace volume by a tuple of floats (lat_min, lat_max,
            lon_min, lon_max, alt_min, alt_max)
        """

        # Initialize class attributes
        self.id = id
        self.shape = airspace.shape
        self.lat_max = airspace.shape.bounds[3]
        self.lat_min = airspace.shape.bounds[1]
        self.lon_max = airspace.shape.bounds[2]
        self.lon_min = airspace.shape.bounds[0]
        self.lat_cen = airspace.shape.centroid.y
        self.lon_cen = airspace.shape.centroid.x
        self.alt_min = max(8500, airspace.elements[0].lower)
        self.alt_max = min(45000, airspace.elements[-1].upper)

    def get_data(
        self,
        start_date: str,
        end_date: str,
    ) -> None:
        """
        Fetches and combines ADS-B data for the entire cellspace for the provided time
        period. The data is downloaded in daily chunvks which are saved under
        'data/cellspace_id/01_raw'. The data is subsequently aggregated and saved as
        monthly Traffic objects under 'data/cellspace_id/02_combined'.

        Parameters
        ----------
        start_date : str
            start date for data collection in the format 'YYYY-MM-DD'
        end_date : str
            end date for data collection in the format 'YYYY-MM-DD'
        """

        # Define home path
        home_path = util_general.get_project_root()

        # Data fetching per day
        print("Fetching data...")
        util_adsb.download_adsb_para(
            start=start_date,
            stop=end_date,
            folder=f"{home_path}/data/{self.id}/01_raw",
            bounds=self.shape,
            lower=self.alt_min,
            upper=self.alt_max,
        )

        # Combining data to monthly files
        print("Combining data...")
        util_adsb.combine_adsb(
            path_raw=f"{home_path}/data/{self.id}/01_raw",
            path_combined=f"{home_path}/data/{self.id}/02_combined",
        )

    def preprocess_data(self) -> None:
        """
        Preprocesses the montly data packages and saves it again as monthly Traffic
        objects under 'data/cellspace_id/03_preprocessed/monthly' before combining them
        to one Traffic object under 'data/cellspace_id/03_preprocessed. Preprocessing
        includes the following steps:
            - assigning an id to each trajectory
            - removing invalid trajectories
            - applying filtering to the trajectories
            - resampling the trajectories to 5s intervals
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
        # Check if file already exists
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
            all_df = pd.DataFrame()
            for file in tqdm(files):
                all_df = pd.concat([all_df, Traffic.from_file(file).data])
            # Save combined dataframe as Traffic object
            Traffic(all_df).to_parquet(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_rec.parquet"
            )

        # Reduce to actual TMA extent
        print("Reducing to TMA extent...")
        check_file = Path(
            f"{home_path}/data/{self.id}/03_preprocessed/preprocessed_all_tma.parquet"
        )
        # If file does not exist, crop data
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
                "03_preprocessed/preprocessed_all_tma.parquet"
            )

    def plot(
        self, traj_sample: bool = False, traj_num: int = 200
    ) -> go.Figure:
        """
        Generates a plot of the airspace shape. If the parameter 'traj_sample' is set to
        True, a sample of the trajectories is added to the plot.

        Parameters
        ----------
        traj_sample : bool, optional
            If set to True, a random sample of trajectories will be added to the plot,
            by default False
        traj_num : int, optional
            Amount of trajectories which are plotted if traj_sample is set to True, by
            default 100

        Returns
        -------
        go.Figure
            Figure object containing the plot
        """

        # Create mapbox
        fig = go.Figure(go.Scattermapbox())
        fig.update_layout(
            mapbox_style="mapbox://styles/jakrum/clgqc6e8u00it01qzgtb4gg1z",
            mapbox_accesstoken="pk.eyJ1IjoiamFrcnVtIiwiYSI6ImNsZ3FjM3BiMzA3dzYzZHMzNHRkZnFtb3EifQ.ydDFlmylEcRCkRLWXqL1Cg",
            showlegend=False,
            height=800,
            width=800,
            margin={"l": 0, "b": 0, "t": 0, "r": 0},
            mapbox_center_lat=self.lat_cen,
            mapbox_center_lon=self.lon_cen,
            mapbox_zoom=4,
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

        # Depending on 'traj_sample', add a sample of trajectories to the plot
        if traj_sample:
            # Define home path and import data
            home_path = util_general.get_project_root()
            trajs = Traffic.from_file(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_tma.parquet"
            )
            # Add random sample of trajectories to the plot
            ids = random.sample(
                trajs.data["flight_id"].unique().tolist(), traj_num
            )
            for traj in trajs[ids]:
                fig.add_trace(
                    go.Scattermapbox(
                        mode="lines",
                        lat=traj.data["latitude"],
                        lon=traj.data["longitude"],
                        line=dict(width=2, color="blue"),
                    )
                )

        return fig

    def get_hourly_df(self, return_df: bool = False) -> pd.DataFrame:
        """
        Generates a dataframe containing hourly aggregated traffic information of the
        trajectories in the cell space. The dataframe is saved as a csv file under
        'data/cellspace_id/04_hourly' and can be returned as a pandas dataframe if
        'return_df' is set to True.

        Parameters
        ----------
        return_df : bool, optional
            If True, the dataframe will be returend by the function, by default False

        Returns
        -------
        pd.DataFrame
            Dataframe containing the hourly aggregated traffic information
        """

        # Define home path
        home_path = util_general.get_project_root()

        # Check if file already exists and if not, create it
        check_file = Path(
            f"{home_path}/data/{self.id}/04_hourly/hourly_df.parquet"
        )
        if check_file.is_file() is False:
            # Import data
            trajs = Traffic.from_file(
                f"{home_path}/data/{self.id}/"
                "03_preprocessed/preprocessed_all_tma.parquet"
            )
            # Aggregate data by flight_id, keeping the minimum and maximum timestamp
            df = trajs.data
            df = (
                df.groupby("flight_id")["timestamp"]
                .agg(["min", "max"])
                .reset_index()
            )
            df = df.rename({"min": "in", "max": "out"}, axis=1)
            # Compute stay time in hours
            df["stay_h"] = (df["out"] - df["in"]).dt.total_seconds() / 3600
            df["timestamp_entered_h"] = df["in"].dt.floor("h")
            df = df.drop(["in", "out"], axis=1)

            # Aggregate data by hour, keeping the number of flights and the total stay
            hourly_stay = df.groupby(["timestamp_entered_h"])["stay_h"].sum()
            hourly_users = df.groupby(["timestamp_entered_h"])[
                "flight_id"
            ].count()
            hourly_users.name = "ac_count"
            hourly_ids = df.groupby(["timestamp_entered_h"])[
                "flight_id"
            ].apply(list)
            hourly_ids.name = "flight_ids"
            hourly_df = pd.concat(
                [hourly_users, hourly_stay, hourly_ids], axis=1
            )
            hourly_df.reset_index(inplace=True)
            hourly_df = hourly_df.rename(
                {"timestamp_entered_h": "hour"}, axis=1
            )

            # Fill missing hours with 0
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
                    "stay_h",
                    "flight_ids",
                ]
            ]
            # Save dataframe as parquet file
            if not os.path.exists(f"{home_path}/data/{self.id}/04_hourly/"):
                os.makedirs(f"{home_path}/data/{self.id}/04_hourly/")
            hourly_df.to_parquet(check_file)
            # Return dataframe if 'return_df' is set to True
            if return_df:
                hourly_df = pd.read_parquet(check_file)
                return hourly_df

        # Return dataframe if 'return_df' is set to True
        else:
            if return_df:
                hourly_df = pd.read_parquet(check_file)
                return hourly_df

    def reduce_low_traffic(self, reference_type: str, reference_value: float):
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

    def generate_cells(self, dim: int = 20, alt_diff: int = 3000) -> None:
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
        count = np.array(
            [*range(math.ceil((self.alt_max - self.alt_min) / alt_diff))]
        )
        alts_low = 18000 + alt_diff * count
        alts_high = 18000 + alt_diff * (count + 1)
        alts = np.array(list(zip(alts_low, alts_high)))
        self.levels = alts

        # Generate cubes
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

    def create_training_data(self):
        # Define home path
        home_path = util_general.get_project_root()

        # Only do the steps if the required file does not exist yet
        if not os.path.exists(f"{home_path}/data/{self.id}/06_training/X.npy"):
            # If directory does not exist, create it
            if not os.path.exists(f"{home_path}/data/{self.id}/06_training/"):
                os.makedirs(f"{home_path}/data/{self.id}/06_training/")

            # If resampled trajectory data does not exist, create it
            if not os.path.exists(
                f"{home_path}/data/{self.id}/06_training/resampled_100.parquet"
            ):
                # Load reduced trajectory data
                trajs = Traffic.from_file(
                    f"{home_path}/data/{self.id}/05_low_traffic/trajs_rec_low.parquet"
                )
                # Resample
                trajs = (
                    trajs.resample("1s")
                    .resample(100)
                    .eval(desc="resampling", max_workers=30)
                )
                # Save resampled trajectory data
                trajs.to_parquet(
                    f"{home_path}/data/{self.id}/06_training/resampled_100.parquet"
                )

            # load resampled trajectory data
            trajs = Traffic.from_file(
                f"{home_path}/data/{self.id}/06_training/resampled_100.parquet"
            )

            # Create training data array
            X = []
            for flight in tqdm(trajs):
                df = flight.data
                start_time = df.timestamp.iloc[0]
                df["timedelta"] = (
                    df["timestamp"] - start_time
                ).dt.total_seconds()
                if len(df) == 100:
                    df = df[
                        [
                            "latitude",
                            "longitude",
                            "altitude",
                            "groundspeed",
                            "timedelta",
                        ]
                    ]
                    df = (
                        df.interpolate(method="linear", limit_direction="both")
                        .ffill()
                        .bfill()
                    )
                    df_as_np = df.to_numpy()
                    X.append(df_as_np)

            X = np.array(X)

            # Remove trajs with missing data
            indexList_X_nan = [np.any(i) for i in np.isnan(X)]
            X = np.delete(X, indexList_X_nan, axis=0)

            # Min Max scaling
            lat_max = np.max(X[:, :, 0])
            lat_min = np.min(X[:, :, 0])
            lon_max = np.max(X[:, :, 1])
            lon_min = np.min(X[:, :, 1])
            alt_max = np.max(X[:, :, 2])
            alt_min = np.min(X[:, :, 2])
            gs_max = np.max(X[:, :, 3])
            gs_min = np.min(X[:, :, 3])
            tm_max = np.max(X[:, :, 4])
            tm_min = np.min(X[:, :, 4])

            X_norm = X.copy()
            X_norm[:, :, 0] = (
                2 * (X_norm[:, :, 0] - lat_min) / (lat_max - lat_min) - 1
            )
            X_norm[:, :, 1] = (
                2 * (X_norm[:, :, 1] - lon_min) / (lon_max - lon_min) - 1
            )
            X_norm[:, :, 2] = (
                2 * (X_norm[:, :, 2] - alt_min) / (alt_max - alt_min) - 1
            )
            X_norm[:, :, 3] = (
                2 * (X_norm[:, :, 3] - gs_min) / (gs_max - gs_min) - 1
            )
            X_norm[:, :, 4] = (
                2 * (X_norm[:, :, 4] - tm_min) / (tm_max - tm_min) - 1
            )

            np.save(f"{home_path}/data/{self.id}/06_training/X", X)
            np.save(f"{home_path}/data/{self.id}/06_training/X_norm", X_norm)
            with open(
                f"{home_path}/data/{self.id}/06_training/normalisation.txt",
                "w",
            ) as f:
                f.write(
                    str(lat_max)
                    + " "
                    + str(lat_min)
                    + " "
                    + str(lon_max)
                    + " "
                    + str(lon_min)
                    + " "
                    + str(alt_max)
                    + " "
                    + str(alt_min)
                    + " "
                    + str(gs_max)
                    + " "
                    + str(gs_min)
                    + " "
                    + str(tm_max)
                    + " "
                    + str(tm_min)
                )

    def visualise_cells(self) -> None:
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
            mapbox_accesstoken='pk.eyJ1IjoiamFrcnVtIiwiYSI6ImNsZ3FjM3BiMzA3dzYzZHMzNHRkZnFtb3EifQ.ydDFlmylEcRCkRLWXqL1Cg',
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

    def run_simulation(self, duration: int, interval: int) -> None:
        # Define home path
        home_path = util_general.get_project_root()

        # Load data and get ids and dataframe
        print("Loading data...")
        trajs_low = Traffic.from_file(
            f"{home_path}/data/{self.id}/05_low_traffic/trajs_tma_low.parquet"
        )
        ids = trajs_low.flight_ids
        trajs_low_data = trajs_low.data

        totalseconds = duration * 24 * 60 * 60
        amount_deploys = int(totalseconds / interval)

        timelist = []

        timer = 0
        for i in range(int(amount_deploys)):
            timelist.append(timer)
            timer = timer + interval

        indices = np.random.default_rng().choice(
            len(ids), len(timelist), replace=True
        )

        random_ids = np.array(ids)[indices]
        random_times = np.array(timelist)

        grouped = trajs_low_data.groupby("flight_id")

        # generate simulated trajectories
        print("Generating simulated trajectories...")
        df_all = []
        start_time = pd.Timestamp("2000-01-01 00:00:00")

        for id, tm in tqdm(
            zip(random_ids, random_times), total=len(random_ids)
        ):
            traj_time = start_time + pd.Timedelta(seconds=tm)
            temp = grouped.get_group(id)
            timedelta = temp["timestamp"] - temp["timestamp"].iloc[0]
            new_timestamp = traj_time + timedelta
            timestring = str(new_timestamp.iloc[0].time())
            flight_id = temp["flight_id"].iloc[0]
            temp = temp[["latitude", "longitude", "altitude", "icao24"]]
            temp.insert(0, "timestamp", new_timestamp)
            temp.insert(1, "flight_id", flight_id + "_" + timestring)
            df_all.append(temp)

        df_traf = (
            pd.concat(df_all, axis=0)
            .sort_values(by=["timestamp"])
            .reset_index()
        )
        if not os.path.exists(f"{home_path}/data/{self.id}/06_simulation/"):
            os.makedirs(f"{home_path}/data/{self.id}/06_simulation/")
        df_traf.to_parquet(
            f"{home_path}/data/{self.id}/06_simulation/trajs_simulation.parquet",
            index=False,
        )

        print("Checking simultaneous trajectories...")
        if not os.path.exists(f"{home_path}/data/{self.id}/07_cube_data/"):
            os.makedirs(f"{home_path}/data/{self.id}/07_cube_data/")
        for cube in tqdm(self.cubes):
            subset = df_traf.loc[
                (df_traf["latitude"] >= cube.lat_min)
                & (df_traf["latitude"] <= cube.lat_max)
                & (df_traf["longitude"] >= cube.lon_min)
                & (df_traf["longitude"] <= cube.lon_max)
                & (df_traf["altitude"] >= cube.alt_low)
                & (df_traf["altitude"] <= cube.alt_high)
            ]
            subset.to_parquet(
                f"{home_path}/data/{self.id}/07_cube_data/{cube.id}_trajs.parquet",
                index=False,
            )

            in_out = (
                subset.groupby("flight_id")["timestamp"]
                .agg(["min", "max"])
                .reset_index()
                .sort_values(by="min")
            )
            in_out.to_parquet(
                f"{home_path}/data/{self.id}/07_cube_data/{cube.id}_inout.parquet",
                index=False,
            )

            df = in_out

            count = 0
            pairs = []

            # Create a dictionary of flights indexed by their max value
            flight_dict = {
                flight.max: [flight.flight_id] for flight in df.itertuples()
            }

            # Find overlapping flights
            pairs = []
            count = 0
            for flight in df.itertuples():
                matches = []
                for other_flight in flight_dict.get(flight.min, []):
                    if (
                        flight.max
                        > df.loc[
                            df["flight_id"] == other_flight, "min"
                        ].values[0]
                    ):
                        matches.append(other_flight)
                        count += 1
                if matches:
                    matches.append(flight.flight_id)
                    pairs.append(tuple(matches))

            with open(
                f"{home_path}/data/{self.id}/07_cube_data/{cube.id}_pairs.pkl",
                "wb",
            ) as fp:
                pickle.dump(pairs, fp)

            # print(
            #     f"{cube.id} with {len(df)} flights with {len(pairs)} overlapping intervals."
            # )


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
