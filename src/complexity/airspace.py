import os
import random
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tqdm.auto import tqdm

import traffic
from traffic.core import Traffic

from utils import adsb as util_adsb
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
        self, traj_sample: bool = False, traj_num: int = 100
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
            mapbox_style="carto-positron",
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
            line=dict(width=2, color="blue"),
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
                        line=dict(width=2, color="red"),
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

    def create_training_data(self):
        # Define home path
        home_path = util_general.get_project_root()

        # Load reduced trajectory data
        trajs = Traffic.from_file(
            f"{home_path}/data/{self.id}/05_low_traffic/trajs_rec_low.parquet"
        )

        trajs = trajs.resample(200).eval(desc="resampling", max_workers=20)

        X = []
        for flight in tqdm(trajs):
            df = flight.data
            t0 = df["timestamp"].iloc[0]
            df["timedelta"] = df["timestamp"] - t0
            df = flight.data[
                ["timedelta", "latitude", "longitude", "altitude"]
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
        X_norm = X.copy()
        X_norm[:, :, 0] = (X_norm[:, :, 0] - lat_min) / (lat_max - lat_min)
        X_norm[:, :, 1] = (X_norm[:, :, 1] - lon_min) / (lon_max - lon_min)
        X_norm[:, :, 2] = (X_norm[:, :, 2] - alt_min) / (alt_max - alt_min)

        np.save(f"{home_path}/data/{self.id}/06_training/X", X)
        np.save(f"{home_path}/data/{self.id}/06_training/X_norm", X_norm)
        with open(
            f"{home_path}/data/{self.id}/06_training/normalisation.txt", "w"
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
            )

        return X
