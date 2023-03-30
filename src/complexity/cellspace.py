import os
import glob
from pathlib import Path
import math

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import plotly.graph_objects as go

from traffic.core import Traffic
import utils.general as util_general
import utils.adsb as util_adsb


class cellspace:
    def __init__(
        self,
        id: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        alt_min: int,
        alt_max: int,
    ) -> None:
        """
        Class for the cellspace object. The cellspace is defined by a bounding box of
        lat/lon/alt constraints. The cellspace is used to generate a grid of cells and
        assign the ADS-B data to the cells.

        Parameters
        ----------
        id : str
            name of the cellspace
        lat_min : float
            minimum latitude of the bounding box
        lat_max : float
            maximum latitude of the bounding box
        lon_min : float
            minimum longitude of the bounding box
        lon_max : float
            maximum longitude of the bounding box
        alt_min : int
            minimum altitude of the bounding box
        alt_max : int
            maximum altitude of the bounding box
        """

        # Initialize cellspace attributes
        self.id = id
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.alt_min = alt_min
        self.alt_max = alt_max
        self.lat_cen = (self.lat_min + self.lat_max) / 2
        self.lon_cen = (self.lon_min + self.lon_max) / 2
        self.alt_cen = (self.alt_min + self.alt_max) / 2

    def __repr__(self):
        self.overview()

    def overview(self) -> str:
        """
        Returns an overview of the state of different steps of the cellspace. The status
        is given as 'Yes' or 'No' for the folmining steps:
            - Downloaded -> ADS-B data downloaded
            - Combined -> ADS-B data combined to monthly files
            - Preprocessed -> ADS-B data preprocessed
            - Cells generated -> Cells generated
            - Data assigned -> ADS-B data assigned to cells and saved as individual
              files

        Returns
        -------
        str
            String containing the above information
        """

        # Define home path
        home_path = util_general.get_project_root()

        # Set default statuses
        downloaded = "No"
        combined = "No"
        preprocessed = "No"
        cells_generated = "No"
        data_assigned = "No"

        # if downloaded data exists, set status 'downloaded' to yes
        if os.path.isdir(f"{home_path}/data/{self.id}/01_raw") is True:
            downloaded = "Yes"
        # if combined data exists, set status 'combined' to yes
        if os.path.isdir(f"{home_path}/data/{self.id}/02_combined") is True:
            combined = "Yes"
        # if preprocessed data exists, set status 'preprocessed' to yes
        if (
            os.path.isdir(f"{home_path}/data/{self.id}/03_preprocessed")
            is True
        ):
            preprocessed = "Yes"
        # if cells exist, set status 'cells generated' to yes
        if hasattr(self, "cells"):
            cells_generated = "Yes"
        # if cell-specific data exists, set status 'preprocessed' to yes
        if os.path.isdir(f"{home_path}/data/{self.id}/04_cells") is True:
            data_assigned = "Yes"

        # Print overview
        print(
            f"Dowloaded: {downloaded} \nCombined: {combined} \
                \nPreprocessed: {preprocessed} \nCells generated: {cells_generated} \
                \nData assigned: {data_assigned}"
        )

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
            bounds=(self.lon_min, self.lat_min, self.lon_max, self.lat_max),
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
            path_save=f"{home_path}/data/{self.id}/03_preprocessed/monthly",
        )

        # Combine preprocessed data to one file
        print("Combining preprocessed data...")
        # Check if file already exists
        check_file = Path(
            f"{home_path}/data/{self.id}/03_preprocessed/preprocessed_all.parquet"
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
                f"{home_path}/data/{self.id}/03_preprocessed/preprocessed_all.parquet"
            )

    def generate_cells(self, dim: int = 20, alt_diff: int = 3000) -> None:
        """
        Generates a grid of cells for the cellspace. The cells are saved as a list of
        tuples (lat, lon, alt_min, alt_max) in the attribute 'grid'.

        Parameters
        ----------
        dim : int, optional
            horizontal cell size [dim x dim] in nautical miles, by default 20
        alt_diff : int, optional
            height of the cells in feet, by default 3000
        """

        # Generate horizontal grid
        lats = [self.lat_max]
        lons = [self.lon_min]
        self.grid = []
        lat = self.lat_max
        lon = self.lon_min
        # Generate latitude intervals to cover the entire cellspace
        while lat > self.lat_min:
            lat = util_general.new_pos_dist((lat, lon), dim, 180)[0]
            if lat > self.lat_min:
                lats.append(lat)
        # Generate longitude intervals to cover the entire cellspace
        while lon < self.lon_max:
            lon = util_general.new_pos_dist((lat, lon), dim, 90)[1]
            if lon < self.lon_max:
                lons.append(lon)
        # Generate grid of cells from the intervals and save them as a list of tuples
        # (lat_min, lat_max, lon_min, lon_max) in the attribute 'grid'
        for i in range(len(lats) - 1):
            for j in range(len(lons) - 1):
                self.grid.append((lats[i], lats[i + 1], lons[j], lons[j + 1]))

        # Generate altitude intervals and save them as a list of tuples (alt_min,
        # alt_max) in the attribute 'levels'
        count = np.array(
            [*range(math.ceil((self.alt_max - self.alt_min) / alt_diff))]
        )
        alts_min = self.alt_min + alt_diff * count
        alts_max = self.alt_min + alt_diff * (count + 1)
        alts = np.array(list(zip(alts_min, alts_max)))
        self.levels = alts

        # Generate cells and save them as a list of cell objects in the attribute
        # 'cells'
        self.cells = []
        for level in self.levels:
            for idx, grid in enumerate(self.grid):
                self.cells.append(
                    cell(
                        id=f"range_{level[0]}_grid_{idx}",
                        lat_min=min(grid[0], grid[1]),
                        lat_max=max(grid[0], grid[1]),
                        lon_min=min(grid[2], grid[3]),
                        lon_max=max(grid[2], grid[3]),
                        alt_min=min(level[0], level[1]),
                        alt_max=max(level[0], level[1]),
                    )
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

        # Plot horizontal grid as a scattermapbox
        print("Grid:")
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
            mapbox_zoom=4,
        )
        # for every cell in the grid, plot a rectangle
        for pos in self.grid:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[pos[0], pos[0], pos[1], pos[1], pos[0]],
                    lon=[pos[2], pos[3], pos[3], pos[2], pos[2]],
                    mode="lines",
                    line=dict(width=2, color="red"),
                    fill="toself",
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    opacity=0.2,
                    name="Rectangle",
                )
            )
        fig.show()

        # Print vertical ranges
        print("Vertical ranges:")
        print("-----------------------------")
        # for every level in the levels, print the range
        for idx, level in enumerate(self.levels[::-1]):
            print(
                "Range "
                + "{:02}".format(len(self.levels) - idx)
                + f" -> {level[0]}ft - {level[1]}ft"
            )

    def assign_cell_traffic(self) -> None:
        """
        Assigns the trajectory data to the cells and saves the data as a individual
        parquet files in the folder '04_cells'. The function requires the attribute
        'cells' to exist. If it does not exist, an error is raised.
        """

        # Define home path
        home_path = util_general.get_project_root()

        # Create directory to save cell data if it does not exist
        if os.path.isdir(f"{home_path}/data/{self.id}/04_cells") is False:
            os.mkdir(f"{home_path}/data/{self.id}/04_cells")

        # Check if all cells have been assigned traffic, if so, skip
        file_amount = len(
            glob.glob(
                f"{home_path}/data/{self.id}/04_cells/*.parquet",
                recursive=True,
            )
        )
        if file_amount < len(self.cells):
            # Load preprocessed data
            print("Loading preprocessed data...")
            alltrajs = Traffic.from_file(
                f"{home_path}/data/{self.id}/03_preprocessed/preprocessed_all.parquet"
            ).data
            # Iterate over all cells
            print("Assigning traffic to cells...")
            for cell in tqdm(self.cells):
                # Check if file for cell already exists, if not, extract the traffic
                # that is within the cell and save it as a parquet file
                check_file = Path(
                    f"{home_path}/data/{self.id}/04_cells/{cell.id}.parquet"
                )
                if check_file.is_file() is False:
                    celldata = alltrajs[
                        (alltrajs.latitude.between(cell.lat_min, cell.lat_max))
                        & (
                            alltrajs.longitude.between(
                                cell.lon_min, cell.lon_max
                            )
                        )
                        & (
                            alltrajs.altitude.between(
                                cell.alt_min, cell.alt_max
                            )
                        )
                    ]
                    Traffic(celldata).to_parquet(
                        f"{home_path}/data/{self.id}/04_cells/{cell.id}.parquet"
                    )


class cell:
    def __init__(
        self,
        id: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        alt_min: int,
        alt_max: int,
    ) -> None:
        """
        Class for the cell object which is a subcomponent of the cellspace. The cell is
        defined by a bounding box of lat/lon/alt constraints.

        Parameters
        ----------
        id : str
            id of the cell in the format 'range_altitude_grid_increasing_number'
        lat_min : float
            minimum latitude of the cell
        lat_max : float
            maximum latitude of the cell
        lon_min : float
            minimum longitude of the cell
        lon_max : float
            maximum longitude of the cell
        alt_min : int
            lower altitude bound of the cell
        alt_max : int
            higher altitude bound of the cell
        """

        # Uinitialise cell attributes
        self.id = id
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.alt_min = alt_min
        self.alt_max = alt_max
        self.lat_cen = (self.lat_min + self.lat_max) / 2
        self.lon_cen = (self.lon_min + self.lon_max) / 2
        self.alt_cen = (self.alt_min + self.alt_max) / 2

    def visualise(self) -> None:
        """
        Visualises the cell. The location of the cell is shown on a map and its upper
        and lower altitude bounds (vertical range) are printed out.
        """
        # Plot cell position on a scattermapbox
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
        print(f"{self.alt_min}ft - {self.alt_max}ft")
        print("-----------------------------")
