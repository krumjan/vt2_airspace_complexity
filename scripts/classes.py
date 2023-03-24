import math
import os
import traffic

import adsb_functions as fn
import numpy as np
import plotly.graph_objects as go

class airspace:
    def __init__(
        self,
        id: str,
        volume: traffic.core.airspace.Airspace
                | tuple[float, float, float, float, int, int],
    ) -> None:
        """
        Class for the definition of an airspace. The airspace can be defined either by
        a traffic.core.airspace.Airspace object or by a tuple of floats (lat_min,
        lat_max, lon_min, lon_max, alt_low, alt_high).

        Parameters
        ----------
        id : str
            id of the airspace (e.g. 'MUAC')
        volume : traffic.core.airspace.Airspace | 
                 tuple[float, float, float, float, int, int]
            Definition of the airspace volume. Can be a traffic.core.airspace.Airspace
            object or a tuple of floats (lat_min, lat_max, lon_min, lon_max, alt_low,
            alt_high)
        """
        # initialisation traffic airspace specific
        if type(volume) == traffic.core.airspace.Airspace:
            pass

        # initialisation tuple specific
        else:
            self.lat_min = volume[0]
            self.lat_max = volume[1]
            self.lon_min = volume[2]
            self.lon_max = volume[3]
            self.alt_low = volume[4]
            self.alt_high = volume[5]

        # general initialisation
        self.id = id
        self.volume = volume
        self.lat_cen = (self.lat_min + self.lat_max) / 2
        self.lon_cen = (self.lon_min + self.lon_max) / 2
        self.alt_cen = (self.alt_low + self.alt_high) / 2

    def __repr__(self):
        return self.overview()
    
    def overview(
        self
    ) -> str:
        """
        Returns an overview of the data status for the airspace. The status is given as
        'Yes' or 'No' for the following steps:
        - Downloaded
        - Combined
        - Preprocessed
        - Cells generated
        - Data assigned

        Returns
        -------
        str
            String containing the above information
        """
        downloaded = 'No'
        combined = 'No'
        preprocessed = 'No'
        cells_generated = 'No'
        data_assigned = 'No'

        if os.path.isdir(f'data/{self.id}/01_raw') is True:
            downloaded = 'Yes'
        if os.path.isdir(f'data/{self.id}/02_combined') is True:
            combined = 'Yes'
        if os.path.isdir(f'data/{self.id}/03_preprocessed') is True:
            preprocessed = 'Yes'
        if hasattr(self, 'cubes'):
            cells_generated = 'Yes'

        return f"Dowloaded: {downloaded} \nCombined: {combined} \
                \nPreprocessed: {preprocessed} \nCells generated: {cells_generated} \
                \nData assigned: {data_assigned}"

    def get_data(
        self,
        start_date: str,
        end_date: str,
    ) -> None:
        """
        Fetches and combines ADS-B data for the airspace for the given time period. The
        data is downloaded in daily packages which are saved under 
        'data/airspace_id/01_raw'. Afterwards the data is aggregated and saved as 
        monthly Traffic objects under 'data/airspace_id/02_combined'.

        Parameters
        ----------
        start_date : str
            start date for data collection
        end_date : str
            end date for data collection
        """
        # Data fetching per day
        print("Fetching data...")
        fn.download_adsb_para(
            start=start_date,
            stop=end_date,
            folder=f"data/{self.id}/01_raw",
            bounds=(self.lon_min, self.lat_min, self.lon_max, self.lat_max),
            lower=self.alt_low,
            upper=self.alt_high,
        )

        # Data combining to monthly files
        print("Combining data...")
        fn.combine_adsb(
            path_raw=f"data/{self.id}/01_raw",
            path_combined=f"data/{self.id}/02_combined",
        )

    def preprocess_data(
        self
    ) -> None:
        """
        Preprocesses the data and saves it as Traffic object in the data folder. The
        preprocessing includes:
        - assigning an id to each trajectory
        - removing invalid trajectories
        - applying filtering to the trajectories
        - resampling the trajectories to 5s intervals
        """

        # Preprocess data
        fn.preprocess_adsb(
            path_get = f"data/{self.id}/02_combined",
            path_save = f"data/{self.id}/03_preprocessed",)

    def generate_cells(
        self,
        dim: int = 20,
        alt_diff: int = 3000
    ) -> None:
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
            lat = fn.new_pos_dist((lat, lon), dim, 180)[0]
            if lat > self.lat_min:
                lats.append(lat)
        while lon < self.lon_max:
            lon = fn.new_pos_dist((lat, lon), dim, 90)[1]
            if lon < self.lon_max:
                lons.append(lon)
        for i in range(len(lats) - 1):
            for j in range(len(lons) - 1):
                self.grid.append((lats[i], lats[i + 1], lons[j], lons[j + 1]))

        # Generate altitude intervals
        count = np.array([*range(math.ceil((self.alt_high - self.alt_low) / alt_diff))])
        alts_low = 18000 + 3000 * count
        alts_high = 18000 + 3000 * (count + 1)
        alts = np.array(list(zip(alts_low, alts_high)))
        self.levels = alts

        # Generate cubes
        self.cubes = []
        for level in self.levels:
            for idx, grid in enumerate(self.grid):
                self.cubes.append(
                    cube(
                        id=f"range_{level[0]}_grid_{idx}",
                        lat_min=grid[0],
                        lat_max=grid[1],
                        lon_min=grid[3],
                        lon_max=grid[2],
                        alt_low=level[0],
                        alt_high=level[1],
                    )
                )

    def visualise_cells(
        self
    ) -> None:
        """
        Visualises the airspace grid and altitude levels. The function requires the
        attribute 'grid' to exist. If it does not exist, an error is raised.

        Raises
        ------
        ValueError
            Error is raised if the attribute 'grid' does not exist.
        """
        # Raise error if grid does not exist
        if not hasattr(self, 'grid'):
            raise ValueError("Grid does not exist. Please execute function "
                             "generate_cells() first.")
        
        # Plot grid
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
            mapbox_zoom=6,
        )
        for pos in self.grid:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[pos[0], pos[0], pos[1], pos[1], pos[0]],
                    lon=[pos[2], pos[3], pos[3], pos[2], pos[2]],
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
        print("Vertical ranges:")
        print("-----------------------------")
        for idx, level in enumerate(self.levels[::-1]):
            print(
                "Range "
                + "{:02}".format(len(self.levels) - idx)
                + f" -> {level[0]}ft - {level[1]}ft"
            )

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
        self.alt_cen = (self.alt_low + self.alt_high) / 2

    def visualise(
        self
    ) -> None:
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
                lat=[self.lat_min, self.lat_min, self.lat_max,
                     self.lat_max, self.lat_min],
                lon=[self.lon_min, self.lon_max, self.lon_max,
                     self.lon_min, self.lon_min],
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
