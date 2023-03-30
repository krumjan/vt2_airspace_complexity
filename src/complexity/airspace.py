import os

import pandas as pd
from tqdm.auto import tqdm
from shapely.geometry import Point
import plotly.graph_objects as go

from traffic.core import Traffic
import traffic
import utils.general as util_general


class airspace:
    def __init__(
        self, id: str, airspace: traffic.core.airspace.Airspace
    ) -> None:
        """
        Class for the definition of an cellspace. The cellspace is defined by a tuple of
        floats (lat_min, lat_max, lon_min, lon_max, alt_min, alt_max).

        Parameters
        ----------
        id : str
            id of the cellspace (e.g. 'rectangle_Switzerland')
        volume : tuple[float, float, float, float, int, int]
            Definition of the cellspace volume by a tuple of floats (lat_min, lat_max,
            lon_min, lon_max, alt_min, alt_max)
        """

        self.id = id
        self.shape = airspace.shape
        self.lat_cen = airspace.shape.centroid.y
        self.lon_cen = airspace.shape.centroid.x
        self.alt_min = airspace.elements[0].lower
        self.alt_max = airspace.elements[0].upper

    def assign_cells(self, cellspace):
        self.cellspace_id = cellspace.id
        self.airspace_cells = []
        for cell in cellspace.cells:
            cell_pos = Point(cell.lon_cen, cell.lat_cen)
            cell_alt = cell.alt_cen

            if self.shape.contains(cell_pos):
                if cell_alt >= self.alt_min and cell_alt <= self.alt_max:
                    self.airspace_cells.append(cell)

    def visualise_cells(self):
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
        lons, lats = self.shape.exterior.xy
        trace = go.Scattermapbox(
            mode="lines",
            lat=list(lats),
            lon=list(lons),
            line=dict(width=2, color="blue"),
        )
        fig.add_trace(trace)

        cell_list = []
        for cell in self.airspace_cells:
            temp = (
                cell.lat_min,
                cell.lat_max,
                cell.lon_min,
                cell.lon_max,
                cell.lon_cen,
                cell.lat_cen,
            )
            if temp not in cell_list:
                cell_list.append(temp)

        for cell in cell_list:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[cell[0], cell[0], cell[1], cell[1], cell[0]],
                    lon=[cell[2], cell[3], cell[3], cell[2], cell[2]],
                    mode="lines",
                    line=dict(width=2, color="red"),
                    fill="toself",
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    opacity=0.2,
                    name="Rectangle",
                )
            )
            fig.add_trace(
                go.Scattermapbox(
                    lat=[cell[5]],
                    lon=[cell[4]],
                    mode="markers",
                    marker=dict(size=3, color="red"),
                    name="Cell Center",
                )
            )

        fig.show()

    def get_combined_traffic(self):
        home_path = util_general.get_project_root()
        all_traffic = pd.concat(
            [
                Traffic.from_file(
                    f"{home_path}/data/{self.cellspace_id}/04_cells/{cell.id}.parquet"
                ).data
                for cell in tqdm(self.airspace_cells)
                if os.path.isfile(
                    f"{home_path}/data/{self.cellspace_id}/04_cells/{cell.id}.parquet"
                )
            ]
        )

        return Traffic(all_traffic)
