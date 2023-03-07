import time

import multiprocessing as mp

from pathlib import Path
import os
from tqdm.auto import tqdm
from traffic.data import opensky
from traffic.data import nm_airspaces
import numpy as np
import random
import colorsys
from typing import Union, Tuple
from shapely.geometry.base import BaseGeometry

# Download function---------------------------------------------------------------------------------
def download_adsb(
    t0: str,
    tf: str,
    folder: str,
    area: Union[None, str, BaseGeometry, Tuple[float, float, float, float]],
):
    check_file = Path(f"{folder}/{t0.date()}_{tf.date()}.parquet",
                      use_deprecated_int96_timestamps=True)
    if check_file.is_file() == False:
        print(f"Downloading {t0.date()}...")
        traffic_data = opensky.history(
            t0,
            tf,
            bounds=area,
            progressbar=tqdm,
            cached=True,
        )
        if traffic_data is None:
            print("empty day")
        else:
            traffic_data.to_parquet(f"{folder}/{t0.date()}_{tf.date()}.parquet",
                                    use_deprecated_int96_timestamps=True)


# Parralelisation-----------------------------------------------------------------------------------
def download_adsb_para(dates, folder, airspace):
    if not os.path.exists('data/rectangle_1/2022/raw'):
        os.makedirs('data/rectangle_1/2022/raw')
    t0 = dates [:-1]
    tf = dates [1:]
    fol = [folder for i in range(len(t0))]
    airs = [airspace for i in range(len(t0))]
    t = [(t0, t1, fol, airs) for t0, t1, fol, airs in zip(t0, tf, fol, airs)]
    max_process = 8
    with mp.Pool(max_process) as pool:
        pool.starmap(download_adsb, t)

def generate_color_list(num_colors):
    # Generate a list of evenly spaced hues
    hues = [i / float(num_colors) for i in range(num_colors)]

    # Shuffle the list of hues
    random.shuffle(hues)

    # Convert the hues to RGB colors
    colors = [tuple(int(i * 255) for i in colorsys.hsv_to_rgb(hue, 0.8, 0.8)) for hue in hues]

    # Convert the RGB colors to hex strings
    hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]

    return hex_colors

def new_pos_dist(pos: tuple, dist: float, bear: float) -> tuple:
    """
    Computes the new position (latitude, longitude), given an origin (latitude, longitude),
    a distance (nautical miles) and a bearing (° -> direction of displacement)
    Parameters
    ----------
    pos : tuple
        Current position in the format (latitude, longitude)
    dist : float
        Distance of displacement in nautical miles
    bear : float
        Bearing of displacement in °
    Returns
    -------
    tuple
        New position computed from initial position, distance and bearing (latitude, longitude)
    """

    # conversion to radians
    lat = np.radians(pos[0])
    lon = np.radians(pos[1])
    # earth circumference
    R = 6371e3
    # dist nm -> m
    dist_m = dist * 1852
    # angular distance
    ang_d = dist_m / R
    # conversion to radians
    brng = np.radians(bear)
    # haversine formula
    lat2 = np.arcsin(
        np.sin(lat) * np.cos(ang_d) + np.cos(lat) * np.sin(ang_d) * np.cos(brng)
    )
    lon2 = lon + np.arctan2(
        np.sin(brng) * np.sin(ang_d) * np.cos(lat),
        np.cos(ang_d) - np.sin(lat) * np.sin(lat2),
    )

    return (np.degrees(lat2), np.degrees(lon2))

def rect_boundary(corner: tuple, width: float, height: float) -> list:
    """
    Computes the coordinates of the vertices of a rectangle given the coordinates of the upper left
    corner, the width and the height of the rectangle.
    Parameters
    ----------
    corner : tuple
        Coordinates of the upper left corner of the rectangle (latitude, longitude)
    width : float
        Width of the rectangled [nm]
    height : float
        Height of the rectangle [nm]
    Returns
    -------
    list
        Coordinates of the vertices of the rectangle
    """
    # ul = upper left, ur = upper right, ll = lower left, lr = lower right
    lat_ul, lon_ul = corner[0], corner[1]
    lat_ur, lon_ur = new_pos_dist((lat_ul, lon_ul), width, 90)
    lat_ll, lon_ll = new_pos_dist((lat_ul, lon_ul), height, 180)
    lat_lr, lon_lr = new_pos_dist((lat_ll, lon_ll), width, 90)

    return [(lat_ul, lon_ul), (lat_ur, lon_ur), (lat_ll, lon_ll), (lat_lr, lon_lr)]

def query_bound(corner: tuple, width: float, height: float) -> list:
    """
    Computes the coordinates of the vertices of a rectangle given the coordinates of the upper left
    corner, the width and the height of the rectangle.
    Parameters
    ----------
    corner : tuple
        Coordinates of the upper left corner of the rectangle (latitude, longitude)
    width : float
        Width of the rectangled [nm]
    height : float
        Height of the rectangle [nm]
    Returns
    -------
    list
        Coordinates of the vertices of the rectangle
    """

    left = corner[1]
    right = new_pos_dist((corner[0], corner[1]), width, 90)[1]
    top = corner[0]
    bottom = new_pos_dist((corner[0], corner[1]), height, 180)[0]
    return [left, bottom, right, top]
