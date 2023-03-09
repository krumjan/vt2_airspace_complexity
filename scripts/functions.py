import time

import multiprocessing as mp
import glob
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
from traffic.core import Traffic
import pandas as pd

# Data fetching ------------------------------------------------------------------------------------
def combine_adsb(path_raw: str, path_combined: str):
    """
    Combines all parquet files in the path provided in "path_raw" into one parquet file and saves it
    in the "path_combined" folder.

    Parameters
    ----------
    path_raw : str
        Folder path where the daily parquet files are stored
    path_combined : str
        Folder path where the combined parquet file will be stored
    """
    # list all parquet files in the raw folder
    files = glob.glob(f"{path_raw}/*.parquet")
    # concatenate all files into one Traffic object
    alldata = Traffic(
        pd.concat([Traffic.from_file(file).data for file in files], ignore_index=True)
    )
    # create "combined" folder if it does not exist
    if os.path.isdir(path_combined) == False:
        os.mkdir(path_combined)
    # save the Traffic object as a parquet file
    alldata.to_parquet(
        f"{path_combined}/combined.parquet", use_deprecated_int96_timestamps=True
    )
    
def download_adsb(
    t0: str,
    tf: str,
    folder: str,
    area: Union[None, str, BaseGeometry, Tuple[float, float, float, float]],
    lower: float,
    upper: float,
):
    """
    Queries ADS-B data from Opensky Network for the given time interval, geographical footprint,
    altitude constraints and saves the data as Traffic in a parquet file format.

    Parameters
    ----------
    t0 : str
        Start datetime for the query in the format "YYYY-MM-DD hh:mm:ss"
    tf : str
        Stop datetime for the query in the format "YYYY-MM-DD hh:mm:ss"
    folder : str
        Path to the folder where the data will be saved
    area : Union[None, str, BaseGeometry, Tuple[float, float, float, float]]
        Geographical footprint. Can be a string (e.g. "LFBB"), a shapely geometry or a tuple of
        floats (lon_west, lat_south, lon_east, lat_north)
    lower : float
        Lower altitude constraint in feet
    upper : float
        Upper altitude constraint in feet
    """
    # check whether file already exists
    check_file = Path(f"{folder}/{t0.date()}_{tf.date()}.parquet",
                      use_deprecated_int96_timestamps=True)
    # if not, print which day is being downloaded and download
    if check_file.is_file() == False:
        print(f"Downloading {t0.date()}...")
        traffic_data = opensky.history(
            t0,
            tf,
            bounds=area,
            progressbar=tqdm,
            cached=True,
            other_params = f' and baroaltitude >= {lower} and baroaltitude <= {upper}\
                  and onground = false '
        )
        # if not empty, save. Otherwise, print empty day
        if traffic_data is None:
            print("empty day")
        else:
            traffic_data.to_parquet(f"{folder}/{t0.date()}_{tf.date()}.parquet",
                                    use_deprecated_int96_timestamps=True)
                
def download_adsb_para(
    start: str,
    stop: str,
    folder: str,
    bounds: Union[None, str, BaseGeometry, Tuple[float, float, float, float]],
    lower: float,
    upper: float,
    max_process: int = 8,
):
    """
    Parllelisation of the download_adsb function. Queries ADS-B data from Opensky Network for the
    given time interval, geographical footprint, altitude constraints and saves the data as Traffic
    in a parquet file format one day at a time.

    Parameters
    ----------
    start : str
        Start datetime for the query in the format "YYYY-MM-DD"
    stop : str
        Stop datetime for the query in the format "YYYY-MM-DD"
    folder : str
        Path to the folder where the data will be saved as one parquet file per day
    bounds : Union[None, str, BaseGeometry, Tuple[float, float, float, float]]
        Geographical footprint. Can be a string (e.g. "LFBB"), a shapely geometry or a tuple of
        floats (lon_west, lat_south, lon_east, lat_north)
    lower : float
        Lower altitude constraint in feet
    upper : float
        Lower altitude constraint in feet
    max_process : int, optional
        Number of processes to use for parallelization, by default 8
    """
    # convert bounds to meters
    lower_m = lower * 0.3048
    upper_m = upper * 0.3048
    # create list of dates between start and stop
    dates = pd.date_range(start, stop, freq="D", tz="UTC")
    # create folder if it does not exist
    if not os.path.exists("data/rectangle_1/2022/raw"):
        os.makedirs("data/rectangle_1/2022/raw")
    # create list of arguments for the download_adsb function
    t0 = dates[:-1]
    tf = dates[1:]
    fol = [folder for i in range(len(t0))]
    airs = [bounds for i in range(len(t0))]
    lowers_m = [lower_m for i in range(len(t0))]
    uppers_m = [upper_m for i in range(len(t0))]
    t = [
        (t0, t1, fol, airs, lowers_m, uppers_m)
        for t0, t1, fol, airs, lowers_m, uppers_m in zip(
            t0, tf, fol, airs, lowers_m, uppers_m
        )
    ]
    # run the download_adsb function in parallel
    with mp.Pool(max_process) as pool:
        pool.starmap(download_adsb, t)

def new_pos_dist(pos: tuple, dist: float, bear: float) -> tuple:
    """
    Computes the new position (latitude, longitude), given an origin (latitude, longitude),
    a distance (nautical miles) and a bearing (° -> direction of displacement) using the
    haversine formula.
    Parameters
    ----------
    pos : tuple
        Current position in decimal degree format (latitude, longitude)
    dist : float
        Distance of displacement in nautical miles
    bear : float
        Bearing of displacement in °
    Returns
    -------
    tuple
        New position in decimal degrees computed from initial position, distance and bearing
        (latitude, longitude)
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
    # combutation of new position with haversine formula
    lat2 = np.arcsin(
        np.sin(lat) * np.cos(ang_d) + np.cos(lat) * np.sin(ang_d) * np.cos(brng)
    )
    lon2 = lon + np.arctan2(
        np.sin(brng) * np.sin(ang_d) * np.cos(lat),
        np.cos(ang_d) - np.sin(lat) * np.sin(lat2),
    )
    # return new position in degrees
    return (np.degrees(lat2), np.degrees(lon2))

def side_boundary(corner: tuple, width: float, height: float) -> list:
    """
    Computes the coordinates of the sides of a rectangle given the coordinates of the upper left
    corner, the width and the height of the rectangle.
    Parameters
    ----------
    corner : tuple
        Coordinates of the upper left corner of the rectangle in decimal degree format
        (latitude, longitude)
    width : float
        Width of the rectangled [nm]
    height : float
        Height of the rectangle [nm]
    Returns
    -------
    list
        Coordinates of the sides of the rectangle in decimal degree format (lon_west, lat_south,
        lon_east, lat_north)
    """
    # Determine the coordinates of the sides of the rectangle and return them
    left = corner[1]
    right = new_pos_dist((corner[0], corner[1]), width, 90)[1]
    top = corner[0]
    bottom = new_pos_dist((corner[0], corner[1]), height, 180)[0]
    return [left, bottom, right, top]

def vertice_boundary(corner: tuple, width: float, height: float) -> list:
    """
    Computes the coordinates of the vertices of a rectangle given the coordinates of the upper left
    corner, the width and the height of the rectangle.
    Parameters
    ----------
    corner : tuple
        Coordinates of the upper left corner of the rectangle in decimal degrees
        (latitude, longitude)
    width : float
        Width of the rectangled [nm]
    height : float
        Height of the rectangle [nm]
    Returns
    -------
    list
        Coordinates of the vertices of the rectangle in decimal degrees (upper left, upper right,
        lower left, lower right)
    """
    # ul = upper left, ur = upper right, ll = lower left, lr = lower right
    lat_ul, lon_ul = corner[0], corner[1]
    lat_ur, lon_ur = new_pos_dist((lat_ul, lon_ul), width, 90)
    lat_ll, lon_ll = new_pos_dist((lat_ul, lon_ul), height, 180)
    lat_lr, lon_lr = new_pos_dist((lat_ll, lon_ll), width, 90)

    return [(lat_ul, lon_ul), (lat_ur, lon_ur), (lat_ll, lon_ll), (lat_lr, lon_lr)]

# Visualisation-------------------------------------------------------------------------------------
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