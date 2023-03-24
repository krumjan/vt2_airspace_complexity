import glob
import math
import multiprocessing as mp
import os
import pandas as pd

from pathlib import Path
from shapely.geometry.base import BaseGeometry

from tqdm.auto import tqdm
from traffic.core import Traffic
from traffic.data import opensky
from typing import Union, Tuple

# Data fetching and combination --------------------------------------------------------
def download_adsb(
    t0: str,
    tf: str,
    folder: str,
    area: Union[None, str, BaseGeometry, Tuple[float, float, float, float]],
    lower: float,
    upper: float,
) -> None:
    """
    Queries ADS-B data from Opensky Network for the given time interval, geographical
    footprint and altitude constraints and saves the data as parquetet traffic object.

    Parameters
    ----------
    t0 : str
        Start datetime for the query in the format "YYYY-MM-DD hh:mm:ss"
    tf : str
        Stop datetime for the query in the format "YYYY-MM-DD hh:mm:ss"
    folder : str
        Path to the folder where the data will be saved
    area : Union[None, str, BaseGeometry, Tuple[float, float, float, float]]
        Geographical footprint. Can be a string (e.g. "LFBB"), a shapely geometry or a
        tuple of floats (lon_west, lat_south, lon_east, lat_north)
    lower : float
        Lower altitude constraint in feet
    upper : float
        Upper altitude constraint in feet
    """
    # check whether file already exists
    year = t0.year
    month = t0.month
    path = f"{folder}/{year}_{month}"
    check_file = Path(f"{path}/{t0.date()}_{tf.date()}.parquet")
    # if not, print which day is being downloaded and download
    if check_file.is_file() is False:
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
            if not os.path.exists(path):
                os.makedirs(path)
            traffic_data.to_parquet(f"{path}/{t0.date()}_{tf.date()}.parquet")
                
def download_adsb_para(
    start: str,
    stop: str,
    folder: str,
    bounds: Union[None, str, BaseGeometry, Tuple[float, float, float, float]],
    lower: float,
    upper: float,
    max_process: int = 8,
) -> None:
    """
    Parllelisation of the download_adsb function. Queries ADS-B data from Opensky
    Network for the given time interval, geographical footprint, altitude constraints
    and saves the data as Traffic in a parquet file format one day at a time.

    Parameters
    ----------
    start : str
        Start datetime for the query in the format "YYYY-MM-DD"
    stop : str
        Stop datetime for the query in the format "YYYY-MM-DD"
    folder : str
        Path to the folder where the data will be saved as one parquet file per day
    bounds : Union[None, str, BaseGeometry, Tuple[float, float, float, float]]
        Geographical footprint. Can be a string (e.g. "LFBB"), a shapely geometry or a
        tuple of floats (lon_west, lat_south, lon_east, lat_north)
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

def combine_adsb(
    path_raw: str,
    path_combined: str
) -> None:
    """
    Combines all parquet files in each folder in the provided path "path_raw" into
    one parquet file and saves it in the "path_combined" folder. Data is treated by
    year and month.

    Parameters
    ----------
    path_raw : str
        Folder path where the daily parquet files are stored
    path_combined : str
        Folder path where the combined parquet file will be stored
    """
    # create combined folder if it does not exist
    if os.path.isdir(path_combined) is False:
        os.mkdir(path_combined)
    # for each year and month folder
    folders = glob.glob(f"{path_raw}/*/", recursive = True)
    for folder in tqdm(folders):
        year = folder.split('/')[3].split('_')[0]
        month = folder.split('/')[3].split('_')[1]
        # check whether combined file already exists
        check_file = Path(f"{path_combined}/combined_{year}_{month}.parquet")
        if check_file.is_file() is False:
            # read all parquets in the folder and combine them into one Traffic object
            files = glob.glob(f"{folder}/*.parquet", recursive=True)
            alldata = Traffic(
                pd.concat([Traffic.from_file(file).data for file in files],
                        ignore_index=True)
            )
            # save combined file in the combined folder
            alldata.to_parquet(f"{path_combined}/combined_{year}_{month}.parquet")

# Data preprocessing -------------------------------------------------------------------
def preprocess_adsb(
        path_get: str,
        path_save: str,
) -> None:
    """
    Preprocesses the ADS-B data and saves it in the provided path "path_save". Data is
    treated by year and month.
    Preprocessing includes:
        - assigning an id to each trajectory
        - removing invalid trajectories
        - applying filtering to the trajectories
        - resampling the trajectories to 5s intervals

    Parameters
    ----------
    path_get : str
        Path to the folder where the not yet preprocessed parquet files are stored
    path_save : str
        Path to the folder where the preprocessed parquet files will be stored
    """
    # create preprocessed folder if it does not exist
    if os.path.isdir(path_save) is False:
        os.mkdir(path_save)
    # iterate over each monthly file
    files = glob.glob(f"{path_get}/*.parquet", recursive=True)
    for file in tqdm(files):
        # if file has not been preprocessed yet
        year = file.split('/')[3].split('_')[1]
        month = file.split('/')[3].split('_')[2]
        check_file = Path(f'{path_save}/preprocessed_{year}_{month}.parquet')
        if check_file.is_file() is False:
            # load data
            trajs = Traffic.from_file(file)
            # preprocess data
            trajs_proc =(
                trajs.clean_invalid()
                .assign_id()
                .filter()
                .resample("5s")
                .eval(max_workers= 20, desc='resampling')
            )
            # type correction due to traffic bug
            x = trajs.data.dtypes.to_dict()
            del x['last_position']
            trajs_save = Traffic(trajs_proc.data.astype(x))
            # save data
            trajs_save.to_parquet(f'{path_save}/preprocessed_{year}_{month}.parquet')

# Other functions ----------------------------------------------------------------------
def new_pos_dist(
    pos: Tuple[float, float],
    distance: float,
    direction: float
) -> Tuple[float, float]:
    """
    Calculates the new position given an initial position, a distance and a direction.

    Parameters
    ----------
    pos : Tuple[float, float]
        Initial position (latitude, longitude)
    distance : float
        Distance in nautical miles
    direction : float
        Direction/bearing in degrees

    Returns
    -------
    Tuple[float, float]
        New position (latitude, longitude)
    """
    # Convert latitude and longitude to radians
    lat = pos[0]
    lon = pos[1]
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Convert nautical miles to meters
    dist_meters = distance * 1852.0

    # Convert direction to radians
    dir_rad = math.radians(direction)

    # Calculate new latitude and longitude
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(dist_meters/6378137.0) +
                            math.cos(lat_rad) * math.sin(dist_meters/6378137.0) *
                            math.cos(dir_rad))
    new_lon_rad = lon_rad + math.atan2(math.sin(dir_rad) *
                                       math.sin(dist_meters/6378137.0) *
                                       math.cos(lat_rad),
                                       math.cos(dist_meters/6378137.0) -
                                       math.sin(lat_rad) * math.sin(new_lat_rad))

    # Convert new latitude and longitude to degrees
    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    # Return new position
    return new_lat, new_lon
