import glob
import multiprocessing as mp
import os

from typing import Union, Tuple
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from shapely.geometry.base import BaseGeometry

from traffic.core import Traffic
from traffic.data import opensky


def download_adsb(
    t0: str,
    tf: str,
    folder: str,
    area: Union[None, str, BaseGeometry, Tuple[float, float, float, float]],
    lower: float,
    upper: float,
) -> None:
    """
    Queries ADS-B data from Opensky Network for the time interval, geographical
    footprint and altitude constraints defined trough the parameters and saves the data
    as parquetet traffic object in the folder also defined by the parameters.

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
        Lower altitude constraint in meters
    upper : float
        Upper altitude constraint in meters
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
            cached=False,
            other_params=f" and baroaltitude >= {lower} and baroaltitude <= {upper}\
                  and onground = false ",
        )
        # if not empty, save. Otherwise, print "empty day"
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
    Parllelisation application of the download_adsb() function. Queries ADS-B data from
    Opensky Network for the time interval, geographical footprint and altitude
    constraints defined trough the parameters and saves the data as parquetet traffic
    object in the folder also defined by the parameters.

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
        Number of processes to use for parallelization, by default 8. It is not advised
        to use more than 8 processes as it will not improve the speed of the download
        due to limitations on the side of Opensky Netork.
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
    # run the download_adsb function in parallel processes
    with mp.Pool(max_process) as pool:
        pool.starmap(download_adsb, t)


def combine_adsb(path_raw: str, path_combined: str) -> None:
    """
    Combines daily parquet files situated in the same sobfolder of "path_raw" into one
    parquet file and saves the combined file in the "path_combined" folder.

    Parameters
    ----------
    path_raw : str
        Folder where subfolders with daily parquet files are stored
    path_combined : str
        Folder path where the combined parquet file will be stored
    """

    # create combined folder if it does not exist
    if os.path.isdir(path_combined) is False:
        os.mkdir(path_combined)
    # for each year and month folder
    folders = glob.glob(f"{path_raw}/*/", recursive=True)
    for folder in tqdm(folders):
        year = folder.split("/")[-2].split("_")[0]
        month = folder.split("/")[-2].split("_")[1]
        # check whether combined file already exists
        check_file = Path(f"{path_combined}/combined_{year}_{month}.parquet")
        if check_file.is_file() is False:
            # read all parquets in the folder and combine them into one Traffic object
            files = glob.glob(f"{folder}/*.parquet", recursive=True)
            alldata = Traffic(
                pd.concat(
                    [Traffic.from_file(file).data for file in files],
                    ignore_index=True,
                )
            )
            # save combined file in the combined folder
            alldata.to_parquet(
                f"{path_combined}/combined_{year}_{month}.parquet"
            )


def preprocess_adsb(
    path_get: str,
    path_save: str,
) -> None:
    """
    Preprocesses each parqueted traffic object in the "path_get" folder and saves the
    preprocessed traffic object in the "path_save" folder.
    Preprocessing steps includes:
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
        Path(path_save).mkdir(parents=True, exist_ok=True)
    # iterate over each monthly file
    files = glob.glob(f"{path_get}/*.parquet", recursive=True)
    for file in tqdm(files):
        # if file has not been preprocessed yet
        year = file.split("/")[-1].split("_")[1]
        month = file.split("/")[-1].split("_")[2].split(".")[0]
        check_file = Path(f"{path_save}/preprocessed_{year}_{month}.parquet")
        if check_file.is_file() is False:
            # load data
            trajs = Traffic.from_file(file)
            # preprocess data
            trajs_proc = (
                trajs.clean_invalid()
                .assign_id()
                .filter()
                .resample("5s")
                .eval(max_workers=20, desc="resampling")
            )
            # type correction due to traffic bug
            x = trajs.data.dtypes.to_dict()
            del x["last_position"]
            trajs_save = Traffic(trajs_proc.data.astype(x))
            # save data
            trajs_save.to_parquet(
                f"{path_save}/preprocessed_{year}_{month}.parquet"
            )
