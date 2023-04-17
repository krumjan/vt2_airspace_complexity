import math
from typing import Tuple


def new_pos_dist(
    pos: Tuple[float, float], distance: float, direction: float
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
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(dist_meters / 6378137.0)
        + math.cos(lat_rad)
        * math.sin(dist_meters / 6378137.0)
        * math.cos(dir_rad)
    )
    new_lon_rad = lon_rad + math.atan2(
        math.sin(dir_rad)
        * math.sin(dist_meters / 6378137.0)
        * math.cos(lat_rad),
        math.cos(dist_meters / 6378137.0)
        - math.sin(lat_rad) * math.sin(new_lat_rad),
    )

    # Convert new latitude and longitude to degrees
    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    # Return new position
    return new_lat, new_lon
