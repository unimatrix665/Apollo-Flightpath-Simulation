import astropy.constants as const
import astropy.units as u
from astropy.coordinates.representation import cartesian
from astropy.time import Time
import astropy.coordinates
from astropy.coordinates import (
    solar_system_ephemeris,
    get_body_barycentric_posvel,
    get_body,
    SkyCoord,
    get_body_barycentric)
import numpy as np
from astropy.coordinates import SkyCoord

time = Time("1969-07-16T00:00:00", scale="utc")

def get_moon_coord(time):
    with solar_system_ephemeris.set("jpl"):
        moon_pos_geo = get_body("moon", time)
        return moon_pos_geo.obsgeoloc

def get_moon_vel(time):
    with solar_system_ephemeris.set("jpl"):
        moon_pos_bary = get_body_barycentric()

def testfuncposgeo(epoch):
    with solar_system_ephemeris.set("jpl"):
        moon_pos_bary, moon_vel_bary = get_body_barycentric_posvel("moon", epoch)
        earth_pos_bary, earth_vel_bary = get_body_barycentric_posvel("earth", epoch)
        moon_pos_geo = moon_pos_bary - earth_pos_bary
        moon_vel_geo = moon_vel_bary - earth_vel_bary
        return moon_pos_geo

print(get_moon_coord(time))
print(testfuncposgeo(time))