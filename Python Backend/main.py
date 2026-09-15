import astropy.constants as const
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    solar_system_ephemeris,
    get_body_barycentric_posvel)
import numpy as np
from astropy.coordinates import SkyCoord

from test import mass_earth

start_date = Time("1969-07-16T00:00:00", scale="utc")

class Himmelskörper:
    def __init__(self):
        mass = 0 * u.kg
        pos = [0,0,0]
        vel = [0,0,0]
        jpl_name = ""

    def get_pos_over_time(self):
        with solar_system_ephemeris.set("jpl"):
            self.pos, self.vel = get_body_barycentric_posvel(self.jpl_name, start_date)
            return self.pos

    def get_vel_over_time(self):
        with solar_system_ephemeris.set("jpl"):
            self.pos, self.vel = get_body_barycentric_posvel(self.jpl_name, start_date)
            return  self.vel

