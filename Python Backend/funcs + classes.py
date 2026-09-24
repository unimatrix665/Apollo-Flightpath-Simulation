import astropy.constants as const
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    solar_system_ephemeris,
    get_body_barycentric_posvel,
    get_body,
    SkyCoord,
    get_body_barycentric)
from math import sqrt
import numpy as np

time = Time("1969-07-16T00:00:00", scale="utc")
delta_t = 1 * u.s

class Himmelskörper:
    def __init__(self):
        self.pos_geo = get_moon_coord(time)
        self.vel = get_moon_vel(time)
        self.mass = 0 * u.kg

    def set_vel(self, value):
        self.vel = value

    def set_pos(self, value):
        self.vel = value

    def set_mass(self, value):
        self.mass = value

    def print_Vals(self):
        print(self.pos_geo)
        print(self.vel)
        print(self.mass)

class Apollo(Himmelskörper):
    def __init__(self):
        super().__init__()

class Erde:
    def __init__(self):
        self.mass = const.M_earth

def get_moon_coord(time):
    with solar_system_ephemeris.set("jpl"):
        moon_pos_bary = get_body_barycentric("moon", time)
        earth_pos_bary = get_body_barycentric("earth", time)
        moon_pos_geo = moon_pos_bary - earth_pos_bary
        return moon_pos_geo

def get_moon_vel(time):
    with solar_system_ephemeris.set("jpl"):
        moon_pos_bary, moon_vel_bary = get_body_barycentric_posvel("moon", time)
        return moon_vel_bary

def get_moon_distance(time, moon_pos_geo):
    moon_distance_geo = sqrt(moon_pos_geo.x.value ** 2 + moon_pos_geo.y.value ** 2 + moon_pos_geo.z.value ** 2)
    moon_distance_geo = moon_distance_geo * u.km
    return moon_distance_geo

def get_accel(Body, ref_Body,):
    distance = sqrt(pow(Body.pos_geo.x.value - ref_Body.pos_geo.x.value, 2) +
                    pow(Body.pos_geo.y.value - ref_Body.pos_geo.y.value, 2) +
                    pow(Body.pos_geo.z.value - ref_Body.pos_geo.z.value, 2)
    )
    F = const.G * ref_Body.mass**2/distance**2
    a = F / ref_Body.mass
    dif_x = ref_Body.pos_geo.x - Body.pos_geo.x
    dif_y = ref_Body.pos_geo.y - Body.pos_geo.y
    dif_z = ref_Body.pos_geo.z - Body.pos_geo.z
    dif = dif_x + dif_y + dif_z
    partition_x = dif_x / dif
    partition_y = dif_y / dif
    partition_z = dif_z / dif
    Body.vel.x = Body.vel.x + partition_x * a * delta_t.to(u.s)
    Body.vel.y = Body.vel.y + partition_y * a * delta_t.to(u.s)
    Body.vel.z = Body.vel.z + partition_z * a * delta_t.to(u.s)

def get_accel2(body, ref_body):
    """Acceleration of `body` due to `ref_body`'s gravity, in km/s^2."""
    diff = ref_body.pos_geo.xyz.to(u.km).value - body.pos_geo.xyz.to(u.km).value  # ref - body
    distance = np.linalg.norm(diff)
    direction = diff / distance                     # unit vector, points toward ref_body

    mu = const.G.to(u.km**3 / (u.kg * u.s**2)).value * ref_body.mass.to(u.kg).value
    return mu / distance**2 * direction              # km/s^2

moon = Himmelskörper
Apollo = Himmelskörper