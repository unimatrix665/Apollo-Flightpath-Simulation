import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import constants as const
from astropy.coordinates import (
    solar_system_ephemeris,
    get_body_barycentric_posvel,
)
import numpy as np
from math import sqrt

# Epoch
epoch = Time ("1969-07-16T00:00:00", scale="utc")
epoch1 = Time ("1969-08-16T00:00:00", scale="utc")
epoch2= Time ("1969-09-16T00:00:00", scale="utc")
mass_earth = const.M_earth

# Use a higher-accuracy JPL ephemeris if available
def tempfunc(epoch):
    with solar_system_ephemeris.set("jpl"):
        moon_pos_bary, moon_vel_bary = get_body_barycentric_posvel("moon", epoch)
        earth_pos_bary, earth_vel_bary = get_body_barycentric_posvel("earth", epoch)

    # Convert from solar-system-barycentric to geocentric vectors
    moon_pos_geo = moon_pos_bary - earth_pos_bary
    moon_vel_geo = moon_vel_bary - earth_vel_bary

    print("Position:", moon_pos_geo.xyz.to(u.km))
    print("Velocity:", moon_vel_geo.xyz.to(u.km / u.s))

#tempfunc(epoch)
#tempfunc(epoch1)
#tempfunc(epoch2)



t0 = Time("2026-09-08T00:00:00", scale="tdb")

n = 1_000_000
step = 0.1 * u.s

indices = np.arange(n, dtype=np.int64)
times = t0 + indices * step

#for t in times:
    #print(t)

def testfunc2(epoch):
    with solar_system_ephemeris.set("jpl"):
        moon_pos_bary, moon_vel_bary = get_body_barycentric_posvel("moon", epoch)
        earth_pos_bary, earth_vel_bary = get_body_barycentric_posvel("earth", epoch)
        moon_pos_geo = moon_pos_bary - earth_pos_bary
        moon_vel_geo = moon_vel_bary - earth_vel_bary
        moon_distance_geo = sqrt(moon_pos_geo.x.value **2 + moon_pos_geo.y.value **2 + moon_pos_geo.z.value **2)
        moon_distance_geo = moon_distance_geo * u.km

        return moon_distance_geo

def testfuncposgeo(epoch):
    with solar_system_ephemeris.set("jpl"):
        moon_pos_bary, moon_vel_bary = get_body_barycentric_posvel("moon", epoch)
        earth_pos_bary, earth_vel_bary = get_body_barycentric_posvel("earth", epoch)
        moon_pos_geo = moon_pos_bary - earth_pos_bary
        moon_vel_geo = moon_vel_bary - earth_vel_bary
        return moon_pos_geo

def testfuncplot(epoch):
    distances = []
    for i in range(100):
        distances.append(testfunc2(epoch).value)
        epoch = epoch + 1*u.day
    t = np.linspace(0, 100, 100) #erstellt nen array? mit num zahlen in gleichen abständen zwischen start und stop

    fig, ax = plt.subplots()
    ax.plot(t, distances)
    plt.show()

def testfuncplotmap(epoch):
    pos_x = []
    pos_y = []
    for i in range(600): #range hier ist anzahl zeit steps
        pos_x.append(testfuncposgeo(epoch).x.value)
        pos_y.append(testfuncposgeo(epoch).y.value)

        epoch = epoch + 1*u.hour

    fig, ax = plt.subplots()
    ax.plot(0,0,'o')
    ax.plot(pos_x, pos_y,)
    plt.show()

testmap = {
    epoch :  "test",
    epoch1 : "test2"
}

