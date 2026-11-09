"""
Ephemeris, force model and propagation for the Apollo 11 reconstruction.

Conventions (kept fixed everywhere, so nothing ever needs unit conversion):
    * lengths in km, velocities in km/s, times in GET seconds (see apollo11_data)
    * inertial frame: Earth-centred, axes aligned with J2000 (astropy GCRS/ICRS)
    * state vectors and positions are PLAIN numpy arrays. astropy objects only
      exist inside Ephemeris.__init__ and are converted to ndarrays immediately.
"""
import numpy as np
import erfa
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import apollo11_data as D

# 'builtin' needs no download. On a machine that can reach NAIF, set this to
# "de440" (or "jpl") for a JPL kernel -- nothing else in the code changes.
EPHEMERIS = "JPL"

RANGE_ZERO = Time(D.RANGE_ZERO_UTC, scale="utc")


def get_time(get_seconds):
    """astropy Time for a GET in seconds (scalar or array)."""
    return RANGE_ZERO + np.asarray(get_seconds) * u.s


# ---------------------------------------------------------------------------
# Moon / Sun position, relative to Earth, as smooth functions of GET
# ---------------------------------------------------------------------------
def get_moon_coord(time):
    """Moon relative to Earth (astropy CartesianRepresentation, astropy boundary only)."""
    with solar_system_ephemeris.set(EPHEMERIS):
        return get_body_barycentric("moon", time) - get_body_barycentric("earth", time)


def get_sun_coord(time):
    with solar_system_ephemeris.set(EPHEMERIS):
        return get_body_barycentric("sun", time) - get_body_barycentric("earth", time)


class Ephemeris:
    """Moon & Sun geocentric position (km) as cubic splines of GET.

    Built once with ONE vectorised astropy call per body, then evaluated
    cheaply thousands of times by the integrator. A cubic spline over a 60 s
    grid is far smoother than linear interpolation (the Moon's acceleration
    is continuous, so the spline's error is negligible: << 1 m)."""

    def __init__(self, get_start, get_end, step=60.0):
        self.t = np.arange(get_start, get_end + step, step)
        times = get_time(self.t)
        self._moon = CubicSpline(self.t, get_moon_coord(times).xyz.to(u.km).value.T)
        self._sun = CubicSpline(self.t, get_sun_coord(times).xyz.to(u.km).value.T)

    def moon(self, t):
        return self._moon(t)

    def moon_vel(self, t):
        return self._moon(t, 1)

    def sun(self, t):
        return self._sun(t)


# ---------------------------------------------------------------------------
# Frames: Apollo's published state is in the mean equator/equinox OF DATE
# (1969); our integrator works in J2000 axes. erfa.pmat06 is the
# bias-precession rotation J2000 -> of-date.
# ---------------------------------------------------------------------------
def precession_matrix(get_seconds):
    t = get_time(get_seconds).tt
    return erfa.pmat06(t.jd1, t.jd2)         # J2000 -> mean of date


def state_from_geocentric(r, dec_deg, ra_deg, speed, fpa_deg, heading_rad, get_seconds):
    """Inertial (J2000) position & velocity from the Apollo-style description:
    geocentric distance, declination, right ascension (of date), space-fixed
    speed, flight-path angle above local horizontal, heading east of north."""
    dec, ra, fpa = np.radians(dec_deg), np.radians(ra_deg), np.radians(fpa_deg)
    up = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    north = np.cross(up, east)
    vel_hat = np.cos(fpa) * (np.sin(heading_rad) * east + np.cos(heading_rad) * north) + np.sin(fpa) * up
    to_j2000 = precession_matrix(get_seconds).T
    return to_j2000 @ (r * up), to_j2000 @ (speed * vel_hat)


def declination_and_fpa(pos, vel, get_seconds):
    """Geocentric latitude (deg) and flight-path angle (deg) of a J2000 state."""
    P = precession_matrix(get_seconds)
    pd = P @ pos
    r = np.linalg.norm(pd)
    dec = np.degrees(np.arcsin(pd[2] / r))
    fpa = np.degrees(np.arcsin((pos @ vel) / (np.linalg.norm(pos) * np.linalg.norm(vel))))
    return dec, fpa


# ---------------------------------------------------------------------------
# Force model
# ---------------------------------------------------------------------------
class Dynamics:
    """Spacecraft acceleration in the Earth-centred inertial frame.

    Earth: point mass + J2.  Moon and Sun: third-body terms INCLUDING the
    indirect term. The indirect term matters: the frame is centred on Earth,
    which is itself accelerating toward the Moon and Sun, so the correct
    acceleration is  mu * ( (r_b - r)/|r_b - r|^3  -  r_b/|r_b|^3 ).
    Leaving it out (treating Earth as fixed) overstates lunar/solar effects
    by an amount comparable to the effect itself."""

    def __init__(self, ephemeris, ref_time=D.GET_TLI):
        self.eph = ephemeris
        self.pole = precession_matrix(ref_time).T @ np.array([0.0, 0.0, 1.0])

    def accel(self, t, pos):
        r = np.linalg.norm(pos)
        a = -D.MU_EARTH * pos / r**3
        z = pos @ self.pole
        k = 1.5 * D.J2_EARTH * D.MU_EARTH * D.R_EARTH**2 / r**5
        a = a + k * ((5.0 * z * z / (r * r) - 1.0) * pos - 2.0 * z * self.pole)
        for mu, body in ((D.MU_MOON, self.eph.moon(t)), (D.MU_SUN, self.eph.sun(t))):
            d = body - pos
            a = a + mu * (d / np.linalg.norm(d) ** 3 - body / np.linalg.norm(body) ** 3)
        return a

    def rhs(self, t, y):
        return np.concatenate([y[3:], self.accel(t, y[:3])])

    def propagate(self, state, t0, t1, events=None):
        """High-order adaptive propagation (DOP853). Returns the scipy solution
        object (dense output + events). This replaces a fixed-step loop: it
        takes tiny steps near perigee and huge ones in deep space automatically."""
        return solve_ivp(self.rhs, [t0, t1], state, method="DOP853",
                         rtol=1e-11, atol=1e-9, dense_output=True, events=events)


# ---------------------------------------------------------------------------
# Lambert solver (two-body transfer between two position vectors in a given
# time). Used only to SEED the return-leg fit; the full force model refines it.
# ---------------------------------------------------------------------------

def _stumpff(z):
    if abs(z) < 1e-8: return 0.5, 1.0/6.0
    if z > 0:
        s = np.sqrt(z); return (1-np.cos(s))/z, (s-np.sin(s))/s**3
    s = np.sqrt(-z); return (np.cosh(s)-1)/(-z), (np.sinh(s)-s)/s**3

def lambert(mu, r1, r2, tof, prograde=True, axis=np.array([0,0,1.0])):
    """Universal-variable Lambert solver (Bate-Mueller-White). Returns v1, v2."""
    r1n, r2n = np.linalg.norm(r1), np.linalg.norm(r2)
    cosd = np.clip(r1@r2/(r1n*r2n), -1, 1)
    h = np.cross(r1, r2)
    dnu = np.arccos(cosd)
    if (h@axis >= 0) != prograde: dnu = 2*np.pi - dnu
    A = np.sin(dnu)*np.sqrt(r1n*r2n/(1-cosd))
    if abs(A) < 1e-12: raise ValueError("degenerate")
    def y(z):
        C,S = _stumpff(z); return r1n+r2n + A*(z*S-1)/np.sqrt(C)
    def F(z):
        C,S = _stumpff(z); yy = y(z)
        return (yy/C)**1.5*S + A*np.sqrt(yy) - np.sqrt(mu)*tof
    # find z lower bound such that y>0
    lo = -4*np.pi**2*4
    while y(lo) < 0 and lo < 0: lo *= 0.5
    zl = lo
    if A>0:
        # y(z) must be >0: increase z until y>0
        zl=-100.0
        while y(zl)<=0: zl+=1.0
    hi = 4*np.pi**2 - 1e-6
    zs = np.linspace(zl, hi, 400); fs=[]
    for z in zs:
        try: fs.append(F(z))
        except Exception: fs.append(np.nan)
    fs=np.array(fs); root=None
    for i in range(len(zs)-1):
        if np.isfinite(fs[i]) and np.isfinite(fs[i+1]) and fs[i]*fs[i+1] < 0:
            root = brentq(F, zs[i], zs[i+1], xtol=1e-12); break
    if root is None: raise ValueError("no root")
    yy = y(root)
    f = 1 - yy/r1n; g = A*np.sqrt(yy/mu); gd = 1 - yy/r2n
    return (r2 - f*r1)/g, (gd*r2 - r1)/g
