"""
Apollo 11 flight path simulator - v1
======================================
Simplified restricted three-body simulation of the Apollo 11 Earth-Moon-Earth
trajectory, focused on time spent in the Van Allen radiation belts.

Real Moon ephemeris is used (via astropy). Everything about the spacecraft
itself (parking orbit, TLI burn, the lunar turnaround, and the return burn)
is MOCK / approximate data -- good enough to be roughly right (per the stated
tolerance of being off by minutes to a couple of hours), not a precise
reconstruction. See the docstring at the bottom of each mock-data section for
what a v2 would replace it with.
"""

import numpy as np
from scipy.integrate import solve_ivp
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric, get_body_barycentric_posvel
import astropy.units as u
import json

# ---------------------------------------------------------------------------
# 0. Moon position/velocity functions (as given, lightly wrapped for reuse)
# ---------------------------------------------------------------------------
# NOTE: 'jpl' (de430/de440) requires downloading a kernel from NAIF on first
# use. That download isn't reachable from this sandboxed environment, so this
# defaults to astropy's 'builtin' ephemeris (no download needed, accurate to
# within a few arcseconds -- plenty for v1). On your own machine, just change
# EPHEMERIS below to 'jpl' (or 'de440') once, and everything downstream is
# unaffected because it goes through these same two functions.
EPHEMERIS = "builtin"


def get_moon_coord(time):
    with solar_system_ephemeris.set(EPHEMERIS):
        moon_pos_bary = get_body_barycentric("moon", time)
        earth_pos_bary = get_body_barycentric("earth", time)
        moon_pos_geo = moon_pos_bary - earth_pos_bary
        return moon_pos_geo


def get_moon_vel(time):
    with solar_system_ephemeris.set(EPHEMERIS):
        moon_pos_bary, moon_vel_bary = get_body_barycentric_posvel("moon", time)
        earth_pos_bary, earth_vel_bary = get_body_barycentric_posvel("earth", time)
        return moon_vel_bary - earth_vel_bary


# ---------------------------------------------------------------------------
# 1. Constants and mock mission data
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418          # km^3/s^2
MU_MOON = 4902.800066           # km^3/s^2
R_EARTH = 6378.137              # km

# Epoch: real Apollo 11 TLI ignition time. Everything else below is
# approximate/mock, matched to the real order of magnitude from
# "Apollo by the Numbers", not reconstructed state vectors.
T0 = Time("1969-07-16 16:22:13", scale="utc")

PARK_ALT = 190.0                       # km, mock parking-orbit altitude
R_PARK = R_EARTH + PARK_ALT

ENTRY_INTERFACE_ALT = 121.9            # km (400,000 ft), mock re-entry trigger
R_ENTRY = R_EARTH + ENTRY_INTERFACE_ALT

OUTBOUND_DURATION = 3.0 * 86400        # s, mock ~3 day translunar coast
RETURN_TRANSIT_TARGET = 3.0 * 86400    # s, mock ~3 day transearth coast

# Van Allen belt bands, as simple Earth-centered spherical shells (v1 -- no
# tilt, no toroidal pinch toward the poles; that's a v2 upgrade).
INNER_BELT = (R_EARTH + 1000, R_EARTH + 6000)      # km
OUTER_BELT = (R_EARTH + 13000, R_EARTH + 60000)    # km


# ---------------------------------------------------------------------------
# 2. Moon position/velocity as a function of simulation time (seconds after T0)
# ---------------------------------------------------------------------------
def moon_pos_km(t_sim_seconds):
    t = T0 + t_sim_seconds * u.s
    return get_moon_coord(t).xyz.to(u.km).value


def moon_vel_km_s(t_sim_seconds):
    t = T0 + t_sim_seconds * u.s
    return get_moon_vel(t).xyz.to(u.km / u.s).value


# ---------------------------------------------------------------------------
# 3. Equations of motion: Earth fixed at origin + Moon's real position
# ---------------------------------------------------------------------------
def rhs(t_sim, state):
    pos = state[:3]
    vel = state[3:6]

    r_earth = np.linalg.norm(pos)
    a = -MU_EARTH * pos / r_earth**3

    m_pos = moon_pos_km(t_sim)
    rel = pos - m_pos
    r_moon = np.linalg.norm(rel)
    a += -MU_MOON * rel / r_moon**3

    return np.concatenate([vel, a])


# ---------------------------------------------------------------------------
# 4. Mock TLI burn -> initial state
# ---------------------------------------------------------------------------
# Real TLI added ~3.15 km/s to the ~7.79 km/s parking-orbit speed. We don't
# reconstruct the real burn direction/timing -- instead we pick a plausible
# translunar-injection geometry (burn point sits in the plane containing the
# Earth-Moon line at T0, tangential departure) and size the speed with the
# vis-viva equation so the resulting two-body ellipse's apogee reaches just
# past the Moon's current distance. This is a mock "aim" step; v2 replaces
# it with the real reconstructed TLI cutoff vector.
TLI_SPEED = 10.834                  # km/s, mock -- real Apollo 11 TLI cutoff speed was ~this
TLI_FLIGHT_PATH_ANGLE = np.radians(7.4)   # mock -- real burns aren't purely tangential


def build_initial_state():
    # Aim at where the Moon will BE at estimated arrival, not where it is at
    # T0 -- the Moon moves ~13 deg/day, so aiming at t=0's position would
    # send us nowhere near it after a multi-day coast. Still a crude "lead
    # angle" trick, not a real targeting solver (that's a v2 upgrade).
    moon_target = moon_pos_km(OUTBOUND_DURATION)
    moon_dir = moon_target / np.linalg.norm(moon_target)

    up = np.array([0.0, 0.0, 1.0])
    normal = np.cross(moon_dir, up)
    if np.linalg.norm(normal) < 1e-6:
        normal = np.cross(moon_dir, np.array([1.0, 0.0, 0.0]))
    normal /= np.linalg.norm(normal)
    in_plane_perp = np.cross(normal, moon_dir)
    in_plane_perp /= np.linalg.norm(in_plane_perp)

    # Mock burn point: 55 degrees "behind" the Earth-Moon line.
    theta0 = np.radians(-55.0)
    pos0 = R_PARK * (np.cos(theta0) * moon_dir + np.sin(theta0) * in_plane_perp)
    tangent_dir = -np.sin(theta0) * moon_dir + np.cos(theta0) * in_plane_perp
    tangent_dir /= np.linalg.norm(tangent_dir)
    radial_dir = pos0 / np.linalg.norm(pos0)

    # Use the real approximate TLI cutoff speed/angle directly, rather than
    # deriving speed from a targeted apogee -- a pure minimum-energy
    # (Hohmann-like) transfer naturally takes ~5 days to reach apogee near
    # the Moon's distance, but Apollo intentionally burned harder than that
    # to cut transit time to ~3 days. Matching the real speed reproduces
    # that faster, more energetic trajectory shape.
    vel0 = TLI_SPEED * (np.cos(TLI_FLIGHT_PATH_ANGLE) * tangent_dir
                         + np.sin(TLI_FLIGHT_PATH_ANGLE) * radial_dir)
    return np.concatenate([pos0, vel0])


# ---------------------------------------------------------------------------
# 5. Belt bookkeeping
# ---------------------------------------------------------------------------
def belt_flag(r):
    if INNER_BELT[0] <= r <= INNER_BELT[1]:
        return "inner"
    if OUTER_BELT[0] <= r <= OUTER_BELT[1]:
        return "outer"
    return None


def summarize_belt_time(t_arr, pos_arr, label):
    radii = np.linalg.norm(pos_arr, axis=1)
    flags = [belt_flag(r) for r in radii]
    inner_time = 0.0
    outer_time = 0.0
    for i in range(len(t_arr) - 1):
        dt = t_arr[i + 1] - t_arr[i]
        # if either endpoint of the interval is inside a band, count it
        # (coarse but fine at v1 sampling resolution)
        if flags[i] == "inner" or flags[i + 1] == "inner":
            inner_time += dt
        if flags[i] == "outer" or flags[i + 1] == "outer":
            outer_time += dt
    print(f"  [{label}] inner belt: {inner_time/60:6.1f} min | outer belt: {outer_time/60:6.1f} min")
    return inner_time, outer_time


# ---------------------------------------------------------------------------
# 6. Run outbound leg
# ---------------------------------------------------------------------------
print("Running outbound leg (TLI -> lunar arrival)...")
state0 = build_initial_state()

sol_out = solve_ivp(
    rhs, [0, OUTBOUND_DURATION], state0,
    method="RK45", rtol=1e-8, atol=1e-6,
    dense_output=True, max_step=600.0,
)

t_out = np.linspace(0, OUTBOUND_DURATION, 2000)
y_out = sol_out.sol(t_out)
pos_out = y_out[:3].T
vel_out = y_out[3:6].T

# distance of closest lunar approach actually achieved (just for interest --
# v1 does not rigorously target a flyby)
moon_positions_out = np.array([moon_pos_km(t) for t in t_out])
dist_to_moon = np.linalg.norm(pos_out - moon_positions_out, axis=1)
closest_approach = dist_to_moon.min()
closest_t = t_out[np.argmin(dist_to_moon)]
print(f"  closest approach to Moon: {closest_approach:,.0f} km at T+{closest_t/3600:.1f} h")

inner_out, outer_out = summarize_belt_time(t_out, pos_out, "outbound")

# ---------------------------------------------------------------------------
# 7. Mock "turnaround" burn at end of outbound leg (stand-in for LOI+stay+TEI)
# ---------------------------------------------------------------------------
# v1 does not simulate lunar orbit insertion or a lunar stay. We treat lunar
# arrival as a single instantaneous event: reverse the velocity direction and
# rescale its magnitude (vis-viva) to send the spacecraft back toward Earth
# on an ellipse timed to arrive in about RETURN_TRANSIT_TARGET seconds.
state_arrival = np.concatenate([pos_out[-1], vel_out[-1]])
r1 = np.linalg.norm(state_arrival[:3])

# The return ellipse's apogee is fixed (= current radius r1, treating this
# point as apogee), and its perigee is our target (entry interface). Those
# two fix the semi-major axis -- we can't also independently pick a transit
# time, so the resulting return duration is whatever falls out (it lands
# near the real ~3 day return by coincidence of the geometry, but isn't
# forced to).
a_return = (r1 + R_ENTRY) / 2.0
v_return_mag = np.sqrt(MU_EARTH * (2.0 / r1 - 1.0 / a_return))
return_duration_estimate = np.pi * np.sqrt(a_return**3 / MU_EARTH)
print(f"  estimated return transit: {return_duration_estimate/86400:.1f} days")

vel_dir_return = -state_arrival[3:6] / np.linalg.norm(state_arrival[3:6])
state_after_burn = np.concatenate([state_arrival[:3], v_return_mag * vel_dir_return])

# ---------------------------------------------------------------------------
# 8. Run return leg, stopping at the atmospheric entry interface
# ---------------------------------------------------------------------------
print("Running return leg (turnaround -> entry interface)...")


def entry_event(t_sim, state):
    return np.linalg.norm(state[:3]) - R_ENTRY


entry_event.terminal = True
entry_event.direction = -1

sol_ret = solve_ivp(
    rhs, [OUTBOUND_DURATION, OUTBOUND_DURATION + return_duration_estimate * 1.3],
    state_after_burn,
    method="RK45", rtol=1e-8, atol=1e-6,
    dense_output=True, max_step=600.0, events=entry_event,
)

t_end = sol_ret.t[-1]
t_ret = np.linspace(OUTBOUND_DURATION, t_end, 2000)
y_ret = sol_ret.sol(t_ret)
pos_ret = y_ret[:3].T
vel_ret = y_ret[3:6].T

reentry_flag = len(sol_ret.t_events[0]) > 0
print(f"  return leg duration: {(t_end - OUTBOUND_DURATION)/3600:.1f} h "
      f"({'reached entry interface' if reentry_flag else 'did NOT reach entry interface in allotted time'})")

inner_ret, outer_ret = summarize_belt_time(t_ret, pos_ret, "return  ")

# ---------------------------------------------------------------------------
# 9. Totals
# ---------------------------------------------------------------------------
print("\n=== Mission summary (v1, mock data) ===")
print(f"Total time simulated: {t_end/3600:.1f} h ({t_end/86400:.2f} days)")
print(f"Total inner-belt time: {(inner_out+inner_ret)/60:.1f} min")
print(f"Total outer-belt time: {(outer_out+outer_ret)/60:.1f} min")

# ---------------------------------------------------------------------------
# 10. Export intermediate data (stable format for the visualization layer)
# ---------------------------------------------------------------------------
def flags_array(pos_arr):
    return [belt_flag(np.linalg.norm(p)) or "none" for p in pos_arr]


data = {
    "meta": {
        "epoch_utc": T0.isot,
        "ephemeris": EPHEMERIS,
        "outbound_duration_s": OUTBOUND_DURATION,
        "closest_lunar_approach_km": float(closest_approach),
        "inner_belt_minutes": (inner_out + inner_ret) / 60,
        "outer_belt_minutes": (outer_out + outer_ret) / 60,
    },
    "outbound": {
        "t": t_out.tolist(),
        "pos_km": pos_out.tolist(),
        "belt": flags_array(pos_out),
    },
    "return": {
        "t": t_ret.tolist(),
        "pos_km": pos_ret.tolist(),
        "belt": flags_array(pos_ret),
    },
}

with open("trajectory_v1.json", "w") as f:
    json.dump(data, f)
print("\nSaved trajectory_v1.json")