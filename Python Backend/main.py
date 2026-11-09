"""
Apollo 11 trajectory reconstruction (Earth -> Moon -> Earth) from documented data.

What is DOCUMENTED and what is FITTED (see apollo11_data.py for sources):

  documented, used as input   TLI state vector; event times; burn magnitudes;
                              lunar-orbit altitudes; entry-interface conditions
  fitted (3 small things)     (1) MCC-2 burn direction, so the spacecraft reaches the
                                  documented pericynthion altitude and time
                              (2) the return trajectory's departure velocity, so it
                                  reaches the documented entry interface
                              (3) lunar stay is idealised (not simulated)
  checks (NOT fitted)         free-flight arrival from TLI alone, LOI/TEI delta-v,
                              entry speed, implied entry longitude

Run:  python main.py   ->  prints a validation report and writes
                           apollo11_draufsicht.png            (top-down trajectory)
                           apollo11_abflug_guertel.png        (outbound, Van Allen belts)
                           apollo11_rueckflug_guertel.png     (return, Van Allen belts)
                           apollo11_trajectory.json           (data export)
"""
import json
from dataclasses import dataclass

import erfa
import matplotlib
import numpy as np

matplotlib.use("Agg")           # remove this line to get interactive windows
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares, minimize_scalar

import apollo11_data as D
from funcs_classes import (Ephemeris, Dynamics, state_from_geocentric, precession_matrix,
                           declination_and_fpa, get_time, lambert)

# ---------------------------------------------------------------------------
# Constants and small helpers
# ---------------------------------------------------------------------------
H = 3600.0                       # seconds per hour

# Van Allen belts, v1 model: spherical shells around the Earth's centre (no geomagnetic tilt).
# Limits after Ripoll et al. (2020), the same values as in the paper (section 4.2.1):
# inner belt 1-3 Earth radii, outer belt 3-7 Earth radii, both measured from the Earth's
# CENTRE (not from the surface), with R_E = 6370 km.
R_E_BELT_KM = 6370.0
INNER_BELT_RE = (1.0, 3.0)
OUTER_BELT_RE = (3.0, 7.0)
BELT_WINDOW_S = 12 * H                 # how much of each end of the flight is analysed

FIG_TOP_VIEW = "apollo11_draufsicht.png"
FIG_OUTBOUND = "apollo11_abflug_guertel.png"
FIG_RETURN = "apollo11_rueckflug_guertel.png"
EXPORT_JSON = "apollo11_trajectory.json"

SUPTITLE = ("Apollo-11-Flugbahnrekonstruktion anhand dokumentierter Zustandsvektoren\n"
            "(Gravitation von Erde mit Abplattung, Mond und Sonne; Mondaufenthalt idealisiert)")

report = []


def log(text=""):
    print(text)
    report.append(text)


def unit(v):
    return v / np.linalg.norm(v)


def de(x, digits=1):
    """Zahl mit Dezimalkomma fuer deutsche Beschriftungen."""
    return f"{x:.{digits}f}".replace(".", ",")


def direction(theta, phi):
    """Unit vector from polar angle theta and azimuth phi."""
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def spherical_unit(dec, ra):
    """Unit vector from declination and right ascension (radians)."""
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
def setup_simulation():
    """Ephemeris, force model and the documented TLI state vector."""
    eph = Ephemeris(D.GET_TLI, D.GET_ENTRY_INTERFACE + H)
    dyn = Dynamics(eph)

    heading = np.arcsin(np.cos(np.radians(D.TLI_INCLINATION_DEG)) /
                        np.cos(np.radians(D.TLI_GEOCENTRIC_LATITUDE_DEG)))
    pos0, vel0 = state_from_geocentric(
        D.TLI_GEOCENTRIC_DISTANCE_KM, D.TLI_GEOCENTRIC_LATITUDE_DEG, D.TLI_RIGHT_ASCENSION_DEG,
        D.TLI_SPACE_FIXED_SPEED_KMS, D.TLI_FLIGHT_PATH_ANGLE_DEG, heading, D.GET_TLI)
    return eph, dyn, pos0, np.concatenate([pos0, vel0])


def closest_approach_to_moon(eph, sol, t_lo, t_hi):
    """(time, distance to Moon centre) of closest approach within [t_lo, t_hi]."""
    ts = np.linspace(t_lo, t_hi, 4000)
    dist = np.linalg.norm(sol.sol(ts)[:3].T - eph.moon(ts), axis=1)
    i = int(dist.argmin())
    f = lambda t: np.linalg.norm(sol.sol(t)[:3] - eph.moon(t))
    r = minimize_scalar(f, bounds=(ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]),
                        method="bounded", options={"xatol": 1e-3})
    return r.x, r.fun


# ---------------------------------------------------------------------------
# 1. CHECK (unfitted): what does the documented TLI vector do on its own?
# ---------------------------------------------------------------------------
def check_free_flight(eph, dyn, y_tli):
    free = dyn.propagate(y_tli, D.GET_TLI, D.GET_LOI1 + 3 * H)
    t_free, d_free = closest_approach_to_moon(eph, free, D.GET_LOI1 - 4 * H, D.GET_LOI1 + 3 * H)
    log("CHECK 1 - TLI vector alone, no midcourse burns, full Earth+Moon+Sun gravity:")
    log(f"  reaches the Moon at GET {t_free / H:.2f} h (real LOI: {D.GET_LOI1 / H:.2f} h), "
        f"closest approach {d_free - D.LUNAR_RADIUS_KM:,.0f} km above the surface")


# ---------------------------------------------------------------------------
# 2. Outbound: TLI -> separation burn -> MCC-2 -> pericynthion
# ---------------------------------------------------------------------------
@dataclass
class Outbound:
    tli_to_sep: object
    sep_to_mcc: object
    mcc_to_peri: object


def fit_mcc_direction(eph, dyn, y_before_mcc, t_peri, peri_r):
    """Solve the MCC-2 burn direction (2 angles) so the craft reaches the documented
    pericynthion altitude at the documented time. Magnitude is fixed."""
    def apply(angles):
        y = y_before_mcc.copy()
        y[3:] += D.MCC2_DV_KMS * direction(*angles)
        return y

    def residual(angles):
        flight = dyn.propagate(apply(angles), D.GET_MCC2, D.GET_LOI1 + 3 * H)
        t_c, d_c = closest_approach_to_moon(eph, flight, D.GET_LOI1 - 4 * H, D.GET_LOI1 + 3 * H)
        return [(t_c - t_peri) / 60.0, (d_c - peri_r) / 10.0]

    best = None
    for theta0 in np.radians([30, 90, 150]):
        for phi0 in np.radians([0, 90, 180, 270]):
            try:
                r = least_squares(residual, [theta0, phi0], diff_step=1e-3,
                                  xtol=1e-10, ftol=1e-10, max_nfev=40)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
            if best.cost < 1e-8:
                return best.x, apply(best.x)
    return best.x, apply(best.x)


def fit_outbound(eph, dyn, y_tli, t_peri):
    peri_r = D.LUNAR_RADIUS_KM + D.PERICYNTHION_ALT_KM

    tli_to_sep = dyn.propagate(y_tli, D.GET_TLI, D.GET_SEPARATION_BURN)
    y = tli_to_sep.y[:, -1].copy()
    y[3:] -= D.SEPARATION_BURN_DV_KMS * unit(y[3:])               # [ASSUMED] retrograde

    sep_to_mcc = dyn.propagate(y, D.GET_SEPARATION_BURN, D.GET_MCC2)
    y_before_mcc = sep_to_mcc.y[:, -1].copy()

    angles, y_mcc = fit_mcc_direction(eph, dyn, y_before_mcc, t_peri, peri_r)
    mcc_to_peri = dyn.propagate(y_mcc, D.GET_MCC2, t_peri)

    y_peri = mcc_to_peri.y[:, -1]
    rel_p = y_peri[:3] - eph.moon(t_peri)
    rel_v = y_peri[3:] - eph.moon_vel(t_peri)
    mcc_angle = np.degrees(np.arccos(np.clip(unit(y_mcc[3:] - y_before_mcc[3:]) @ unit(y_before_mcc[3:]), -1, 1)))

    log("\nFIT 1 - MCC-2 direction (magnitude fixed at the documented 20.9 ft/s):")
    log(f"  pericynthion altitude {np.linalg.norm(rel_p) - D.LUNAR_RADIUS_KM:.1f} km at GET "
        f"{t_peri / H:.3f} h  (fitted to documented {D.PERICYNTHION_ALT_KM:.1f} km)")
    log(f"  solved burn direction is {mcc_angle:.0f} deg from the velocity vector")

    # LOI delta-v (impulsive) -> documented 60.9 x 169.9 nmi orbit
    r_p = np.linalg.norm(rel_p)
    r_a = D.LUNAR_RADIUS_KM + D.LOI1_ORBIT_ALT_NM[1] * D.NM_TO_KM
    v_post = np.sqrt(D.MU_MOON * 2 * r_a / (r_p * (r_p + r_a)))
    log(f"CHECK 2 - implied LOI burn: {(np.linalg.norm(rel_v) - v_post) * 1000:.0f} m/s retrograde "
        f"(approach speed {np.linalg.norm(rel_v):.3f} km/s -> {v_post:.3f} km/s)")

    return Outbound(tli_to_sep, sep_to_mcc, mcc_to_peri)


# ---------------------------------------------------------------------------
# 3. Return: TEI -> entry interface (Lambert seed + full-force correction)
# ---------------------------------------------------------------------------
@dataclass
class Return:
    trajectory: object      # scipy solution TEI -> entry interface
    y_entry: np.ndarray     # state at entry interface


def fit_return(eph, dyn):
    t_tei, t_ei = D.GET_TEI, D.GET_ENTRY_INTERFACE
    moon_tei, moon_vel_tei = eph.moon(t_tei), eph.moon_vel(t_tei)
    to_j2000 = precession_matrix(t_ei).T
    dec_ei = np.radians(D.EI_GEOCENTRIC_LATITUDE_DEG)

    def ei_position(ra_deg):
        return to_j2000 @ (D.EI_GEOCENTRIC_DISTANCE_KM * spherical_unit(dec_ei, np.radians(ra_deg)))

    def lambert_fpa_error(ra_deg):
        r2 = ei_position(ra_deg)
        _, v2 = lambert(D.MU_EARTH, moon_tei, r2, t_ei - t_tei, prograde=True)   # eastward return
        fpa = np.degrees(np.arcsin(r2 @ v2 / np.linalg.norm(r2) / np.linalg.norm(v2)))
        return fpa - D.EI_FLIGHT_PATH_ANGLE_DEG

    # --- Lambert seed: entry right ascension that gives the documented flight-path angle
    ra_ei = brentq(lambert_fpa_error, 0.0, 10.0, xtol=1e-10)
    r2 = ei_position(ra_ei)
    v1, _ = lambert(D.MU_EARTH, moon_tei, r2, t_ei - t_tei, prograde=True)
    v_inf_vec = v1 - moon_vel_tei
    v_inf = np.linalg.norm(v_inf_vec)

    # --- Departure hyperbola about the Moon: pericentre radius = lunar-orbit radius
    rc = D.LUNAR_ORBIT_RADIUS_KM
    ecc = 1 + rc * v_inf**2 / D.MU_MOON
    nu_inf = np.arccos(-1 / ecc)
    u_inf = unit(v_inf_vec)
    a1 = unit(np.cross(u_inf, [0, 0, 1.0]))
    a2 = np.cross(u_inf, a1)
    anti_earth = -unit(moon_tei)
    phi = np.arctan2(anti_earth @ a2, anti_earth @ a1)
    # [ASSUMED] burn point: the one of the allowed points nearest the lunar far side
    p_hat = np.cos(nu_inf) * u_inf + np.sin(nu_inf) * (np.cos(phi) * a1 + np.sin(phi) * a2)
    h_hat = unit(np.cross(p_hat, u_inf))
    v_hat = np.cross(h_hat, p_hat)
    pos_tei = moon_tei + rc * p_hat
    v_seed = np.sqrt(v_inf**2 + 2 * D.MU_MOON / rc) * v_hat

    # --- Full-force correction of the departure velocity
    def state_at_ei(v_rel):
        start = np.concatenate([pos_tei, moon_vel_tei + v_rel])
        return solve_ivp(dyn.rhs, [t_tei, t_ei], start, method="DOP853",
                         rtol=1e-10, atol=1e-8).y[:, -1]

    def ei_residual(v_rel):
        y = state_at_ei(v_rel)
        dec, fpa = declination_and_fpa(y[:3], y[3:], t_ei)
        return np.array([(np.linalg.norm(y[:3]) - D.EI_GEOCENTRIC_DISTANCE_KM) / 10.0,
                         (dec - D.EI_GEOCENTRIC_LATITUDE_DEG) * 5.0,
                         (fpa - D.EI_FLIGHT_PATH_ANGLE_DEG) * 5.0])

    fit = least_squares(ei_residual, v_seed, diff_step=1e-7, xtol=1e-13, ftol=1e-13, max_nfev=40)
    v_rel_tei = fit.x
    y_tei = np.concatenate([pos_tei, moon_vel_tei + v_rel_tei])
    trajectory = dyn.propagate(y_tei, t_tei, t_ei)
    y_ei = trajectory.y[:, -1]

    # --- Report
    v_hor = v_rel_tei - (v_rel_tei @ p_hat) * p_hat
    tei_dv = np.linalg.norm(v_rel_tei - np.sqrt(D.MU_MOON / rc) * unit(v_hor))
    t_obj = get_time(t_ei)
    gmst = np.degrees(erfa.gmst06(t_obj.ut1.jd1, t_obj.ut1.jd2, t_obj.tt.jd1, t_obj.tt.jd2))
    lon_ei = (ra_ei - gmst + 180.0) % 360.0 - 180.0
    dec_chk, fpa_chk = declination_and_fpa(y_ei[:3], y_ei[3:], t_ei)

    log("\nFIT 2 - return trajectory departure velocity (fitted to documented entry interface):")
    log(f"  geocentric distance {np.linalg.norm(y_ei[:3]):.2f} km, latitude {dec_chk:.3f} deg, "
        f"flight-path angle {fpa_chk:.3f} deg at GET {t_ei / H:.3f} h")
    log(f"CHECK 3 - entry speed {np.linalg.norm(y_ei[3:]):.4f} km/s (documented {D.EI_SPEED_KMS:.4f} km/s)")
    log(f"CHECK 4 - implied TEI burn {tei_dv * 1000:.0f} m/s (about 1 km/s expected)")
    log(f"CHECK 5 - implied entry longitude {lon_ei:.1f} deg E; documented splashdown "
        f"13.3 N, 169.2 W is ~1,500 nmi downrange")

    return Return(trajectory, y_ei)


# ---------------------------------------------------------------------------
# 4. Sample the full trajectory into labelled legs
# ---------------------------------------------------------------------------
def sample_legs(eph, outbound, ret, t_peri):
    """List of (label, times, positions). Lunar stay is idealised: the craft stays within
    ~1,900 km of the Moon's centre, invisible at Earth scale, so it is drawn at the Moon."""
    def sample(sol, t0, t1, dt):
        ts = np.arange(t0, t1, dt)
        return ts, sol.sol(ts)[:3].T

    legs = []
    for label, sol, t0, t1, dt in (
            ("tli_to_separation", outbound.tli_to_sep, D.GET_TLI, D.GET_SEPARATION_BURN, 5.0),
            ("separation_to_mcc2", outbound.sep_to_mcc, D.GET_SEPARATION_BURN, D.GET_MCC2, 60.0),
            ("mcc2_to_pericynthion", outbound.mcc_to_peri, D.GET_MCC2, t_peri, 60.0)):
        legs.append((label, *sample(sol, t0, t1, dt)))

    ts_stay = np.arange(t_peri, D.GET_TEI, 300.0)
    legs.append(("lunar_stay_idealised", ts_stay, np.array([eph.moon(t) for t in ts_stay])))

    legs.append(("transearth", *sample(ret.trajectory, D.GET_TEI, D.GET_ENTRY_INTERFACE, 5.0)))
    return legs


# ---------------------------------------------------------------------------
# 5. Van Allen belt dwell time (v1: spherical shells)
# ---------------------------------------------------------------------------
def dwell_seconds(t, dist_re, band):
    """Seconds spent with the distance from the Earth's centre (in Earth radii) inside
    band = (lower, upper); lower limit included, upper limit excluded."""
    inside = (dist_re >= band[0]) & (dist_re < band[1])
    dt = np.diff(t, append=t[-1])
    return float(dt[inside].sum())


def analyse_belts(all_t, dist_re):
    out_mask = all_t < D.GET_TLI + BELT_WINDOW_S
    ret_mask = all_t > D.GET_ENTRY_INTERFACE - BELT_WINDOW_S
    log("\nVAN ALLEN BELTS (v1 spherical shells, limits after Ripoll et al. 2020):")
    for name, mask in (("outbound (first 12 h)", out_mask), ("return (last 12 h)", ret_mask)):
        inner = dwell_seconds(all_t[mask], dist_re[mask], INNER_BELT_RE) / 60
        outer = dwell_seconds(all_t[mask], dist_re[mask], OUTER_BELT_RE) / 60
        log(f"  {name:22s} inner belt (1-3 R_E) {inner:6.1f} min | outer belt (3-7 R_E) {outer:6.1f} min")
    return out_mask, ret_mask


# ---------------------------------------------------------------------------
# 6. Figures (three separate files)
# ---------------------------------------------------------------------------
def make_projector(eph):
    """Projection onto the Moon's orbital plane (x towards the Moon at LOI)."""
    normal = unit(np.cross(eph.moon(D.GET_LOI1), eph.moon_vel(D.GET_LOI1)))
    e1 = unit(eph.moon(D.GET_LOI1))
    e2 = np.cross(normal, e1)
    return lambda P: np.column_stack([np.atleast_2d(P) @ e1, np.atleast_2d(P) @ e2])


def plot_top_view(eph, legs, pos0, outbound, y_ei, t_peri, path):
    proj = make_projector(eph)
    fig, ax = plt.subplots(figsize=(11, 9))

    # Moon's orbit
    mt = np.arange(D.GET_TLI, D.GET_ENTRY_INTERFACE, 600.0)
    mp = proj(np.array([eph.moon(t) for t in mt]))
    ax.plot(mp[:, 0], mp[:, 1], "--", color="0.65", lw=1, label="Mondbahn")

    # Spacecraft legs: (colour, legend entry)
    style = {"tli_to_separation": ("tab:blue", "Hinflug"),
             "separation_to_mcc2": ("tab:blue", "Hinflug"),
             "mcc2_to_pericynthion": ("tab:blue", "Hinflug"),
             "lunar_stay_idealised": ("0.4", "Mondaufenthalt (idealisiert)"),
             "transearth": ("tab:orange", "Rückflug")}
    legend_done = set()
    for label, _, positions in legs:
        color, name = style[label]
        q = proj(positions)
        ax.plot(q[:, 0], q[:, 1], color=color, lw=1.8,
                label=None if name in legend_done else name)
        legend_done.add(name)

    # Earth
    ax.add_patch(plt.Circle((0, 0), D.R_EARTH, color="tab:blue", alpha=0.5, zorder=3,
                            label="Erde (maßstabsgetreu)"))

    # Mission events: (name, marker, time, position, label offset in points).
    # TLI and entry interface both sit right next to the Earth, so their labels are
    # pushed apart (TLI up, entry down) and tied to their points with a thin arrow.
    events = [("TLI", "o", D.GET_TLI, pos0, (0, 60)),
              ("MCC-2", "s", D.GET_MCC2, outbound.sep_to_mcc.sol(D.GET_MCC2)[:3], (8, 6)),
              ("LOI", "D", t_peri, eph.moon(t_peri), (8, 6)),
              ("TEI", "^", D.GET_TEI, eph.moon(D.GET_TEI), (8, 6)),
              ("Atmosphäreneintritt", "v", D.GET_ENTRY_INTERFACE, y_ei[:3], (0, -60))]
    for name, marker, t, position, offset in events:
        pt = proj(position)[0]
        ax.plot(*pt, marker, color="k", ms=6, zorder=5)
        near_earth = np.linalg.norm(position) < 2 * D.R_EARTH
        ax.annotate(f"{name}\nGET {de(t / H)} h", pt, textcoords="offset points",
                    xytext=offset, fontsize=8, ha="center" if near_earth else "left",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                    arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8) if near_earth else None)

    # Moon at LOI and TEI (radius exaggerated 3.5x, only the first circle gets a legend entry)
    for i, t in enumerate((D.GET_LOI1, D.GET_TEI)):
        ax.add_patch(plt.Circle(proj(eph.moon(t))[0], 6000, fc="0.85", ec="0.4", zorder=2,
                                label="Mond bei LOI und TEI (Radius 3,5-fach überhöht)" if i == 0 else None))

    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(6))      # weniger Zahlen, damit sie sich nicht überlappen
    ax.grid(alpha=0.25)
    ax.set_xlabel("x [km]  (Richtung vom Erdmittelpunkt zum Mond bei der LOI)")
    ax.set_ylabel("y [km]  (in der Mondbahnebene, senkrecht zu x)")
    ax.set_title("Apollo 11, Draufsicht (Blick senkrecht auf die Mondbahnebene)")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(SUPTITLE, fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_belt_panel(all_t, dist_re, mask, t_ref, title, xlabel, path,
                    xlim=None, legend_loc="best"):
    """Distance from the Earth's centre (Earth radii) vs. time with the Van Allen belt bands."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhspan(*INNER_BELT_RE, color="tab:red", alpha=0.25, label="innerer Gürtel (1–3 Erdradien)")
    ax.axhspan(*OUTER_BELT_RE, color="tab:green", alpha=0.18, label="äußerer Gürtel (3–7 Erdradien)")
    ax.plot((all_t[mask] - t_ref) / H, dist_re[mask], color="k", lw=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Abstand vom Erdmittelpunkt [Erdradien]")
    ax.set_ylim(0, 12)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc=legend_loc)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Export (stable intermediate format for a future 3D / interactive viewer)
# ---------------------------------------------------------------------------
def export_json(legs, path):
    export = {
        "meta": {"frame": "Earth-centred inertial, J2000 axes",
                 "units": {"t": "GET s", "pos": "km"},
                 "range_zero_utc": D.RANGE_ZERO_UTC, "ephemeris": "astropy builtin",
                 "notes": ["lunar stay idealised at the Moon's position",
                           "MCC-2 direction and return departure velocity are fitted"]},
        "legs": [{"label": l, "t": ts.tolist(), "pos_km": ps.tolist()} for l, ts, ps in legs],
    }
    with open(path, "w") as f:
        json.dump(export, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    eph, dyn, pos0, y_tli = setup_simulation()
    t_peri = D.GET_LOI1 + D.LOI1_BURN_DURATION_S / 2.0       # pericynthion = mid-burn

    check_free_flight(eph, dyn, y_tli)
    outbound = fit_outbound(eph, dyn, y_tli, t_peri)
    ret = fit_return(eph, dyn)

    legs = sample_legs(eph, outbound, ret, t_peri)
    all_t = np.concatenate([leg[1] for leg in legs])
    all_p = np.vstack([leg[2] for leg in legs])
    dist_re = np.linalg.norm(all_p, axis=1) / R_E_BELT_KM        # distance from Earth's centre in R_E
    out_mask, ret_mask = analyse_belts(all_t, dist_re)

    plot_top_view(eph, legs, pos0, outbound, ret.y_entry, t_peri, FIG_TOP_VIEW)
    plot_belt_panel(all_t, dist_re, out_mask, D.GET_TLI,
                    "Abflug: erste 12 Stunden nach der TLI", "Zeit nach der TLI [h]",
                    FIG_OUTBOUND, legend_loc="upper left")
    plot_belt_panel(all_t, dist_re, ret_mask, D.GET_ENTRY_INTERFACE,
                    "Rückflug: letzte 12 Stunden vor dem Atmosphäreneintritt",
                    "Zeit relativ zum Atmosphäreneintritt [h]  (negativ = davor)",
                    FIG_RETURN, xlim=(-12, 0), legend_loc="upper right")
    export_json(legs, EXPORT_JSON)

    log(f"\nWrote {FIG_TOP_VIEW}, {FIG_OUTBOUND}, {FIG_RETURN} and {EXPORT_JSON}")


if __name__ == "__main__":
    main()
