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

Run:  python main.py        ->  prints a validation report, writes
                                apollo11_trajectory.png and apollo11_trajectory.json
"""
import json
import numpy as np
import erfa
import matplotlib
matplotlib.use("Agg")           # remove this line to get an interactive window
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar, brentq
from scipy.integrate import solve_ivp

import apollo11_data as D
from funcs_classes import (Ephemeris, Dynamics, state_from_geocentric, precession_matrix,
                           declination_and_fpa, get_time, lambert)

H = 3600.0
unit = lambda v: v / np.linalg.norm(v)

# 0. Setup

eph = Ephemeris(D.GET_TLI, D.GET_ENTRY_INTERFACE + H)
dyn = Dynamics(eph)

heading = np.arcsin(np.cos(np.radians(D.TLI_INCLINATION_DEG)) /
                    np.cos(np.radians(D.TLI_GEOCENTRIC_LATITUDE_DEG)))
pos0, vel0 = state_from_geocentric(
    D.TLI_GEOCENTRIC_DISTANCE_KM, D.TLI_GEOCENTRIC_LATITUDE_DEG, D.TLI_RIGHT_ASCENSION_DEG,
    D.TLI_SPACE_FIXED_SPEED_KMS, D.TLI_FLIGHT_PATH_ANGLE_DEG, heading, D.GET_TLI)
y_tli = np.concatenate([pos0, vel0])

T_PERI = D.GET_LOI1 + D.LOI1_BURN_DURATION_S / 2.0     # pericynthion = mid-burn
PERI_R = D.LUNAR_RADIUS_KM + D.PERICYNTHION_ALT_KM


def closest_approach_to_moon(sol, t_lo, t_hi):
    ts = np.linspace(t_lo, t_hi, 4000)
    d = np.linalg.norm(sol.sol(ts)[:3].T - eph.moon(ts), axis=1)
    i = int(d.argmin())
    f = lambda t: np.linalg.norm(sol.sol(t)[:3] - eph.moon(t))
    r = minimize_scalar(f, bounds=(ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]),
                        method="bounded", options={"xatol": 1e-3})
    return r.x, r.fun


report = []
def log(s=""):
    print(s); report.append(s)



# 1. CHECK (unfitted): what does the documented TLI vector do on its own?

free = dyn.propagate(y_tli, D.GET_TLI, D.GET_LOI1 + 3 * H)
t_free, d_free = closest_approach_to_moon(free, D.GET_LOI1 - 4 * H, D.GET_LOI1 + 3 * H)
log("CHECK 1 - TLI vector alone, no midcourse burns, full Earth+Moon+Sun gravity:")
log(f"  reaches the Moon at GET {t_free/H:.2f} h (real LOI: {D.GET_LOI1/H:.2f} h), "
    f"closest approach {d_free - D.LUNAR_RADIUS_KM:,.0f} km above the surface")


# 2. Outbound: TLI -> separation burn -> MCC-2 -> pericynthion

seg_tli_sep = dyn.propagate(y_tli, D.GET_TLI, D.GET_SEPARATION_BURN)
y = seg_tli_sep.y[:, -1].copy()
y[3:] -= D.SEPARATION_BURN_DV_KMS * unit(y[3:])           # [ASSUMED] retrograde
y_after_sep = y.copy()

seg_sep_mcc = dyn.propagate(y_after_sep, D.GET_SEPARATION_BURN, D.GET_MCC2)
y_before_mcc = seg_sep_mcc.y[:, -1].copy()


def apply_mcc(angles):
    th, ph = angles
    d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    y = y_before_mcc.copy()
    y[3:] += D.MCC2_DV_KMS * d
    return y, d


def mcc_flight(angles):
    y, _ = apply_mcc(angles)
    return dyn.propagate(y, D.GET_MCC2, D.GET_LOI1 + 3 * H)


def mcc_resid(angles):
    s = mcc_flight(angles)
    tc, dc = closest_approach_to_moon(s, D.GET_LOI1 - 4 * H, D.GET_LOI1 + 3 * H)
    return [(tc - T_PERI) / 60.0, (dc - PERI_R) / 10.0]


best = None
for th0 in np.radians([30, 90, 150]):
    for ph0 in np.radians([0, 90, 180, 270]):
        try:
            r = least_squares(mcc_resid, [th0, ph0], diff_step=1e-3, xtol=1e-10, ftol=1e-10, max_nfev=40)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
        if best.cost < 1e-8:
            break
    if best is not None and best.cost < 1e-8:
        break

y_mcc, mcc_dir = apply_mcc(best.x)
seg_mcc_peri = dyn.propagate(y_mcc, D.GET_MCC2, T_PERI)
y_peri = seg_mcc_peri.y[:, -1]
rel_p = y_peri[:3] - eph.moon(T_PERI)
rel_v = y_peri[3:] - eph.moon_vel(T_PERI)
angle_mcc_vs_vel = np.degrees(np.arccos(np.clip(mcc_dir @ unit(y_before_mcc[3:]), -1, 1)))

log("\nFIT 1 - MCC-2 direction (magnitude fixed at the documented 20.9 ft/s):")
log(f"  pericynthion altitude {np.linalg.norm(rel_p) - D.LUNAR_RADIUS_KM:.1f} km at GET "
    f"{T_PERI/H:.3f} h  (fitted to documented {D.PERICYNTHION_ALT_KM:.1f} km)")
log(f"  solved burn direction is {angle_mcc_vs_vel:.0f} deg from the velocity vector")

# LOI delta-v (impulsive) -> documented 60.9 x 169.9 nmi orbit
r_p = np.linalg.norm(rel_p)
r_a = D.LUNAR_RADIUS_KM + D.LOI1_ORBIT_ALT_NM[1] * D.NM_TO_KM
v_post = np.sqrt(D.MU_MOON * 2 * r_a / (r_p * (r_p + r_a)))
loi_dv = np.linalg.norm(rel_v) - v_post
log(f"CHECK 2 - implied LOI burn: {loi_dv*1000:.0f} m/s retrograde (approach speed "
    f"{np.linalg.norm(rel_v):.3f} km/s -> {v_post:.3f} km/s)")


# 3. Return: TEI -> entry interface   (Lambert seed + full-force correction)

t_tei, t_ei = D.GET_TEI, D.GET_ENTRY_INTERFACE
moon_tei, moon_vel_tei = eph.moon(t_tei), eph.moon_vel(t_tei)
to_j2000 = precession_matrix(t_ei).T
dec_ei = np.radians(D.EI_GEOCENTRIC_LATITUDE_DEG)


def ei_position(ra_deg):
    ra = np.radians(ra_deg)
    return to_j2000 @ (D.EI_GEOCENTRIC_DISTANCE_KM *
                       np.array([np.cos(dec_ei) * np.cos(ra), np.cos(dec_ei) * np.sin(ra), np.sin(dec_ei)]))


def lambert_fpa_error(ra_deg):
    r2 = ei_position(ra_deg)
    _, v2 = lambert(D.MU_EARTH, moon_tei, r2, t_ei - t_tei, prograde=True)   # eastward return
    return (np.degrees(np.arcsin(r2 @ v2 / np.linalg.norm(r2) / np.linalg.norm(v2)))
            - D.EI_FLIGHT_PATH_ANGLE_DEG)


ra_ei = brentq(lambert_fpa_error, 0.0, 10.0, xtol=1e-10)
r2 = ei_position(ra_ei)
v1, v2 = lambert(D.MU_EARTH, moon_tei, r2, t_ei - t_tei, prograde=True)
v_inf_vec = v1 - moon_vel_tei
v_inf = np.linalg.norm(v_inf_vec)

# Departure hyperbola about the Moon: pericentre radius = lunar-orbit radius.
rc = D.LUNAR_ORBIT_RADIUS_KM
ecc = 1 + rc * v_inf**2 / D.MU_MOON
nu_inf = np.arccos(-1 / ecc)
u_inf = unit(v_inf_vec)
a1 = unit(np.cross(u_inf, [0, 0, 1.0]))
a2 = np.cross(u_inf, a1)
anti_earth = -unit(moon_tei)
phi = np.arctan2(anti_earth @ a2, anti_earth @ a1)      # [ASSUMED] burn point: the one of the
p_hat = np.cos(nu_inf) * u_inf + np.sin(nu_inf) * (np.cos(phi) * a1 + np.sin(phi) * a2)  # allowed points
h_hat = unit(np.cross(p_hat, u_inf))                    # nearest the lunar far side
v_hat = np.cross(h_hat, p_hat)
pos_tei = moon_tei + rc * p_hat
v_seed = np.sqrt(v_inf**2 + 2 * D.MU_MOON / rc) * v_hat


def state_at_ei(v_rel, rtol=1e-10):
    st = np.concatenate([pos_tei, moon_vel_tei + v_rel])
    return solve_ivp(dyn.rhs, [t_tei, t_ei], st, method="DOP853", rtol=rtol, atol=1e-8).y[:, -1]


def ei_resid(v_rel):
    y = state_at_ei(v_rel)
    dec, fpa = declination_and_fpa(y[:3], y[3:], t_ei)
    return np.array([(np.linalg.norm(y[:3]) - D.EI_GEOCENTRIC_DISTANCE_KM) / 10.0,
                     (dec - D.EI_GEOCENTRIC_LATITUDE_DEG) * 5.0,
                     (fpa - D.EI_FLIGHT_PATH_ANGLE_DEG) * 5.0])


fit = least_squares(ei_resid, v_seed, diff_step=1e-7, xtol=1e-13, ftol=1e-13, max_nfev=40)
v_rel_tei = fit.x
y_tei = np.concatenate([pos_tei, moon_vel_tei + v_rel_tei])
seg_return = dyn.propagate(y_tei, t_tei, t_ei)
y_ei = seg_return.y[:, -1]

v_hor = v_rel_tei - (v_rel_tei @ p_hat) * p_hat
tei_dv = np.linalg.norm(v_rel_tei - np.sqrt(D.MU_MOON / rc) * unit(v_hor))
t_obj = get_time(t_ei)
gmst = np.degrees(erfa.gmst06(t_obj.ut1.jd1, t_obj.ut1.jd2, t_obj.tt.jd1, t_obj.tt.jd2))
lon_ei = (ra_ei - gmst + 180.0) % 360.0 - 180.0
dec_chk, fpa_chk = declination_and_fpa(y_ei[:3], y_ei[3:], t_ei)

log("\nFIT 2 - return trajectory departure velocity (fitted to documented entry interface):")
log(f"  geocentric distance {np.linalg.norm(y_ei[:3]):.2f} km, latitude {dec_chk:.3f} deg, "
    f"flight-path angle {fpa_chk:.3f} deg at GET {t_ei/H:.3f} h")
log(f"CHECK 3 - entry speed {np.linalg.norm(y_ei[3:]):.4f} km/s (documented {D.EI_SPEED_KMS:.4f} km/s)")
log(f"CHECK 4 - implied TEI burn {tei_dv*1000:.0f} m/s (about 1 km/s expected)")
log(f"CHECK 5 - implied entry longitude {lon_ei:.1f} deg E; documented splashdown "
    f"13.3 N, 169.2 W is ~1,500 nmi downrange")


# 4. Sample the full trajectory

def sample(sol, t0, t1, dt):
    ts = np.arange(t0, t1, dt)
    return ts, sol.sol(ts)[:3].T


def cadence(t0, t1, fine_until=None, fine_from=None):
    """5 s resolution in the belt-crossing windows, 60 s elsewhere."""
    return 5.0


legs = []     # (label, t array, pos array)
for label, sol, a, b in (("tli_to_separation", seg_tli_sep, D.GET_TLI, D.GET_SEPARATION_BURN),
                         ("separation_to_mcc2", seg_sep_mcc, D.GET_SEPARATION_BURN, D.GET_MCC2),
                         ("mcc2_to_pericynthion", seg_mcc_peri, D.GET_MCC2, T_PERI),
                         ("transearth", seg_return, t_tei, t_ei)):
    dt = 5.0 if label in ("tli_to_separation", "transearth") else 60.0
    ts, ps = sample(sol, a, b, dt)
    legs.append((label, ts, ps))

# lunar stay: idealised -- the spacecraft is within ~1,900 km of the Moon's centre,
# which is invisible at Earth scale, so it is drawn at the Moon's position.
ts_stay = np.arange(T_PERI, t_tei, 300.0)
legs.insert(3, ("lunar_stay_idealised", ts_stay, np.array([eph.moon(t) for t in ts_stay])))

all_t = np.concatenate([l[1] for l in legs])
all_p = np.vstack([l[2] for l in legs])
alt = np.linalg.norm(all_p, axis=1) - D.R_EARTH


# Van Allen belts -- v1 model: spherical altitude shells (no geomagnetic tilt)

INNER = (1000.0, 6000.0)        # km altitude  [approximate]
OUTER = (13000.0, 60000.0)


def dwell(t, a, band):
    inside = (a >= band[0]) & (a <= band[1])
    dt = np.diff(t, append=t[-1])
    return float(dt[inside].sum())


out_mask = all_t < D.GET_TLI + 12 * H
ret_mask = all_t > t_ei - 12 * H
log("\nVAN ALLEN BELTS (v1 spherical approximation):")
for name, m in (("outbound (first 12 h)", out_mask), ("return (last 12 h)", ret_mask)):
    log(f"  {name:22s} inner shell {dwell(all_t[m], alt[m], INNER)/60:6.1f} min | "
        f"outer shell {dwell(all_t[m], alt[m], OUTER)/60:6.1f} min")
#log("  (published estimate using geomagnetic coordinates: ~14 min inside the proton-belt limits, "
#    "outbound -- Hutchison)")


# 5. Figure

moon_orbit_normal = unit(np.cross(eph.moon(D.GET_LOI1), eph.moon_vel(D.GET_LOI1)))
e1 = unit(eph.moon(D.GET_LOI1))
e2 = np.cross(moon_orbit_normal, e1)
proj = lambda P: np.column_stack([np.atleast_2d(P) @ e1, np.atleast_2d(P) @ e2])

fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.32, wspace=0.18)
axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 1])

# --- A: top-down, looking down the Moon's orbital-plane normal -------------
mt = np.arange(D.GET_TLI, t_ei, 600.0)
mp = proj(np.array([eph.moon(t) for t in mt]))
axA.plot(mp[:, 0], mp[:, 1], "--", color="0.65", lw=1, label="Mond orbit")
colors = {"tli_to_separation": "tab:blue", "separation_to_mcc2": "tab:blue",
          "mcc2_to_pericynthion": "tab:blue", "lunar_stay_idealised": "0.4", "transearth": "tab:orange"}
done = set()
for label, ts, ps in legs:
    q = proj(ps)
    key = "Hinflug" if colors[label] == "tab:blue" else ("Rückflug" if label == "transearth" else "Mondaufenthalt")
    axA.plot(q[:, 0], q[:, 1], color=colors[label], lw=1.8, label=None if key in done else key)
    done.add(key)
axA.add_patch(plt.Circle((0, 0), D.R_EARTH, color="tab:blue", alpha=0.5, zorder=3))
for t, name, mk in ((D.GET_TLI, "", ""), (D.GET_MCC2, "MCC-2", "s"), (T_PERI, "LOI", "D"),
                    (t_tei, "TEI", "^"), (t_ei, "Wiedereintritt", "v")):
    pt = proj(eph.moon(t) if name in ("LOI", "TEI") else
              (pos0 if name == "TLI" else
               (seg_sep_mcc.sol(t)[:3] if name == "MCC-2" else y_ei[:3])))[0]
    axA.plot(*pt, mk, color="k", ms=6, zorder=5)
    axA.annotate(f"{name}\nGET {t/H:.1f} h", pt, textcoords="offset points", xytext=(8, 6), fontsize=8)
for t, name in ((D.GET_LOI1, "Moon at LOI"), (t_tei, "Moon at TEI")):
    c = proj(eph.moon(t))[0]
    axA.add_patch(plt.Circle(c, 6000, fc="0.85", ec="0.4", zorder=2))   # Moon radius exaggerated 3.5x
axA.set_aspect("equal"); axA.grid(alpha=0.25)
axA.set_xlabel("km  (Abstand zum Mond)"); axA.set_ylabel("km  (in the Moon's orbital plane)")
axA.set_title("Apollo 11, Draufsicht (Blick senkrecht auf die Mondbahnebene)")
axA.legend(loc="upper left", fontsize=8)

# --- B/C: altitude vs time with belt bands ----------------------------------
def belt_panel(ax, mask, t_ref, title, xlabel):
    ax.axhspan(*INNER, color="tab:red", alpha=0.25, label="innerer Gürtel (ungefähr)")
    ax.axhspan(*OUTER, color="tab:green", alpha=0.18, label="äußerer Gürtel (ungefähr)")
    ax.plot((all_t[mask] - t_ref) / H, alt[mask], color="k", lw=1.5)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Höhe [km]"); ax.grid(alpha=0.25)
    ax.set_ylim(0, 75000)

belt_panel(axB, out_mask, D.GET_TLI, "Abflug: erste 12 Stunden nach der TLI", "Stunden nach der TLI")
axB.legend(fontsize=8, loc="lower right")
belt_panel(axC, ret_mask, t_ei, "Rückflug: letzte 12 Stunden vor dem Athmosspäreneintritt", "Stunden bis zum Athmosspäreneintritt")
axC.set_xlim(-12, 0)
fig.suptitle("Apollo-11-Flugbahnrekonstruktion anhand dokumentierter Zustandsvektoren (Erde–Mond, ohne Sonne, ideale Zweikörperbahnen, "
             "Mondaufenthalt idealisiert)", fontsize=11)
fig.savefig("apollo11_trajectory.png", dpi=140, bbox_inches="tight")


# 6. Export (stable intermediate format for a future 3D / interactive viewer)

export = {
    "meta": {"frame": "Earth-centred inertial, J2000 axes", "units": {"t": "GET s", "pos": "km"},
             "range_zero_utc": D.RANGE_ZERO_UTC, "ephemeris": "astropy builtin",
             "notes": ["lunar stay idealised at the Moon's position",
                       "MCC-2 direction and return departure velocity are fitted"]},
    "legs": [{"label": l, "t": ts.tolist(), "pos_km": ps.tolist()} for l, ts, ps in legs],
}
with open("apollo11_trajectory.json", "w") as f:
    json.dump(export, f)
log("\nWrote apollo11_trajectory.png and apollo11_trajectory.json")
