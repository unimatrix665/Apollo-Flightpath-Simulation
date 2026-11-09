"""
Documented Apollo 11 mission data, with the provenance of every number.

Anything marked [DOC] is taken from a published mission document.
Anything marked [DERIVED] is computed from [DOC] values.
Anything marked [ASSUMED] is NOT documented in the sources used here and is a
modelling choice -- those are the places where this reconstruction is
approximate, and where you should look for better data first.

All times are Ground Elapsed Time (GET) in seconds since "range zero".
"""

# --- Time reference ---------------------------------------------------------
# [DOC] Apollo 11 Mission Report (MSC-00171): range zero = 13:32:00 GMT,
# 16 July 1969. GET is measured from here.
RANGE_ZERO_UTC = "1969-07-16 13:32:00"


def hms(h, m, s=0.0):
    return h * 3600.0 + m * 60.0 + s


# --- Event times [DOC: Mission Report, Table 3-I "Sequence of Events"] ------
GET_TLI_IGNITION = hms(2, 44, 16.2)
GET_SEPARATION_BURN = hms(4, 40, 1.8)      # SPS burn after S-IVB ejection
GET_MCC2 = hms(26, 44, 58.7)               # the mission's first (only) TLC correction
GET_LOI1 = hms(75, 49, 50.4)               # ignition time
GET_LOI2 = hms(80, 11, 36.8)
GET_TEI = hms(135, 23, 42.3)               # ignition time
GET_ENTRY_INTERFACE = hms(195, 3, 5.7)

# --- Translunar injection state ----------------------------------------------
# [DOC] Boeing "Apollo 11 Postflight Trajectory" state-vector table, as
# transcribed and worked through by Grant Hutchison (oikofuge.com, "Finding
# Apollo Trajectory Data" and "Converting Apollo State Vectors To Orbits").
# TLI is defined as 10 s after S-IVB cutoff.
GET_TLI = 10213.030                        # s
TLI_GEOCENTRIC_DISTANCE_KM = 6711.964
TLI_GEOCENTRIC_LATITUDE_DEG = 9.9204       # = declination
TLI_RIGHT_ASCENSION_DEG = 15.0446          # mean equator & equinox of date
TLI_SPACE_FIXED_SPEED_KMS = 10.8343
TLI_FLIGHT_PATH_ANGLE_DEG = 7.367          # above local horizontal
# Inclination of the departure orbit; the Postflight Trajectory summary table
# and the state-vector calculation agree on this value.
TLI_INCLINATION_DEG = 31.383
# [DERIVED] heading = asin(cos(i)/cos(dec)), eastward (<90 deg) branch.

# --- Small documented burns ---------------------------------------------------
# [DOC] Mission Report Table 7-IV (maneuver summary) and Orloff, "Apollo by the
#       Numbers" (Translunar Phase table):
#         separation maneuver  04:40:01.72, burn time 2.93 s, dV 19.7 ft/s
#         first midcourse corr. 26:44:58.64, burn time 3.13 s, dV 20.9 ft/s
#       (Mission Report sec. 4.5 only says "~20 ft/s" for the separation burn;
#       the 19.7 ft/s of the table is the precise value. Sec. 3.0 confirms the
#       20.9 ft/sec of the midcourse correction.)
FT_TO_KM = 0.3048e-3
SEPARATION_BURN_DV_KMS = 19.7 * FT_TO_KM       # ~6.0 m/s
MCC2_DV_KMS = 20.9 * FT_TO_KM                  # ~6.4 m/s
# [ASSUMED] The reports used here give the MAGNITUDE of these burns but not
# their direction. The separation burn is modelled as retrograde (it was
# described as cancelling a targeted overspeed); the MCC-2 direction is
# SOLVED so the spacecraft reaches the documented pericynthion.

# --- Lunar orbit [DOC: Mission Report sec. 3.0 / 4.6] ------------------------
NM_TO_KM = 1.852
LUNAR_RADIUS_KM = 1737.4        # IAU value; NASA Moon Fact Sheet versions give 1737.1 (mean) / 1738.1 (equatorial)
LOI1_ORBIT_ALT_NM = (60.9, 169.9)              # pericynthion x apocynthion after LOI
LOI2_ORBIT_ALT_NM = (54.4, 66.1)               # after circularisation
PERICYNTHION_ALT_KM = LOI1_ORBIT_ALT_NM[0] * NM_TO_KM
# [DOC] Mission Report, service propulsion maneuver summary: LOI ignition
#       75:49:50.37, cutoff 75:55:47.90 -> burn time 357.53 s (sec. 4.6 calls it
#       "a 6-minute service propulsion maneuver", which is only the rounded value).
# [DERIVED] Burn straddles pericynthion: pericynthion = ignition + half the duration.
LOI1_BURN_DURATION_S = 357.53
# [ASSUMED] Lunar stay is idealised as a circular orbit at the mean LOI-2
# altitude, in the plane of the approach. It matters only for the Moon-centred
# geometry at TEI; at Earth scale it is invisible.
LUNAR_ORBIT_RADIUS_KM = LUNAR_RADIUS_KM + 0.5 * sum(LOI2_ORBIT_ALT_NM) * NM_TO_KM

# --- Entry interface [DOC] ----------------------------------------------------
# [DOC] Mission Report Table 7-VII "Entry trajectory parameters", entry interface
# (400,000 ft): time 195:03:05.7, geodetic latitude 3.19 deg S, longitude
# 171.96 deg E, altitude 65.8 nmi, space-fixed velocity 36,194.4 ft/s, space-fixed
# flight-path angle -6.48 deg, heading 50.18 deg east of north.
# Other values for the flight-path angle (Mission Report sec. 8.6.7): -6.483 deg
# from tracking, -6.488 deg from the onboard computer (= the "-6.49" of the Entry PAD).
# [DERIVED] Geocentric distance/latitude from the geodetic table values, after
# Hutchison (oikofuge.com): r = 6500.02 km, geocentric latitude = -3.17 deg.
EI_GEOCENTRIC_DISTANCE_KM = 6500.02
EI_GEOCENTRIC_LATITUDE_DEG = -3.17
EI_FLIGHT_PATH_ANGLE_DEG = -6.48
# Not used as input, but available as an independent check of the fit:
EI_LONGITUDE_DEG_E = 171.96                    # Table 7-VII
EI_HEADING_DEG = 50.18                         # Table 7-VII, east of north
EI_SPEED_FTS = 36194.0                         # used only as an independent check
EI_SPEED_KMS = EI_SPEED_FTS * FT_TO_KM

# --- Physical constants -------------------------------------------------------
# [DOC] JPL DE440 (Park et al. 2021), as listed at ssd.jpl.nasa.gov/astro_par.html.
# All three GM values now come from the same set (the previous MU_MOON/MU_SUN
# belonged to older ephemeris sets).
MU_EARTH = 398600.435507     # km^3/s^2
MU_MOON = 4902.800118
MU_SUN = 1.32712440041279419e11
R_EARTH = 6378.137
J2_EARTH = 1.08262668e-3
