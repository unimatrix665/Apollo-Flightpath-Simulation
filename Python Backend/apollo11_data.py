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
# [DOC] Mission Report sec. 4.5: ~20 ft/s SPS burn 23 min after spacecraft ejection
#       (the 04:40 separation maneuver).
# [DOC] Mission Report sec. 3.0: MCC at ~26.5 h "provided a 20.9 ft/sec velocity change".
FT_TO_KM = 0.3048e-3
SEPARATION_BURN_DV_KMS = 20.0 * FT_TO_KM       # ~6.1 m/s
MCC2_DV_KMS = 20.9 * FT_TO_KM                  # ~6.4 m/s
# [ASSUMED] The reports used here give the MAGNITUDE of these burns but not
# their direction. The separation burn is modelled as retrograde (it was
# described as cancelling a targeted overspeed); the MCC-2 direction is
# SOLVED so the spacecraft reaches the documented pericynthion.

# --- Lunar orbit [DOC: Mission Report sec. 3.0 / 4.6] ------------------------
NM_TO_KM = 1.852
LUNAR_RADIUS_KM = 1737.4
LOI1_ORBIT_ALT_NM = (60.9, 169.9)              # pericynthion x apocynthion after LOI
LOI2_ORBIT_ALT_NM = (54.4, 66.1)               # after circularisation
PERICYNTHION_ALT_KM = LOI1_ORBIT_ALT_NM[0] * NM_TO_KM
# [DOC] Mission Report sec. 4.6: LOI was "a 6-minute service propulsion maneuver".
# [DERIVED] Burn straddles pericynthion: pericynthion = ignition + half the duration.
LOI1_BURN_DURATION_S = 360.0
# [ASSUMED] Lunar stay is idealised as a circular orbit at the mean LOI-2
# altitude, in the plane of the approach. It matters only for the Moon-centred
# geometry at TEI; at Earth scale it is invisible.
LUNAR_ORBIT_RADIUS_KM = LUNAR_RADIUS_KM + 0.5 * sum(LOI2_ORBIT_ALT_NM) * NM_TO_KM

# --- Entry interface [DOC] ----------------------------------------------------
# Altitude 400,000 ft; Entry PAD / Flight Journal: speed 36,194 ft/s, flight-path
# angle -6.49 deg. Geocentric distance/latitude derived from the Mission Report
# entry table by Hutchison: r = 6500.02 km, geocentric latitude = -3.17 deg.
EI_GEOCENTRIC_DISTANCE_KM = 6500.02
EI_GEOCENTRIC_LATITUDE_DEG = -3.17
EI_FLIGHT_PATH_ANGLE_DEG = -6.49
EI_SPEED_FTS = 36194.0                         # used only as an independent check
EI_SPEED_KMS = EI_SPEED_FTS * FT_TO_KM

# --- Physical constants -------------------------------------------------------
MU_EARTH = 398600.435507     # km^3/s^2 (JPL)
MU_MOON = 4902.800066
MU_SUN = 1.32712440018e11
R_EARTH = 6378.137
J2_EARTH = 1.08262668e-3
