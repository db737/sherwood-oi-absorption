import numpy as np
import scipy.integrate as si
import scipy.special as ss
import scipy.constants as consts
import scipy.signal as spsig
import math

from numpy import pi
from scipy import ndimage
from read_spec_ewald_script import spectra

# Middle z value
z_mid = "5.600"

def filename(x):
	return "../../los/" + x + "2048_n5000_z" + z_mid + ".dat"

flag_spectype = "se_onthefly"
spec_obj = spectra(flag_spectype, filename("los"), taufilename = filename("tau"))

# -----------------
# --- Constants ---
# -----------------

# All in SI units unless otherwise stated
G = consts.value("Newtonian constant of gravitation")
# Matter contribution
Om_m0 = spec_obj.om
# Baryon contribution
Om_b0 = spec_obj.ob
# Cosmological constant contribution
Om_La = spec_obj.ol
# Hubble constant
H_0 = spec_obj.H0
# Critical density now
rh_crit0 = 3.0 * H_0 ** 2.0 / (8.0 * pi * G)
# Hydrogen fraction
x_H = spec_obj.xh
# Transition frequencies
# (https://physics.nist.gov/PhysRefData/ASD/lines_form.html) [NIST]
nu_12_HI = 2.4661e15
nu_12_OI = 2.3023e15
# Quantum-mechanical damping constants from the 'atom.dat' data file of
# VPFIT 10.4 [atom.dat]
Ga_HI = 6.265e8
Ga_OI = 5.65e8
# HI mass
m_HI = consts.value("proton mass") + consts.value("electron mass")
# OI mass
m_OI = 8.0 * (m_HI + consts.value("neutron mass"))
k_B = consts.value("Boltzmann constant")
c = consts.value("speed of light in vacuum")
# Prefactors I_{\alpha} to the integral, calculated using above constants and
# https://www.astro.ncu.edu.tw/~wchen/Courses/ISM/04.EinsteinCoefficients.pdf
# and [atom.dat]
I_al_HI = 4.45e-22
I_al_OI = 5.5e-23
# Ionising background in units of 10^{-12}s^{-1}
Ga_12 = 0.158
# Solar metallicity from Keating et al. (2014) [K2014]
Z_solar_oxygen = 10.0**-3.13
# Helium fraction
Y = 0.2485

# ------------
# --- Data ---
# ------------

# All in SI units
# Neutral hydrogen fraction
fHIss = np.transpose(spec_obj.nHI_frac)
# Hydrogen overdensity
DeHss = np.transpose(spec_obj.rhoH2rhoHmean)
# Temperature
Tss = np.transpose(spec_obj.temp_HI)
# Peculiar velocity along the line of sight
vss = np.transpose(spec_obj.vel_HI) * 1.0e3
# HI optical depths
ta_HIss = np.transpose(spec_obj.tau_HI)

# Number of sightlines
num = len(fHIss[0, :])

# Number of elements in a sightline
count = len(fHIss[:, 0])

# The minimum number of points away before we consider two troughs to be a
# single one
min_dist = count // 100

# The maximum number of points away before we consider a trough to have ended
max_dist = count // 40

# The threshold distance between neighbouring points to count as a peak
thresh = 1.0e-4

# Convert temperature to b as defined in Choudhury et al. (2001) [C2001],
# equation 31, for the nth sightline
def bs(n, mass):
	return np.sqrt(2.0 * k_B * Tss[:, n] / mass)

# Box size (box is in units of h^{-1} ckPc)
box = spec_obj.box * 1.0e3 * consts.parsec / spec_obj.h

# -----------------
# -- Calculation --
# -----------------

# See [C2001] equation 30; assume no radiation or curvature contributions
def dz_by_dX(z):
	return (H_0 / c) * (Om_La + Om_m0 * (1.0 + z) ** 3.0) ** 0.5

# Compute redshift axis
zs = np.full(count, float(z_mid))
middleIndex = (count - 1) // 2
for i in range(middleIndex - 1, -1, -1):
	z = zs[i + 1]
	zs[i] = z - dz_by_dX(z) * box / count
for i in range(middleIndex + 1, count):
	z = zs[i - 1]
	zs[i] = z + dz_by_dX(z) * box / count

# Compute baryon number densities
rh_bars = rh_crit0 * Om_b0 * (1.0 + zs) ** 3.0

# Voigt function computed from the Faddeeva function
def voigt(As, Bs):
	return ss.wofz(Bs + As * 1.0j).real

# The \alpha used in the 1st argument of the Voigt function in equation 30 in
# [C20001], for the nth sightline
def als(n, hydrogen):
	mass = m_HI if hydrogen else m_OI
	Ga = Ga_HI if hydrogen else Ga_OI
	nu_12 = nu_12_HI if hydrogen else nu_12_OI
	return c * Ga / (4 * pi * nu_12 * bs(n, mass))

# 2nd argument to be passed to the Voigt function in [C2001] equation 30, for
# the nth sightline
def vArg2s(n, z0, mass):
	return (vss[:, n] + c * (zs - z0) / (1.0 + z0)) / bs(n, mass)

# Neutral hydrogen number density
def nHIs(n):
	nHs = DeHss[:, n] * rh_bars / m_HI # Number density from mass density
	return nHs * fHIss[:, n] * (1.0 - Y)

# Metallicity using formula 5 from [K2014]
def Zs(n):
	Z_80 = Z_solar_oxygen * 10.0 ** -2.65
	return Z_80 * (DeHss[:, n] / 80.0) ** 1.3

# The overdensity at which a region becomes 'self-shielded' (Keating et al.
# (2016) [K2016]), computed for the nth sightline.
def cutoffsSS(n):
	T4s = Tss[:, n] / 1.0e4
	p1 = 2.0 / 3.0
	p2 = 2.0 / 15.0
	zFactors = ((1.0 + zs) / 7.0) ** -3.0
	return 54.0 * (Ga_12 ** p1) * (T4s ** p2) * zFactors

# The number density of neutral oxygen at a point, for the nth sightline; the
# second argument, if True, will assume OI is only present in self-shielded
# regions and nowhere else
def nOIs(n, ssOnly):
	ss = np.heaviside(DeHss[:, n] - cutoffsSS(n), 1.0)
	# Shift and scale the step function to get the unshielded neutral fraction
	fOI = ss if ssOnly else fHIss[:, n] + (1.0 - fHIss[:, n]) * ss
	return fOI * Zs(n) * DeHss[:, n] * rh_bars / m_OI

# The integrand as in [C2001] equation 30 except with a change of variables to
# be an integral over z, for the nth sightline; 'hydrogen' is a boolean setting
def integrand1s(n, z0, hydrogen, ssOnly):
	mass = m_HI if hydrogen else m_OI
	ns = nHIs(n) if hydrogen else nOIs(n, ssOnly)
	I_al = I_al_HI if hydrogen else I_al_OI
	prefactor = c * I_al * math.pi ** -0.5
	voigtFn = voigt(als(n, hydrogen), vArg2s(n, z0, mass))
	measure = 1.0 / dz_by_dX(zs)
	return prefactor * measure * voigtFn * ns / (bs(n, mass) * (1.0 + zs))

# Optical depth of the nth sightline from the farthest redshift up to z0, for
# the nth sightline; we integrate using Simpson's rule over all the points that
# fall in the region and assume the redshifts are in increasing order
def opticalDepth(n, z0, hydrogen, ssOnly):
	return si.simps(integrand1s(n, z0, hydrogen, ssOnly), zs)

def opticalDepths(n, hydrogen, ssOnly):
	return np.array([opticalDepth(n, z0, hydrogen, ssOnly) for z0 in zs])

# Attenuation coefficient
def fluxes(n, hydrogen, ssOnly):
	return np.exp(-opticalDepths(n, hydrogen, ssOnly))

# Find minima or maxima in the flux
def extrema(n, hydrogen, ssOnly, minima):
	flux_data = fluxes(n, hydrogen, ssOnly)
	if minima:
		flux_data = 1.0 - flux_data
	peaks, _ = spsig.find_peaks(flux_data, distance = min_dist, threshold = thresh)
	return peaks

# Force the value to fit within the indices of the data
def clamp(i):
	return max(0, min(count - 1, i))

# Find the next or previous element present in an array of indices
def adjacent(i, xs, prev):
	comp = np.less if prev else np.greater
	op = np.subtract if prev else np.add
	filtereds = list(filter(lambda x: comp(x, i), xs))
	if len(filtereds) == 0:
		return clamp(op(i, max_dist))
	else:
		if prev:
			return np.max(filtereds)
		else:
			return np.min(filtereds)

# Find the indices of the positions where an absorber starts and ends
def trough_boundaries(i, mins, maxes):
	prev_min = adjacent(i, mins, True)
	prev_max = adjacent(i, maxes, True)
	next_min = adjacent(i, mins, False)
	next_max = adjacent(i, maxes, False)
	prev = clamp(max(prev_min, prev_max) - 1)
	next = clamp(min(next_min, next_max) + 1)
	if prev_min > prev_max:
		prev = (i + prev_min) // 2
	if next_min < next_max:
		next = (i + next_min) // 2 + 1
	prev = max(prev, i - max_dist)
	next = min(next, i + max_dist)
	return prev, next

# Find the equivalent widths of all OI absorbers in the spectrum.
def equiv_widths(n, ssOnly):
	mins = extrema(n, False, ssOnly, True)
	maxes = extrema(n, False, ssOnly, False)
	print(f"mins, maxes {n + 1}")
	num_mins = len(mins)
	widths = np.zeros(num_mins)
	for j in range(0, num_mins):
		prev, next = trough_boundaries(mins[j], mins, maxes)
		print(f"boundaries {n + 1}, {j}")
		# The area above the trough equals its equivalent width
		width = si.simps(1.0 - fluxes(n, False, ssOnly)[prev : next], zs[prev : next])
		print(f"simpson {n + 1}, {j}")
		# Use units of angstroms
		widths[j] = width * c * 1.0e10 / nu_12_OI
	print(f"EW {n + 1}")
	return widths