import numpy as np
import scipy.integrate as si
import scipy.special as ss
import matplotlib.pyplot as plt
import matplotlib.lines as ml
import math
import sys
import os

print(os.path.join(os.path.dirname(__file__), "..", "..", "read_spec_ewald_script.py"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "read_spec_ewald_script.py"))

from numpy import pi
from read_spec_ewald_script import spectra

# -----------------
# --- Constants ---
# -----------------

# Matter contribution
om_m = 0.308
# Cosmological constant contribution
om_La = 0.692
# Hubble constant, in SI units (s^{-1})
H_0 = 2.1972e-18
# Transition frequency (Hz)
# (https://physics.nist.gov/PhysRefData/ASD/lines_form.html) [NIST]
nu_12 = 2.3023e15
# Quantum-mechanical damping constant [NIST]
Ga = 3.41e8
# Particle mass (kg)
m_p = 2.6788e-26
# Boltzmann constant (JK^{-1})
k_B = 1.3806e-23
# Speed of light (ms^{-1})
c = 2.9979e8
# Prefactor I_\alpha to the integral, calculated using above constants and
# https://www.astro.ncu.edu.tw/~wchen/Courses/ISM/04.EinsteinCoefficients.pdf
# and the 'atom.dat' data file of VPFIT 10.4 
I_al = 5.5e-19
# Ionising background in units of 10^{-12}s^{-1}
Ga_UV = 7.0
# \sqrt{\pi}
sqrt_pi = math.sqrt(pi)

# ------------
# --- Data ---
# ------------

# Upper z limit
z_max = "3.000"

def filename(x):
	return "./los/" + x + "2048_n5000_z" + zmax + ".dat"

flag_spectype = "se_onthefly"
spec_obj = spectra(flag_spectype, filename("los"), taufilename = filename("tau"))

# All in SI units
# Neutral hydrogen fraction
fnss = np.transpose(spec_obj.nHI_frac)
# Hydrogen number density
nHss = np.transpose(spec_obj.rhoH2rhoHmean) * 1.0e6
# Temperature
Tss = np.transpose(spec_obj.temp_HI)
# Peculiar velocity along the line of sight
vss = np.transpose(spec_obj.vel_HI) * 1000.0

# Number of elements in a sightline
count = len(fnss[:, 0])

# Neutral hydrogen number density
def nHIs(n):
	return np.multiply(fnss[:, n], nHss[:, n])

# Convert temperature to b as defined in Choudhury et al. (2001) [C2001],
# equation 31, for the nth sightline
def bs(n):
	return np.sqrt(2.0 * k_B * Tss[:, n] / m_p)

# Box size
box = spec_obj.box

# -----------------
# -- Calculation --
# -----------------

# See [C2001] equation 30; assume no radiation or curvature contributions
def dz_by_dx(z): 
	return (H_0 / c) * (om_La + om_m * (1.0 + z) ** 3.0) ** 0.5

# Compute redshift axis
zs = np.full(count, float(z_max))
for i in range(count - 1, 0, -1):
	z = zs[i + 1]
	zs[i] = s - dz_by_dx(z) * box /count

# Voigt function computed from the Faddeeva function
def voigt(As, Bs):
	return ss.wofz(Bs + As * 1.0j).real

# 2nd argument to be passed to the Voigt function in [C2001] equation 30, for
# the nth sightline
def vArg2s(n, z0):
	return (vss[:, n] + c * (zs - z0) / (1.0 + z0)) / bs(n)

# The \alpha used in the 1st argument of the Voigt function in equation 30 in
# [C20001], for the nth sightline
def als(n):
	return c * Ga / (4 * pi * nu_12 * bs(n))

# The overdensity at which a region becomes 'self-shielded' (Keating et al.
# (2015)), computed for the nth sightline.
def cutoffsSS(n):
	T4s = Tss[:, n] / 1.0e4
	p1 = 2.0 / 3.0
	p2 = 2.0 / 15.0
	zFactors = ((1.0 + zs) / 7.0) ** -3.0
	return 54.0 * (Ga_UV ** p1) * (T4s ** p2) * zFactors

# The number density of neutral oxygen at a point, for the nth sightline
def nOIs(n):
	dtyAve = 0.0 # TODO Compute average density
	overdensities = (nHIss[:, n] - np.full(count, dtyAve)) / dtyAve
	fOI = np.heaviside(overdensities - cutoffsSS(n), 1.0)
	return fOI # TODO metallicity

# The integrand as in [C2001] equation 30 except with a change of variables to
# be an integral over z, for the nth sightline
def integrand1s(n, z0, f_scale):
	prefactor = c * I_al / sqrt_pi
	voigtFn = voigt(als(n), vArg2s(n, z0))
	measure = dz_by_dx(zs)
	return prefactor * measure * voigtFn * nHIs(n) / (bs(n) * (1.0 + zs)) # TODO return to OI

# Optical depth of the nth sightline from the farthest redshift up to z0, for
# the nth sightline; we integrate using Simpson's rule over all the points that
# fall in the region and assume the redshifts are in increasing order
def opticalDepth(n, z0, f_scale):
	return si.simps(integrand1s(n, z0, f_scale), zs)

def output1s(n, f_scale):
	return np.array([opticalDepth(n, z0, f_scale) for z0 in zs])

# Attenuation coefficient
def output2s(n, f_scale):
	return np.exp(-output1s(n, f_scale))


# --------------
# -- Plotting --
# --------------

depthLabel = "$\\tau_\\mathrm{O\\,I}$"
fluxLabel = "$F=e^{-" + depthLabel[1 : len(depthLabel) - 1] + "}$"

# Optical depth and flux
def plot1(n, f_scale):
	plt.title(f"Optical depth for sightline {n + 1}")
	plt.plot(zs, output2s(n))
	plt.xlabel("$z$")
	plt.ylabel(fluxLabel)
	plt.show()

# Main
n = 0
if len(sys.argv) > 0:
	n = int(sys.argv[1]) - 1
plot1(n)