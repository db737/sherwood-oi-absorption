import numpy as np
import scipy.integrate as si
import scipy.special as ss
import scipy.constants as consts
import matplotlib.pyplot as plt
import matplotlib.lines as ml
import matplotlib
import math
import sys

from numpy import pi
from read_spec_ewald_script import spectra

matplotlib.rcParams["text.usetex"] = True
plt.style.use("custom_plot_style.py")

# Middle z value
z_mid = "3.000"

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
# Transition frequency
# (https://physics.nist.gov/PhysRefData/ASD/lines_form.html) [NIST]
nu_12 = 2.3023e15
# Quantum-mechanical damping constant [NIST]
Ga = 3.41e8
# HI mass
m_HI = consts.value("proton mass") + consts.value("electron mass")
# OI mass
m_OI = 8.0 * (m_HI + consts.value("neutron mass"))
k_B = consts.value("Boltzmann constant")
c = consts.value("speed of light in vacuum")
# Prefactor I_{\alpha} to the integral, calculated using above constants and
# https://www.astro.ncu.edu.tw/~wchen/Courses/ISM/04.EinsteinCoefficients.pdf
# and the 'atom.dat' data file of VPFIT 10.4 
I_al = 5.5e-19
# Ionising background in units of 10^{-12}s^{-1}
Ga_UV = 7.0
# Solar metallicity
Z_solar = 0.0134

# Metallicity prefactor
Z_80 = Z_solar * 10.0 ** -2.65
# Metallicity exponent in formula 5 of Keating et al. (2014) [K2014]
n_Z = 1.3

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

# Number of elements in a sightline
count = len(fHIss[:, 0])

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
def dz_by_dx(z):
	return (H_0 / c) * (Om_La + Om_m0 * (1.0 + z) ** 3.0) ** 0.5

# Compute redshift axis
zs = np.full(count, float(z_mid))
middleIndex = (count - 1) // 2
for i in range(middleIndex - 1, -1, -1):
	z = zs[i + 1]
	zs[i] = z - dz_by_dx(z) * box / count
for i in range(middleIndex + 1, count):
	z = zs[i - 1]
	zs[i] = z + dz_by_dx(z) * box / count

# Average density of baryons
def rhBars():
	rh_crits = rh_crit0 * (Om_La + Om_m0 * (1.0 + zs) ** 3.0)
	return Om_b0 * rh_crits

# Neutral hydrogen number density
def nHIs(n):
	rh_bars = rh_crit0 * Om_b0 * (1.0 + zs) ** 3.0
	nHs = DeHss[:, n] * rh_bars / m_HI # Number density from mass density
	return nHs * fHIss[:, n]

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

# Metallicity using formula 5 from Keating et al. (2014) [K2014]
def Zs(n):
	return Z_80 * (DeHss[:, n] / 80.0) ** n_Z

# The overdensity at which a region becomes 'self-shielded' (Keating et al.
# (2016) [K2016]), computed for the nth sightline.
def cutoffsSS(n):
	T4s = Tss[:, n] / 1.0e4
	p1 = 2.0 / 3.0
	p2 = 2.0 / 15.0
	zFactors = ((1.0 + zs) / 7.0) ** -3.0
	return 54.0 * (Ga_UV ** p1) * (T4s ** p2) * zFactors

# The number density of neutral oxygen at a point, for the nth sightline
def nOIs(n):
	fOI = np.heaviside(DeHss[:, n] - cutoffsSS(n), 1.0)
	return fOI * Zs(n)

# The integrand as in [C2001] equation 30 except with a change of variables to
# be an integral over z, for the nth sightline
def integrand1s(n, z0):
	prefactor = c * I_al * math.pi ** -0.5
	voigtFn = voigt(als(n), vArg2s(n, z0))
	measure = 1.0 / dz_by_dx(zs)
	return prefactor * measure * voigtFn * nHIs(n) / (bs(n, m_HI) * (1.0 + zs)) # TODO return to OI

# Optical depth of the nth sightline from the farthest redshift up to z0, for
# the nth sightline; we integrate using Simpson's rule over all the points that
# fall in the region and assume the redshifts are in increasing order
def opticalDepth(n, z0):
	return si.simps(integrand1s(n, z0), zs)

def output1s(n):
	return np.array([opticalDepth(n, z0) for z0 in zs])

# Attenuation coefficient
def output2s(n):
	return np.exp(-output1s(n))

# --------------
# -- Plotting --
# --------------

hiLabel = '\mbox{H\,\sc{i} }'
oiLabel = '\mbox{O\,\sc{i} }'
depthLabel = '$\tau_{' + oiLabel + "}$"
fluxLabel = "$F=e^{-" + depthLabel[1 : len(depthLabel) - 1] + "}$"

# Optical depth and flux
def plot1(n):
	plt.title(f"Optical depth for sightline {n + 1}")
	plt.plot(zs, output2s(n))
	plt.xlabel("$z$")
	plt.ylabel(fluxLabel)
	plt.show()

# Check that overdensity averages to 1 for a given redshift
def test1():
	Des = DeHss[middleIndex, :]
	print(np.mean(Des))

# Check that the ionised fraction agrees with what we would expect from the
# photoionisation equation
def test2(n):
	# Fitted parameters for photoionisation based on temperature
	a = 7.982e-11
	b = 0.748
	T0 = 3.148
	T1 = 7.036e5
	A = 2.91e-8
	E = 13.6
	X = 0.232
	m = 0.39
	# Helium fraction
	Y = 0.2485
	f0s = np.sqrt(Tss[:, n] / T0)
	f1s = np.sqrt(Tss[:, n] / T1)
	Us = 11604.5 * E / Tss[:, n]
	# Recombination rates
	als = a / (f0s * ((1.0 + f0s) ** (1.0 - b)) * ((1.0 + f1s) ** (1.0 + b)))
	# Collisional ionisation rates
	gas = A * (Us ** m) * np.exp(-Us) / (X + Us)
	# Background ionisation rate
	j = 8.28e-13
	# Baryon number density, not in SI units
	nbs = nHIs(n) / fHIss[:, n] * 1.0e-6
	mue = 2.0 * (2.0 - Y) / (4.0 - 3.0 * Y)
	nhis = als * nbs / (als + gas + j / (mue * nbs))
	plt.title(f"Comparison of measured and computed neutral hydrogen number densities for sightline {n + 1}")
	plt.plot(zs, nHIs(n) * 1.0e5, "k")
	plt.plot(zs, nhis * 1.0e11, "b--")
	measured = ml.Line2D([], [], color = "k", label = "measured")
	computed = ml.Line2D([], [], color = "b", ls = "--", label = "computed")
	plt.xlabel("$z$")
	plt.ylabel("$n_{" + hiLabel + '} / \mathrm{10^{-11}\,cm^{-3}}$')
	plt.legend(handles = [measured, computed])
	plt.show()

# Main
n = 0
if len(sys.argv) > 0:
	n = int(sys.argv[1]) - 1
test2(n)