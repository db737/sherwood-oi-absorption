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
# Transition frequencies
# (https://physics.nist.gov/PhysRefData/ASD/lines_form.html) [NIST]
nu_12_HI = 2.4661e15
nu_12_OI = 2.3023e15
# Quantum-mechanical damping constants [NIST]
Ga_HI = 6.265e8
Ga_OI = 3.41e8
# HI mass
m_HI = consts.value("proton mass") + consts.value("electron mass")
# OI mass
m_OI = 8.0 * (m_HI + consts.value("neutron mass"))
k_B = consts.value("Boltzmann constant")
c = consts.value("speed of light in vacuum")
# Prefactors I_{\alpha} to the integral, calculated using above constants and
# https://www.astro.ncu.edu.tw/~wchen/Courses/ISM/04.EinsteinCoefficients.pdf
# and the 'atom.dat' data file of VPFIT 10.4 
I_al_HI = 4.45e-22
I_al_OI = 5.5e-23
# Ionising background in units of 10^{-12}s^{-1}
Ga_12 = 0.158
# Solar metallicity
Z_solar = 0.0134
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

# Neutral hydrogen number density
def nHIs(n):
	rh_bars = rh_crit0 * Om_b0 * (1.0 + zs) ** 3.0
	nHs = DeHss[:, n] * rh_bars / m_HI # Number density from mass density
	return nHs * fHIss[:, n] * (1.0 - Y)

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

# Metallicity using formula 5 from Keating et al. (2014) [K2014]
def Zs(n):
	Z_80 = Z_solar * 10.0 ** -2.65
	return Z_80 * (DeHss[:, n] / 80.0) ** 1.3

# The overdensity at which a region becomes 'self-shielded' (Keating et al.
# (2016) [K2016]), computed for the nth sightline.
def cutoffsSS(n):
	T4s = Tss[:, n] / 1.0e4
	p1 = 2.0 / 3.0
	p2 = 2.0 / 15.0
	zFactors = ((1.0 + zs) / 7.0) ** -3.0
	return 54.0 * (Ga_12 ** p1) * (T4s ** p2) * zFactors

# The number density of neutral oxygen at a point, for the nth sightline
def nOIs(n):
	fOI = np.heaviside(DeHss[:, n] - cutoffsSS(n), 1.0)
	return fOI * Zs(n)

# The integrand as in [C2001] equation 30 except with a change of variables to
# be an integral over z, for the nth sightline; 'hydrogen' is a boolean setting
def integrand1s(n, z0, hydrogen):
	mass = m_HI if hydrogen else m_OI
	ns = nHIs(n) if hydrogen else nOIs(n)
	I_al = I_al_HI if hydrogen else I_al_OI
	prefactor = c * I_al * math.pi ** -0.5
	voigtFn = voigt(als(n, hydrogen), vArg2s(n, z0, mass))
	measure = 1.0 / dz_by_dx(zs)
	return prefactor * measure * voigtFn * ns / (bs(n, mass) * (1.0 + zs))

# Optical depth of the nth sightline from the farthest redshift up to z0, for
# the nth sightline; we integrate using Simpson's rule over all the points that
# fall in the region and assume the redshifts are in increasing order
def opticalDepth(n, z0, hydrogen):
	return si.simps(integrand1s(n, z0, hydrogen), zs)

def opticalDepths(n, hydrogen):
	return np.array([opticalDepth(n, z0, hydrogen) for z0 in zs])

# Attenuation coefficient
def fluxes(n, hydrogen):
	return np.exp(-opticalDepths(n, hydrogen))

# --------------
# -- Plotting --
# --------------

hiLabel = 'HI'
oiLabel = 'OI'
depthLabel = '$\\tau_{' + oiLabel + "}$"
fluxLabel = "$F=e^{-" + depthLabel[1 : len(depthLabel) - 1] + "}$"

# Optical depth and flux
def plot1(n):
	plt.title(f"Optical depth for sightline {n + 1}")
	plt.plot(zs, fluxes(n, False))
	plt.ylim([0.0, 1.1])
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

# Compare Neutral hydrogen optical depths
def test3(n):
	plt.title(f"Comparison of simulation output and computed neutral hydrogen optical depths for sightline {n + 1}")
	plt.plot(zs, ta_HIss[:, n], "k")
	plt.plot(zs, opticalDepths(n, True), "b--")
	measured = ml.Line2D([], [], color = "k", label = "from simulation")
	computed = ml.Line2D([], [], color = "b", ls = "--", label = "computed")
	plt.legend(handles = [measured, computed])
	plt.show()

# Metallicity, self-shielding etc.
def test4(n):
	fig, axes = plt.subplots(5, 1, sharex = True)
	axes[0].set_title(f"Oxygen properties for sightline {n + 1}, using $\\Gamma_{{12}}={Ga_12}$ and $z={z_mid}$")
	axes[0].semilogy(zs, Zs(n))
	axes[0].set_ylabel('$Z$')
	axes[1].plot(zs, cutoffsSS(n))
	axes[1].set_ylabel('$\Delta_{ss}$')
	axes[2].plot(zs, np.heaviside(DeHss[:, n] - cutoffsSS(n), 1.0))
	axes[2].set_ylabel("Self-shielding", fontsize = 12, rotation = "horizontal")
	axes[2].set_yticklabels([])
	axes[3].semilogy(zs, nOIs(n) / 1.0e6)
	axes[3].set_ylabel("$n_{" + oiLabel + '} / \mathrm{cm^{-3}}$')
	axes[4].plot(zs, fluxes(n, False))
	axes[4].set_xlabel("$z$")
	axes[4].set_ylabel("$F$")
	axes[4].set_ylim([0.0, 1.1])
	plt.subplots_adjust(hspace = 0)
	fig.align_ylabels()
	plt.show()
	
# Check inputs are as expected
def check1(n):
	print("HI fraction: {}".format(fHIss[0, n]))
	print("Overdensity: {}".format(DeHss[0, n]))
	print("Temperature: {}".format(Tss[0, n]))
	print("Velocity: {}".format(vss[0, n]))
	print("HI depth: {}".format(ta_HIss[0, n]))
	print("HI number density: {}".format(nHIs(n)[0]))

# Check initial computed quantities are as expected
def check2(n):
	print("b: {}".format(bs(n, m_HI)[0]))
	print("z: {}".format(zs[0]))
	print("dz/dx: {}".format(dz_by_dx(zs)[0]))
	print("alpha (HI): {}".format(als(n, True)[0]))
	print("alpha (OI): {}".format(als(n, False)[0]))
	print("V arg 2: {}".format(vArg2s(n, 3.0, m_HI)[0]))
	print("V(..., ...): {}".format(voigt(als(n, True), vArg2s(n, 3.0, m_HI))[0]))
	print("integrand: {}".format(integrand1s(n, 3.0, True)[0]))

# Compare hydrogen outputs directly
def check3(n):
	print("from data: {}".format(ta_HIss[0, n]))
	print("computed: {}".format(opticalDepth(n, zs[0], True)))

# Check oxygen stuff
def check4(n):
	print("Zs: {}".format(Zs(n)))
	print("cutoffs: {}".format(cutoffsSS(n)))

# Main
n = 0
if len(sys.argv) > 0:
	n = int(sys.argv[1]) - 1
test3(n)