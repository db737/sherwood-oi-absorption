from numpy import pi
import numpy as np
import scipy.integrate as si
import scipy.special as ss
import matplotlib.pyplot as plt
import matplotlib.lines as ml
import math
import sys

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
# TODO: find more accurate OI value
I_al = 1.971e-22
# \sqrt{\pi}
sqrt_pi = math.sqrt(pi)

# ------------
# --- Data ---
# ------------

# All in SI units
zs = np.loadtxt("Input_0_Redshift_axis.txt")
nHIss = np.loadtxt("Input_1_nHI_Field.txt") * 1.0e6
Tss = np.loadtxt("Input_2_Temperature_Field.txt")
vss = np.loadtxt("Input_3_Line_of_Sight_Velocity_Field.txt") * 1000.0

# Convert temperature to b as defined in Choudhury et al. (2001) [C2001],
# equation 31, for the nth sightline
def bs(n):
	return np.sqrt(2.0 * k_B * Tss[:, n] / m_p)

# Measure of integration in [C2001] equation 30 in terms of that of redshift;
# assume no radiation or curvature contributions
dxs = (c / H_0) * (om_La + om_m * (1.0 + zs) ** 3.0) ** -0.5

# -- Calculation --
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

# The integrand as in [C2001] equation 30 except with a change of variables to
# be an integral over z, for the nth sightline
def integrand1s(n, z0, f_scale):
	prefactor = c * I_al / sqrt_pi
	voigtFn = voigt(als(n), vArg2s(n, z0))
	return prefactor * dxs * voigtFn * nHIss[:, n] * f_scale / (bs(n) * (1.0 + zs))

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

depthName = "$\\tau_\\mathrm{O\\,I}$"
fluxName = "$F=e^{-" + depthName[1 : len(depthName) - 1] + "}$"

# Optical depth and flux
def plot1(n, f_scale):
	plt.subplot(211)
	plt.title(f"Optical depth for sightline {n + 1}")
	plt.semilogy(zs, output1s(n, f_scale))
	plt.ylabel(depthName)
	plt.subplot(212)
	plt.plot(zs, output2s(n, f_scale))
	plt.xlabel("$z$")
	plt.ylabel(fluxName)
	plt.show()

# A single line for plot2
def plot2single(n, color, f_scale):
	plt.plot(zs, output2s(n, f_scale), color = color)
	powerstring = str(np.log10(f_scale))
	powerstring = powerstring[: len(powerstring) - 2]
	return ml.Line2D([], [], color = color, label = f"$f_{{scale}} = 10^{{{powerstring}}}$")

# Varying hydrogen to oxygen fraction
def plot2(n):
	plt.subplot(211)
	plt.title(f"Observed flux for different oxygen to hydrogen ratios $f_{{scale}}$ for sightline ${n + 1}$")
	count = 3
	colors = ["r", "g", "b"]
	f_scales = [1000.0, 100.0, 10.0]
	lines = [None] * count
	for i in range(0, count):
		lines[i] = plot2single(n, colors[i], f_scales[i])
	plt.ylabel(fluxName)
	plt.legend(handles = lines, loc = "center right", framealpha = 0.95)
	plt.subplot(212)
	colors = ["k", "c", "m"]
	f_scales = [1.0, 0.1, 0.01]
	for i in range(0, count):
		lines[i] = plot2single(n, colors[i], f_scales[i])
	plt.xlabel("$z$")
	plt.ylabel(fluxName)
	plt.legend(handles = lines, loc = "center right", framealpha = 0.95)
	plt.show()

# Main
n = 0
if len(sys.argv) > 0:
	n = int(sys.argv[1]) - 1
plot2(n)