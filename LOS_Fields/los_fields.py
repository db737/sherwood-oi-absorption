from numpy import pi
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
import math

# -----------------
# --- Constants ---
# -----------------

# Matter contribution
om_m = 0.308
# Cosmological constant contribution
om_La = 0.692
# Hubble constant, in SI units (s^{-1})
H_0 = 2.1972e-18
# Lyman-alpha transition frequency (Hz)
nu_12 = 2.4661e15
# Lyman-alpha damping constant
Ga = 6.265e8
# Proton mass (kg)
m_p = 1.6726e-27
# Boltmann constant (JK^{-1})
k_B = 1.3806e-23
# Speed of light (ms^{-1})
c = 2.9979e8
# Prefactor I_\alpha as given in Choudhury et al. (2001) [C2001], (m^2)
I_al = 4.45e-22
# \sqrt{\pi}
sqrt_pi = math.sqrt(pi)

# ------------
# --- Data ---
# ------------

zs = np.loadtxt("Input_0_Redshift_axis.txt")
nHIss = np.loadtxt("Input_1_nHI_Field.txt") * 1.0e6
Tss = np.loadtxt("Input_2_Temperature_Field.txt")
vss = np.loadtxt("Input_3_Line_of_Sight_Velocity_Field.txt") * 1000.0

# Convert temperature to b as defined in [C2001], equation 31, for the nth
# sightline
def bs(n):
	return np.sqrt(2.0 * k_B * Tss[:, n] / m_p)

# Measure of integration in [C2001] equation 30 in terms of that of redshift;
# assume no radiation or curvature contributions
dxs = (c / H_0) * (om_La + om_m * (1.0 + zs) ** 3.0) ** -0.5

# -- Calculation --
# Voigt function (Galaxy Formation and Evolution (H. Mo, F. Bosch and
# S. White) [GFaE], equation 16.104)
def voigt(A, B):
	integrand = lambda ys: np.exp(-ys ** 2.0) / ((B - ys) ** 2.0 + A ** 2.0)
	integral, err = si.quad(integrand, -math.inf, math.inf)
	return A * integral / pi

# The approximation to the Voigt function given in [GFaE] equation 16.106
def voigtApprox(A, B):
	return np.exp(-B ** 2.0) + A / (sqrt_pi * (A ** 2.0 + B ** 2.0))

# 2nd argument to be passed to the Voigt function int [C2001] equation 30, for
# the nth sightline
def vArg2s(n, z0):
	return (vss[:, n] + c * (zs - z0) / (1.0 + z0)) / bs(n)

# The \alpha used in the 1st argument of the Voigt function in equation 30 in
# [C20001], for the nth sightline
def als(n):
	return c * Ga / (4 * pi * nu_12 * bs(n))

# The integrand as in [C2001] equation 30 except with a change of variables to
# be an integral over z, for the nth sightline
def integrand1s(n, z0):
	prefactor = c * I_al / sqrt_pi
	voigtFn = voigtApprox(als(n), vArg2s(n, z0))
	return prefactor * dxs * voigtFn * nHIss[:, n] / (bs(n) * (1.0 + zs))

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

def plot(n):
	plt.subplot(211)
	plt.semilogy(zs, output1s(n))
	plt.title(f"Optical depth for sightline {n + 1}")
	plt.ylabel("$\\tau_{HI}$")
	plt.subplot(212)
	plt.plot(zs, output2s(n))
	plt.ylabel("$F=e^{-\\tau_{HI}}$")
	plt.xlabel("$z$")
	plt.show()

plot(2)