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
om_la = 0.692
# Hubble constant, in SI units (s^{-1})
h_0 = 2.1972e-18
# Lyman-alpha transition frequency (Hz)
nu_12 = 2.4661e15
# Lyman-alpha intrinsic linewidth
# (https://www.roe.ac.uk/~jsd/Rad_Matt/notes_part2.pdf)
a_12 = 4.9804e8
# Proton mass (kg)
m_p = 1.6726e-27
# Boltmann constant (JK^{-1})
k_B = 1.3806e-23
# Speed of light (ms^{-1})
c = 2.9979e8
# Prefactor I_\alpha as given in Choudhury et al. (2001) [C2001], (cm^{-2})
i_al = 4.45e-18
# \sqrt{\pi}
sqrt_pi = math.sqrt(pi)

# ------------
# --- Data ---
# ------------

zs = np.loadtxt("Input_0_Redshift_axis.txt")
nHIss = np.loadtxt("Input_1_nHI_Field.txt")
tss = np.loadtxt("Input_2_Temperature_Field.txt")
vss = np.loadtxt("Input_3_Line_of_Sight_Velocity_Field.txt")

# Convert temperature to b as defined in [C2001], equation 31, for the nth
# sightline
def bs(n):
	return np.sqrt(2.0 * k_B * tss[:, n] / m_p)

# Find the position in the redshift array of z0 via a recursive binary search;
# if there is no match, take the lower of the two adjacent indices
def zIndex(z0, z0s):
	l = len(z0s)
	if l < 2:
		return 0
	else:
		splitAt = l // 2
		if z0 < z0s[splitAt]:
			return zIndex(z0, z0s[: splitAt])
		else:
			return splitAt + zIndex(z0, z0s[splitAt :])

# Measure of integration in [C2001] equation 30 in terms of that of redshift;
# assume no radiation or curvature contributions
dxs = (h_0 / c) * (om_la + om_m * (1.0 + zs) ** 3.0) ** -0.5

# -- Calculation --
# Voigt function (Galaxy Formation and Evolution (H. Mo, F. Bosch and
# S. White) [GFaE], equation 16.104)
def voigt(A, B):
	integrand = lambda y: np.exp(-y ** 2.0) / ((B - y) ** 2.0 + A ** 2.0)
	integral, err = si.quad(integrand, -math.inf, math.inf)
	return A * integral / pi

# The approximation to the Voigt function given in [GFaE] equation 16.106
def voigtApprox(A, B):
	return np.exp(-B ** 2.0) + A / (sqrt_pi * (A ** 2.0 + B ** 2.0))

# The approximation to the Voigt function given in [C2001] equation 33
def voigtApprox2(A, B):
	return np.exp(-B ** 2.0)

# 2nd argument to be passed to the Voigt function int [C2001] equation 30, for
# the nth sightline
def vArg2s(n, z0):
	return (vss[:, n] * 1000.0 + c * (zs - z0) / (1.0 + z0)) / bs(n)

# The \alpha used in the 1st argument of the Voigt function in equation 30 in
# [C20001], for the nth sightline
def als(n):
	return c * a_12 / (4 * pi * nu_12 * bs(n))

# The integrand as in [C2001] equation 30 except with a change of variables to
# be an integral over z, for the nth sightline
def integrand1s(n, z0):
	prefactor = c * i_al / sqrt_pi
	voigtFn = voigtApprox(als(n), vArg2s(n, z0))
	return prefactor * dxs * voigtFn * nHIss[:, n] / (bs(n) * (1.0 + zs))

# Optical depth of the nth sightline from the farthest redshift up to z0, for
# the nth sightline; we integrate using Simpson's rule over all the points that
# fall in the region and assume the redshifts are in increasing order
def opticalDepth(n, z0):
	index = zIndex(z0, zs)
	return -si.simps(integrand1s(n, z0)[index :], zs[index :])

def output1s(n):
	return np.array([opticalDepth(n, z0) for z0 in zs])

# Attenuation coefficient
def output2s(n):
	return np.exp(-output1s(n))

# --------------
# -- Plotting --
# --------------

plt.scatter(