from numpy import pi
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
import math

# --- Constants ---
# matter contribution
om_m = 0.308
# cosmological constant contribution
om_la = 0.692
# Hubble constant, in SI units (s^-1)
h_0 = 2.1972e-18
# Lyman-alpha transition frequency (Hz)
nu_a = 2.4661e15
# Proton mass (kg)
m_p = 1.6726e-27
# Boltmann constant (JK^-1)
k_B = 1.3806e-23
# Speed of light (ms^-1)
c = 2.9979e8
# Prefactor I_alpha as given in Choudhury et al. (2001) (C2001), (cm^-2)
i_al = 4.45e-18

# --- Data ---
zs = np.loadtxt("Input_0_Redshift_axis.txt")
nHIs = np.loadtxt("Input_1_nHI_Field.txt")
ts = np.loadtxt("Input_2_Temperature_Field.txt")
vPecs = np.loadtxt("Input_3_Line_of_Sight_Velocity_Field.txt")

# Convert temperature to b as defined in (C2001), equation 31
bs = [math.sqrt(2.0 * k_B * t / m_p) for t in ts]

# -- Calculation --
# Voigt function (Galaxy Formation and Evolution, equation 16.104)
def voigt(A, B):
	integrand = lambda y: exp(-y ** 2.0) / ((B - y) ** 2.0 + A ** 2.0)
	return A * si.quad(integrand, -math.inf, math.inf) / pi

# Differential of redshift in terms of integration variable in (C2001)
# equation 30; assume no radiation or curvature contributions
def dz(dx, z):
	return c * math.sqrt(om_la + om_m * (1.0 + z) ** 3.0) / h_0