
import matplotlib.pyplot as plt
from common_lib.plot_setting import load_plt

from scipy.interpolate import interp1d
import numpy as np
import scipy.integrate as si
import scipy.special as ss
import scipy.constants as consts
import scipy.signal as spsig
import math
import operator

from numpy import pi
from scipy import ndimage
from read_spec_ewald_script import spectra
from common_lib.statistics_functions.simple_cdf import COMPUTE_CDF

# los array
n_arr = np.arange(100,dtype=np.int32)

# Middle z value
z_mid = "6.000"

def filename(x, patchy = True):
    if patchy:
        return "/data/emergence12/prace_relics/planck1_40_2048_patchy/los/" + x + "2048_n5000_z" + z_mid + ".dat"
    else:
        return "/data/curie3/PRACE_ra1865/planck1_40_2048/los/" + x + "2048_n5000_z" + z_mid + ".dat"

flag_spectype = "se_onthefly"
spec_obj, spec_obj_patchy = None, None

def obtain_spec_objs():
    # TODO warn if trying to use non-existent tau data for bubbles
    global spec_obj, spec_obj_patchy
    spec_obj = spectra(flag_spectype, filename("los", patchy = False), taufilename = filename("tau", patchy = False))
    spec_obj_patchy = spectra(flag_spectype, filename("los"), taufilename = filename("tau", patchy = False))

obtain_spec_objs()

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
Ga_12 = 0.16 #0.36
# Solar metallicity from Keating et al. (2014) [K2014]
Z_solar_oxygen = 10.0**-3.13 
# Fraction of the solar metallicity to cap the metallicity at
cap_Z_frac = 0.01
# Helium fraction
Y = 0.2485

# Number of bins
num_bins = 128

# The threshold height to count as a peak
min_height = 0.01

# The minimum number of points away for 2 peaks to count as separate
min_dist = 6

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

# Overall flag for whether we are using patchy or homogeneous data
patchy = False

# Imperative function to switch to patchy data
def enable_bubbles():
    global x_H, fHIss, DeHss, Tss, vss, patchy
    x_H = spec_obj_patchy.xh
    fHIss = np.transpose(spec_obj_patchy.nHI_frac)
    DeHss = np.transpose(spec_obj_patchy.rhoH2rhoHmean)
    Tss = np.transpose(spec_obj_patchy.temp_HI)
    vss = np.transpose(spec_obj_patchy.vel_HI) * 1.0e3
    patchy = True

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

def redshift_array(midpoint):
    zs = np.full(count, midpoint)
    middleIndex = (count - 1) // 2
    for i in range(middleIndex - 1, -1, -1):
        z = zs[i + 1]
        zs[i] = z - dz_by_dX(z) * box / count
    for i in range(middleIndex + 1, count):
        z = zs[i - 1]
        zs[i] = z + dz_by_dX(z) * box / count
    return zs

# Compute redshift axis
zs = redshift_array(float(z_mid))

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
    ns = DeHss[:, n] * rh_bars / m_HI # Number density from mass density
    return ns * fHIss[:, n] * (1.0 - Y)

# Metallicity using formula 5 from [K2014]
def Zs(n):
    Z_80 = Z_solar_oxygen * 10.0 ** -2.65
    Z_data = Z_80 * (DeHss[:, n] / 80.0) ** 1.3
    #return np.clip(Z_data, None, Z_solar_oxygen * cap_Z_frac)
    return Z_data

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
    if ssOnly is None or patchy:
        ss = np.zeros(count)
    # Thss ensures that when ss is 0 we get fHIss[:, n] and when ss is 1.0 we get 1.0
    fOI = ss if ssOnly else fHIss[:, n] + (1.0 - fHIss[:, n]) * ss
    #return fOI * Zs(n) * DeHss[:, n] * rh_bars / m_OI #Z_solar_oxygen * nHIs(n)
    return Zs(n) * nHIs(n) * fOI * 0.36 /Ga_12

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
def extrema(flux_data, minima):
    if minima:
        peaks, _ = spsig.find_peaks(1.0 - flux_data, height = min_height, distance = min_dist)
    else:
        peaks, _ = spsig.find_peaks(flux_data)
    return peaks

# Find the next and previous element present in an array of indices
def adjacent(i, xs):
    prevs = list(filter(lambda x: x < i, xs))
    nexts = list(filter(lambda x: x > i, xs))
    prev = 0 if len(prevs) == 0 else np.max(prevs)
    next = count - 1 if len(nexts) == 0 else np.min(nexts)
    return prev, next

# Find the indices of the positions where an absorber starts and ends
def trough_boundaries(i, maxes, cuts):
    prev_max, next_max = adjacent(i, maxes)
    prev_cut, next_cut = adjacent(i, cuts)
    return max(prev_max, prev_cut), min(next_max, next_cut)

# The points where the spectrum dips above or below the peak height cutoff
def hcuts(flux_data):
    was_above = False
    outs = []
    for i in range(0, count):
        above = flux_data[i] + min_height > 1.0
        if operator.xor(was_above, above):
            outs.append(i)
    return outs

# Find the equivalent widths of all OI absorbers in the spectrum.
#def equiv_widths(n, ssOnly, tracking = None):
def equiv_widths(flux_data, tracking = None):
    #flux_data = fluxes(n, False, ssOnly)
    mins = extrema(flux_data, True)
    maxes = extrema(flux_data, False)
    num_mins = len(mins)
    #print("mins, maxes {n + 1}, count: {num_mins}")
    widths = np.zeros(num_mins)
    peak_zs = zs[mins]
    for j in range(0, num_mins):
        prev, next = trough_boundaries(mins[j], maxes, hcuts(flux_data))
        #print("boundaries {n + 1}, {j}")
        # The area above the trough equals its equivalent width
        #width = si.simps(1.0 - flux_data[prev : next], zs[prev : next])
        width = si.trapz(1.0 - flux_data[prev : next], (zs[prev : next] + 1.0) * c * 1e10 / nu_12_OI)
        #print("EW {n + 1}, {j}")
        # Use units of angstroms

        #widths[j] = width * c * 1.0e10 / nu_12_OI
        #print c * 1e10 /nu_12_OI 
        #print nu_12_OI
        #exit()
        widths[j] = width 

        if tracking is not None:
            if widths[j] >= tracking:
                print("Strong absorber: sightline {n + 1} with EW {widths[j]}")
    #print("EW {n + 1}")
    return peak_zs, widths

def compute_EW_cdf(EW_arr,n_arr):
    dX_val       = 0.6538 
    N_spectra    = n_arr.shape[0]
    EW_cdf_norm  = COMPUTE_CDF(EW_arr)
    EW_total     = EW_arr.shape[0]
    EW_cdf       = np.zeros(EW_cdf_norm.shape)
    EW_cdf[:,0]  = EW_cdf_norm[:,0]
    EW_cdf[:,1]  = (1.0-EW_cdf_norm[:,1]) * EW_total / (dX_val*N_spectra)
    return EW_cdf

def compute_equivalent_width_prakash(Flux_arr,z_arr_sim):
    lambda_rest = 1302.01685
    lambda_arr  = (1.0 + z_arr_sim) * lambda_rest
    F_th        = 0.999
    F_th_valley = F_th
    F_th_inv    = 1.0 - F_th
    N_los       = Flux_arr.shape[1]
    W_list      = []

    for los_indx in xrange(N_los):

        Flux_arr_1d = Flux_arr[:,los_indx]
        peaks,_     = spsig.find_peaks(1.0-Flux_arr_1d,height=F_th_inv)

        for peak_indx in peaks:
            lambda_peak     = lambda_arr[peak_indx]
            Flux_val_peak   = Flux_arr_1d[peak_indx]
            bool_arr        = Flux_arr_1d > F_th_valley
            bool_arr_lb     = bool_arr & (lambda_arr < lambda_peak)
            bool_arr_ub     = bool_arr & (lambda_arr > lambda_peak)

            if np.sum(bool_arr_lb) == 0:
                lambda_lb   = lambda_arr[0]
            else:
                lambda_lb   = np.amax(lambda_arr[bool_arr_lb])

            if np.sum(bool_arr_ub) == 0:
                lambda_ub       = lambda_arr[-1]
            else:
                lambda_ub       = np.amin(lambda_arr[bool_arr_ub])

            bool_arr_region = (lambda_lb <= lambda_arr) & (lambda_arr <= lambda_ub)
            #W_val           =  trapz(1.0 - Flux_arr_1d[bool_arr_region],lambda_arr[bool_arr_region])
            W_val           =  si.simps(1.0 - Flux_arr_1d[bool_arr_region],lambda_arr[bool_arr_region])
            #W_val           =  np.sum(1.0 - Flux_arr_1d[bool_arr_region])
            W_list.append(W_val)
        #print "Analysis done for los_indx =",los_indx

    W_arr = np.array(W_list)
    return W_arr


def plot_observation_data(ax,data="2011"):
    inp_path   = "./"
    inp_file   = "Becker_2011_OI_CDF_Linear.txt"
    data_arr   = np.loadtxt(inp_path + inp_file)
    ax.plot(data_arr[:,0],data_arr[:,1],ls="-",color="k",lw=3.0,label="Becker+2011")
    #W_cdf_obs[:,1] = data_arr[:,2]


hydrogen = False
ssOnly = False
#out_file_tau = "OI_Optical_Depth_10_percent_Solar_Metallicity.txt" #"OI_Optical_Depth.txt"
out_file_tau = "OI_Optical_Depth_Solar_Metallicity.txt" #"OI_Optical_Depth.txt"

#for indx,n in enumerate(n_arr):
#    tau_OI_arr_indx = opticalDepths(n, hydrogen, ssOnly)
#    if indx == 0:
#        tau_OI_arr = np.zeros((count,n_arr.shape[0]))
#    tau_OI_arr[:,indx] = tau_OI_arr_indx.copy()
#    print "OI Optical depth calculated for indx :",indx
#np.savetxt(out_file_tau,tau_OI_arr)
#print "Optical Depth saved ..."

tau_OI_arr = np.loadtxt(out_file_tau)
flux_data = np.exp(-tau_OI_arr)
kounter   = 0

for indx in xrange(flux_data.shape[1]):
    peak_zs,EW_data = equiv_widths(flux_data[:,indx])
    if len(EW_data) != 0:
        if kounter == 0:
            EW_arr = np.array(EW_data)
            kounter += 1
        EW_arr = np.concatenate([EW_arr,np.array(EW_data)])

EW_arr_prakash = compute_equivalent_width_prakash(flux_data,zs)

EW_cdf         = compute_EW_cdf(EW_arr,n_arr)
EW_cdf_prakash = compute_EW_cdf(EW_arr_prakash,n_arr)

fig,ax = plt.subplots(1)
fig.set_figwidth(12.0)
fig.set_figheight(11.0)
ax.plot(EW_cdf[:,0],EW_cdf[:,1],label="Daniel's Code")
ax.plot(EW_cdf_prakash[:,0],EW_cdf_prakash[:,1],color="r",label="Prakash's Code")
plot_observation_data(ax,data="2011")
ax.set_xlabel(r"${\rm EW \; (\AA)}$")
ax.set_ylabel(r"$dn/dX$")
ax.set_xlim(0.0,0.5)
ax.set_ylim(0.0,0.4)
ax.legend(loc="upper right")
plt.savefig("EW_Comparison.pdf",dpi=200)
exit()
print EW_arr.shape
exit()

# Calculate the absorption path length as defined in [K2014]
def abs_length(z):
    return 2.0 * np.sqrt(Om_La + Om_m0 * (1.0 + z) ** 3.0) / (3.0 * Om_m0)

# Cumulative dN/dX data
def cumulative_EW(num_sightlines, ssOnly, incomplete = False, cumulative = True, tracking = None, observed = False):
    widths = np.array([])
    for n in range(0, num_sightlines):
        pzs, ews = equiv_widths(n, ssOnly, tracking)
        widths = np.append(ews, widths)
    DeX = (abs_length(zs[count - 1]) - abs_length(zs[0])) * num_sightlines
    counts, bin_edges = np.histogram(widths, num_bins)
    rates = counts / DeX
    midpoints = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(0, num_bins)])
    if incomplete:
        data = np.loadtxt("completeness_data.txt", skiprows = 1)
        rates *= np.interp(midpoints, data[:, 2], data[:, 0], left = 0.0) / 100.0
    dN_by_dXs = np.flip(np.cumsum(np.flip(rates)))
    if not cumulative:
        dN_by_dXs = rates
    return midpoints, dN_by_dXs

# Cumulative dN/dX for 2019 data input
def cumulative_EW_2019(num_sightlines, incomplete = False, observed = None, fullwidth = False):
    # TODO use full set of data
    assert(float(z_mid) < 6.6 and float(z_mid) > 5.6)
    widths = np.array([])
    if observed is None:
        for n in range(0, num_sightlines):
            pzs, ews = equiv_widths(n, False)
            widths = np.append(ews, widths)
    else:
        inp = np.loadtxt('raw_2019_data.txt', skiprows = 1)
        if fullwidth:
            for i in range(0, len(inp[:, 0])):
                if inp[i, 0] <= 6.5 and inp[i, 0] >= 5.7:
                    widths = np.append(inp[i, 1], widths)
        else:
            for i in range(0, len(inp[:, 0])):
                if inp[i, 0] <= zs[count - 1] and inp[i, 0] >= zs[0]:
                    widths = np.append(inp[i, 1], widths)
    DeX = (abs_length(zs[count - 1]) - abs_length(zs[0])) * num_sightlines
    if observed is not None:
        if fullwidth:
            DeX = 66.3
        else:
            DeX /= (abs_length(6.5) - abs_length(5.7)) * num_sightlines / 66.3
    counts, bin_edges = np.histogram(widths, num_bins)
    rates = counts / DeX
    midpoints = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(0, num_bins)])
    if incomplete:
        data = np.loadtxt('5.7-6.5 2019 completeness.txt')
        rates *= np.interp(midpoints, np.float_power(10.0, data[:, 0]), data[:, 1], left = 0.0)
    dN_by_dXs = np.flip(np.cumsum(np.flip(rates)))
    return midpoints, dN_by_dXs

# Imperative function to exaggerate the spectrum
def exaggerate():
    DeHss *= 50
    Tss *= 100

# Imperative function to rescale the UV background
def rescale_Ga_12(f):
    global Ga_12
    Ga_12 *= f

def rescale_Z(f):
    global Z_solar_oxygen
    Z_solar_oxygen *= f

def rescale_cap_Z(f):
    global cap_Z_frac
    cap_Z_frac *= f
