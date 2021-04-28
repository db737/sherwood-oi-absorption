import matplotlib.pyplot as plt
import matplotlib.lines as ml
import matplotlib
import scipy.integrate as si
import sys
import numpy.random as nrand

from los_fields import *

matplotlib.rcParams["text.usetex"] = True
plt.style.use("custom_plot_style.py")

hiLabel = 'HI'
oiLabel = 'OI'
depthLabel = '$\\tau_{' + oiLabel + "}$"
fluxLabel = "$F=e^{-" + depthLabel[1 : len(depthLabel) - 1] + "}$"

# Optical depth and flux
def plot1(n):
	plt.title(f"Flux for sightline {n + 1}")
	plt.plot(zs, fluxes(n, False, False))
	plt.ylim([0.0, 1.1])
	plt.xlabel("$z$")
	plt.ylabel(fluxLabel)
	plt.show()

# dN/dz equivalent width plot
def plot2(num_sightlines):
	plt.title('Cumulative incidence rate of $' + oiLabel + '$ absorbers at $z = 5.6$')
	midpoints, dN_by_dXs = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoints, dN_by_dXs, 'k', where = 'mid')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	plt.show()

# Compare the effect of the self-shielding prescription
def plot3(num_sightlines):
	plt.title('Cumulative incidence rate of $' + oiLabel + '$ absorbers at $z = 5.6$')
	midpoint1s, dN_by_dX1s = cumulative_EW(num_sightlines, True, incomplete = True)
	midpoint2s, dN_by_dX2s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoint1s, dN_by_dX1s, 'k', where = 'mid')
	plt.step(midpoint2s, dN_by_dX2s, 'b', linestyle = '--', where = 'mid')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	ss = ml.Line2D([], [], color = 'k', label = 'OI only in SS regions')
	notss = ml.Line2D([], [], color = 'b', ls = '--', label = 'OI everywhere')
	plt.legend(handles = [ss, notss])
	plt.show()

# Vary metallicity
def plot4(num_sightlines):
	rescale_Z(1.3)
	midpoint1s, dN_by_dX1s = cumulative_EW(num_sightlines, False, incomplete = True, tracking = 0.4)
	plt.step(midpoint1s, dN_by_dX1s, 'r', where = 'mid')
	l1 = ml.Line2D([], [], color = 'r', label = f"$Z/Z_0$=1.3")
	rescale_Z(0.5 / 1.3)
	midpoint2s, dN_by_dX2s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoint2s, dN_by_dX2s, 'g', where = 'mid')
	l2 = ml.Line2D([], [], color = 'g', label = f"$Z/Z_0$=0.5")
	rescale_Z(6.0)
	midpoint3s, dN_by_dX3s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoint3s, dN_by_dX3s, 'b', where = 'mid')
	l3 = ml.Line2D([], [], color = 'b', label = f"$Z/Z_0$=3.0")
	inp = np.loadtxt("add_data_mid.txt")
	plt.plot(inp[:, 0], inp[:, 1], 'k', linestyle = '--')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	be = ml.Line2D([], [], color = 'k', ls = '--', label = 'Becker et al. 2011')
	plt.legend(handles = [l1, l2, l3, be])
	plt.show()

# Vary Gamma
def plot5(num_sightlines):
	midpoint1s, dN_by_dX1s = cumulative_EW(num_sightlines, False, incomplete = True)
	midpoint1s = np.append(0.0, midpoint1s)
	dN_by_dX1s = np.append(dN_by_dX1s[0], dN_by_dX1s)
	plt.step(midpoint1s, dN_by_dX1s, 'r', where = 'mid')
	l1 = ml.Line2D([], [], color = 'r', label = f"$\\Gamma_{{12}}={0.16}$")
	rescale_Ga_12(0.25)
	midpoint2s, dN_by_dX2s = cumulative_EW(num_sightlines, False, incomplete = True)
	midpoint2s = np.append(0.0, midpoint2s)
	dN_by_dX2s = np.append(dN_by_dX2s[0], dN_by_dX2s)
	plt.step(midpoint2s, dN_by_dX2s, 'g', where = 'mid')
	l2 = ml.Line2D([], [], color = 'g', label = f"$\\Gamma_{{12}}={0.04}$")
	rescale_Ga_12(16.0)
	midpoint3s, dN_by_dX3s = cumulative_EW(num_sightlines, False, incomplete = True)
	midpoint3s = np.append(0.0, midpoint3s)
	dN_by_dX3s = np.append(dN_by_dX3s[0], dN_by_dX3s)
	plt.step(midpoint3s, dN_by_dX3s, 'b', where = 'mid')
	l3 = ml.Line2D([], [], color = 'b', label = f"$\\Gamma_{{12}}={0.64}$")
	rescale_Ga_12(0.125)
	midpoint4s, dN_by_dX4s = cumulative_EW(num_sightlines, False, incomplete = True)
	midpoint4s = np.append(0.0, midpoint4s)
	dN_by_dX4s = np.append(dN_by_dX4s[0], dN_by_dX4s)
	plt.step(midpoint4s, dN_by_dX4s, 'k', where = 'mid')
	l4 = ml.Line2D([], [], color = 'k', label = f"$\\Gamma_{{12}}={0.08}$")
	inp = np.loadtxt("add_data_mid.txt")
	plt.plot(inp[:, 0], inp[:, 1], 'k', linestyle = '--')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	be = ml.Line2D([], [], color = 'k', ls = '--', label = 'Becker et al. 2011')
	plt.legend(handles = [l1, l2, l3, l4, be])
	plt.show()

# Completeness comparison
def plot6(num_sightlines):
	midpoint1s, dN_by_dX1s = cumulative_EW(num_sightlines, False)
	inp = np.loadtxt("add_data.txt")
	midpoint2, dN_by_dX2s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(inp[:, 0], inp[:, 1], 'k', where = 'mid')
	plt.step(midpoint1s, dN_by_dX1s, 'b', where = 'mid')
	plt.step(midpoint1s, dN_by_dX2s, 'b', linestyle = '--', where = 'mid')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	plt.xscale('log')
	plt.yscale('log')
	complete = ml.Line2D([], [], color = 'b', label = 'Raw computed values')
	incomplete = ml.Line2D([], [], color = 'b', ls = '--', label = 'Scaled for expected completeness')
	be = ml.Line2D([], [], color = 'k', label = 'Becker et al. 2011')
	plt.legend(handles = [complete, incomplete, be])
	plt.show()

# Compare hydrogen and oxygen
def plot7(n):
	fig, axes = plt.subplots(2, 1, sharex = True)
	skipedge = 20
	axes[0].plot(zs[skipedge : count - skipedge], fluxes(n, True, False)[skipedge : count - skipedge], linewidth = 1.0)
	axes[0].legend(['Hydrogen'], loc = 'upper right')
	axes[0].set_ylim([-0.1, 1.1])
	axes[0].set_ylabel('Flux')
	axes[1].plot(zs[skipedge : count - skipedge], fluxes(n, False, False)[skipedge : count - skipedge], 'r', linewidth = 1.0)
	ox = ml.Line2D([], [], color = 'r', label = 'Oxygen')
	axes[1].legend(handles = [ox], loc = 'lower right')
	axes[1].set_ylim([-0.1, 1.1])
	axes[1].set_ylabel('Flux')
	axes[1].set_xlabel('$z$')
	plt.subplots_adjust(hspace = 0)
	fig.align_ylabels()
	plt.show()

# Show completeness
def plot8(n):
	data = np.loadtxt("completeness_data.txt", skiprows = 1)
	plt.plot(data[:, 2], data[:, 0])
	plt.xlabel('Equivalent width / \AA')
	plt.ylabel('Completeness / \%')
	plt.xscale('log')
	plt.show()

# Compare patchy vs. homogeneous
def plot9(num_sightlines):
	midpoint1s, dN_by_dX1s = cumulative_EW(num_sightlines, False, incomplete = True, tracking = 0.4)
	plt.step(midpoint1s, dN_by_dX1s, 'r', where = 'mid')
	l1 = ml.Line2D([], [], color = 'r', label = f"Homogeneous")
	enable_bubbles()
	midpoint2s, dN_by_dX2s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoint2s, dN_by_dX2s, 'g', where = 'mid')
	l2 = ml.Line2D([], [], color = 'g', label = f"Patchy")
	inp = np.loadtxt("add_data_mid.txt")
	plt.plot(inp[:, 0], inp[:, 1], 'k', linestyle = '--')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	be = ml.Line2D([], [], color = 'k', ls = '--', label = 'Becker et al. 2011')
	plt.legend(handles = [l1, l2, be])
	plt.show()

# 2019 data
def plot10(num_sightlines):
	midpoint1s, dN_by_dX1s = cumulative_EW_2019(num_sightlines, incomplete = True)
	plt.step(midpoint1s, dN_by_dX1s, 'b', where = 'mid')
	l = ml.Line2D([], [], color = 'b', label = "Computed")
	inp = np.loadtxt("add_data_mid.txt")
	plt.plot(inp[:, 0], inp[:, 1], 'k-.')
	be11 = ml.Line2D([], [], color = 'k', ls = '-.', label = 'Becker et al. 2011')
	midpoint3s, dN_by_dX3s = cumulative_EW_2019(num_sightlines, observed = 'mid', fullwidth = True)
	plt.step(midpoint3s, dN_by_dX3s, 'k', ls = '--', where = 'mid')
	be19 = ml.Line2D([], [], color = 'k', ls = '--', label = 'Becker et al. 2019')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	plt.title(f"$z={z_mid}$")
	plt.legend(handles = [l, be11, be19])
	plt.show()

# Compare completeness data sets
def plot11(n):
	new6ss = np.loadtxt('5.7-6.5 2019 completeness.txt')
	new5ss = np.loadtxt('4.9-5.7 2019 completeness.txt')
	new4ss = np.loadtxt('4.1-4.9 2019 completeness.txt')
	new3ss = np.loadtxt('3.2-4.1 2019 completeness.txt')
	oldss = np.loadtxt("completeness_data.txt", skiprows = 1)
	newfn = lambda xss, c: plt.plot(np.float_power(10.0, xss[:, 0]), xss[:, 1], c)
	newfn(new6ss, 'k')
	newfn(new5ss, 'r')
	newfn(new4ss, 'g')
	newfn(new3ss, 'b')
	l6 = ml.Line2D([], [], color = 'k', label = 'z = 5.7-6.5')
	l5 = ml.Line2D([], [], color = 'r', label = 'z = 4.9-5.7')
	l4 = ml.Line2D([], [], color = 'g', label = 'z = 4.1-4.9')
	l3 = ml.Line2D([], [], color = 'b', label = 'z = 3.2-4.1')
	lold = ml.Line2D([], [], color = 'k', ls = '--', label = 'Old data')
	endpt = len(oldss[:, 0]) - 3
	plt.plot(oldss[: endpt, 2], oldss[: endpt, 0] / 100.0, 'k--')
	plt.legend(handles = [l6, l5, l4, l3, lold])
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('Completeness')
	plt.show()

# Compare wide vs small range
def plot12(num_sightlines):
	midpoint1s, dN_by_dX1s = cumulative_EW_2019(num_sightlines, observed = 'mid', fullwidth = True)
	plt.step(midpoint1s, dN_by_dX1s, 'b', where = 'mid')
	bew = ml.Line2D([], [], color = 'b', label = 'Wide range')
	midpoint1s, dN_by_dX1s = cumulative_EW_2019(num_sightlines, observed = 'mid')
	plt.step(midpoint1s, dN_by_dX1s, 'k', where = 'mid')
	ben = ml.Line2D([], [], color = 'k', label = 'Narrow range')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	plt.legend(handles = [bew, ben])
	plt.show()
	
# Compare with ATON
def plot13(num_sightlines):
	Ga_12_ATON = 8.109693034979193e-2
	midpoint1s, dN_by_dX1s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoint1s, dN_by_dX1s, 'r', where = 'mid')
	l1 = ml.Line2D([], [], color = 'r', label = "Sherwood, $\\Gamma_{{12}}=0.16$, $Z= Z_0$")
	rescale_Ga_12(Ga_12_ATON / 0.16)
	midpoint2s, dN_by_dX2s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoint2s, dN_by_dX2s, 'g', where = 'mid')
	l2 = ml.Line2D([], [], color = 'g', label = "Sherwood, $\\Gamma_{{12}}=0.081$, $Z= Z_0$")
	enable_bubbles()
	midpoint3s, dN_by_dX3s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoint3s, dN_by_dX3s, 'b', where = 'mid')
	l3 = ml.Line2D([], [], color = 'b', label = f"Patchy, Native $\\Gamma_{{12}}$, $Z= Z_0$")
	inp = np.loadtxt("add_data_mid.txt")
	plt.plot(inp[:, 0], inp[:, 1], 'k', linestyle = '--')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	plt.ylim([0.0, 0.3])
	be = ml.Line2D([], [], color = 'k', ls = '--', label = 'Becker et al. 2011')
	plt.legend(handles = [l1, l2, l3, be])
	plt.show()

# Show degeneracy
def plot14(num_sightlines):
	rescale_Z(1.3)
	midpoint1s, dN_by_dX1s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoint1s, dN_by_dX1s, 'r', where = 'mid')
	l1 = ml.Line2D([], [], color = 'r', label = f"Sherwood, $\\Gamma_{{12}}=0.16$, $Z= 1.3 Z_0$")
	rescale_Z(1.0 / 1.3)
	rescale_Ga_12(0.75)
	midpoint2s, dN_by_dX2s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoint2s, dN_by_dX2s, 'g', where = 'mid')
	l2 = ml.Line2D([], [], color = 'g', label = f"Sherwood, $\\Gamma_{{12}}=0.12$, $Z= 1.0 Z_0$")
	inp = np.loadtxt("add_data_mid.txt")
	plt.plot(inp[:, 0], inp[:, 1], 'k', linestyle = '--')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	plt.ylim([0.0, 0.3])
	be = ml.Line2D([], [], color = 'k', ls = '--', label = 'Becker et al. 2011')
	plt.legend(handles = [l1, l2, be])
	plt.show()

# Basic comparison
def plot15(num_sightlines):
	midpoint1s, dN_by_dX1s = cumulative_EW(num_sightlines, False, incomplete = True)
	plt.step(midpoint1s, dN_by_dX1s, 'b', where = 'mid')
	l1 = ml.Line2D([], [], color = 'b', label = f"Sherwood, $\\Gamma_{{12}}=0.16$, $Z=Z_0$")
	inp = np.loadtxt("add_data_mid.txt")
	plt.plot(inp[:, 0], inp[:, 1], 'k', linestyle = '--')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	plt.ylim([0.0, 0.3])
	be = ml.Line2D([], [], color = 'k', ls = '--', label = 'Becker et al. 2011')
	plt.legend(handles = [l1, be])
	plt.show()
	
# Vary n
def plot16(num_sightlines):
	rescale_Z(1.3)
	midpoint1s, dN_by_dX1s = cumulative_EW(num_sightlines, False, incomplete = True)
	midpoint1s = np.append(0.0, midpoint1s)
	dN_by_dX1s = np.append(dN_by_dX1s[0], dN_by_dX1s)
	plt.step(midpoint1s, dN_by_dX1s, 'r', where = 'mid')
	l1 = ml.Line2D([], [], color = 'r', label = f"$n = 1.3$, $Z = 1.3 Z_0$")
	set_n(0.5)
	rescale_Z(1.5)
	midpoint2s, dN_by_dX2s = cumulative_EW(num_sightlines, False, incomplete = True)
	midpoint2s = np.append(0.0, midpoint2s)
	dN_by_dX2s = np.append(dN_by_dX2s[0], dN_by_dX2s)
	plt.step(midpoint2s, dN_by_dX2s, 'g', where = 'mid')
	l2 = ml.Line2D([], [], color = 'g', label = f"$n = 0.5$, $Z = 2.0 Z_0$")
	rescale_Z(1.0 / 1.5)
	set_n(2.0)
	midpoint3s, dN_by_dX3s = cumulative_EW(num_sightlines, False, incomplete = True)
	midpoint3s = np.append(0.0, midpoint3s)
	dN_by_dX3s = np.append(dN_by_dX3s[0], dN_by_dX3s)
	plt.step(midpoint3s, dN_by_dX3s, 'b', where = 'mid')
	l3 = ml.Line2D([], [], color = 'b', label = f"$n = 2.0$, $Z = 1.3 Z_0$")
	set_n(1.5)
	midpoint4s, dN_by_dX4s = cumulative_EW(num_sightlines, False, incomplete = True)
	midpoint4s = np.append(0.0, midpoint4s)
	dN_by_dX4s = np.append(dN_by_dX4s[0], dN_by_dX4s)
	plt.step(midpoint4s, dN_by_dX4s, 'k', where = 'mid')
	l4 = ml.Line2D([], [], color = 'k', label = f"$n = 1.5$, $Z = 1.3 Z_0$")
	inp = np.loadtxt("add_data_mid.txt")
	plt.plot(inp[:, 0], inp[:, 1], 'k', linestyle = '--')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	be = ml.Line2D([], [], color = 'k', ls = '--', label = 'Becker et al. 2011')
	plt.legend(handles = [l1, l2, l3, l4, be])
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
	#plt.title(f"Comparison of simulation output and computed neutral hydrogen optical depths for sightline {n + 1}")
	rem = 60
	plt.plot(zs[rem : count - rem], ta_HIss[rem : count - rem, n], "k")
	plt.plot(zs[rem : count - rem], opticalDepths(n, True, False)[rem : count - rem], "b--")
	measured = ml.Line2D([], [], color = "k", label = "From simulation outputs")
	computed = ml.Line2D([], [], color = "b", ls = "--", label = "Newly computed")
	plt.legend(handles = [measured, computed])
	plt.show()

# Metallicity, self-shielding etc.
def test4(n):
	fig, axes = plt.subplots(5, 1, sharex = True)
	axes[0].set_title(f"Oxygen properties for sightline {n + 1}, using $\\Gamma_{{12}}={Ga_12}$ and $z={z_mid}$")
	axes[0].semilogy(zs, Zs(n) / Z_solar_oxygen)
	axes[0].set_ylabel('$Z/Z_{\\odot}$')
	axes[1].semilogy(zs, cutoffsSS(n))
	axes[1].set_ylabel('$\Delta_{ss}$')
	axes[2].semilogy(zs, DeHss[:, n])
	axes[2].set_ylabel('$\Delta_H$')
	axes[3].semilogy(zs, nOIs(n, False) / 1.0e6)
	axes[3].set_ylabel("$n_{" + oiLabel + '} / \mathrm{cm^{-3}}$')
	axes[4].plot(zs, fluxes(n, False, False))
	axes[4].set_xlabel("$z$")
	axes[4].set_ylabel("$F$")
	axes[4].set_ylim([0.0, 1.1])
	axes[4].set_yticks([0.0, 0.5, 1.0])
	plt.subplots_adjust(hspace = 0)
	fig.align_ylabels()
	plt.show()

# Plot positions of peaks
def test5(n):
	flux_data = fluxes(n, False, False)
	plt.plot(zs, flux_data)
	plt.title("Trough detection in an Oxygen I spectrum")
	plt.ylim([0.0, 1.1])
	plt.xlabel("$z$")
	plt.ylabel(fluxLabel)
	mins = extrema(flux_data, True)
	maxes = extrema(flux_data, False)
	plt.scatter(zs[mins], flux_data[mins], c = 'r')
	plt.scatter(zs[maxes], flux_data[maxes], c = 'g')
	for i in mins:
		prev, next = trough_boundaries(i, mins, maxes)
		plt.plot(zs[prev], flux_data[prev], color = 'k', marker = '<', markersize = 4.0)
		plt.plot(zs[next], flux_data[next], color = 'k', marker = '>', markersize = 4.0)
	plt.show()

def test6(n):
	fig, axes = plt.subplots(6, 1, sharex = True)
	global Ga_12
	Ga_12 = 0.16
	axes[0].set_title(f"Oxygen properties for sightline {n + 1}, using $\\Gamma_{{12}}={Ga_12}$ and $z={z_mid}$")
	axes[0].semilogy(zs, nOIs(n, False) / nHIs(n) * fHIss[:, n])
	axes[0].set_ylabel('$n_{' + oiLabel + '}/n_H$', fontsize = 18, rotation = "horizontal")
	axes[1].semilogy(zs, DeHss[:, n])
	axes[1].set_ylabel('$\Delta$', fontsize = 18, rotation = "horizontal")
	axes[2].plot(zs, vss[:, n] / 1.0e3)
	axes[2].set_ylabel('$v / \mathrm{kms}^{-1}$', fontsize = 18, rotation = "horizontal")
	axes[3].semilogy(zs, Tss[:, n])
	axes[3].set_ylabel('$T/K$', fontsize = 18, rotation = "horizontal")
	axes[4].semilogy(zs, Zs(n))
	axes[4].set_ylabel('$Z$', fontsize = 18, rotation = "horizontal")
	axes[5].semilogy(zs, opticalDepths(n, False, False))
	axes[5].set_xlabel("$z$")
	axes[5].set_ylabel('$\\tau_{' + oiLabel + '}$', fontsize = 18, rotation = "horizontal")
	plt.subplots_adjust(hspace = 0)
	fig.align_ylabels()
	plt.show()

# Test effect of SS
def test8(n):
	plt.semilogy(zs, opticalDepths(n, False, None), 'r')
	plt.semilogy(zs, opticalDepths(n, False, False), 'g')
	plt.semilogy(zs, opticalDepths(n, False, True), 'b--')
	plt.title(f"Effect of self-shielding at $z={z_mid}$ for sightline {n + 1}")
	red = ml.Line2D([], [], color = 'r', label = 'No self-shielding')
	green = ml.Line2D([], [], color = 'g', label = 'SS with OI also present elsewhere')
	blue = ml.Line2D([], [], color = 'b', ls = '--', label = 'OI only present in SS regions')
	plt.legend(handles = [red, green, blue])
	plt.show()

# Plot Becker et al. 2019 data
def test9(n):
	inp = np.loadtxt("raw_2019_data.txt", skiprows = 1)
	plt.errorbar(inp[:, 0], inp[:, 1], yerr = inp[:, 2], ls = '', capsize = 5, color = 'k')
	plt.xlabel("z")
	plt.ylabel("O I EW")
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
	print("dz/dx: {}".format(dz_by_dX(zs)[0]))
	print("alpha (HI): {}".format(als(n, True)[0]))
	print("alpha (OI): {}".format(als(n, False)[0]))
	print("V arg 2: {}".format(vArg2s(n, 3.0, m_HI)[0]))
	print("V(..., ...): {}".format(voigt(als(n, True), vArg2s(n, 3.0, m_HI))[0]))
	print("integrand: {}".format(integrand1s(n, 3.0, True, False)[0]))

# Compare hydrogen outputs directly
def check3(n):
	print("from data: {}".format(ta_HIss[0, n]))
	print("computed: {}".format(opticalDepth(n, zs[0], True, False)))

# Check oxygen stuff
def check4(n):
	print("Zs: {}".format(Zs(n)))
	print("cutoffs: {}".format(cutoffsSS(n)))

# Check equivalent widths
def check5(n):
	pzs, widths = equiv_widths(n, False)
	for w in widths:
		print(w)

# Check optical depth
def check6(n):
	plt.plot(zs, opticalDepths(n, False, False))
	plt.show()

# Check consistent with Prakash's changes
def check7(num_sightlines):
	midpoint1s, dN_by_dX1s = cumulative_EW(num_sightlines, False, incomplete = False)
	plt.step(midpoint1s, dN_by_dX1s, 'r', linestyle = '--', where = 'mid')
	l1 = ml.Line2D([], [], color = 'r', linestyle = '--', label = f"Prakash's version")
	old_mode(True)
	midpoint2s, dN_by_dX2s = cumulative_EW(num_sightlines, False, incomplete = False)
	plt.step(midpoint2s, dN_by_dX2s, 'g', where = 'mid')
	l2 = ml.Line2D([], [], color = 'g', label = f"Old version")
	inp = np.loadtxt("add_data_mid.txt")
	plt.plot(inp[:, 0], inp[:, 1], 'k', linestyle = '--')
	plt.xlabel('$' + oiLabel + '$ equivalent width / \AA')
	plt.ylabel('$\\frac{dN}{dX}$')
	plt.ylim([0.0, 0.4])
	be = ml.Line2D([], [], color = 'k', ls = '--', label = 'Becker et al. 2011')
	plt.legend(handles = [l1, l2, be])
	plt.show()

def output1():
	ns = [2369, 3231, 251, 2188, 2514]
	for n in ns:
		np.savetxt(f"data {n}.csv", (zs, vss[:, n], Tss[:, n], DeHss[:, n], nOIs(n, False) / nHIs(n) * fHIss[:, n], opticalDepths(n, False)), delimiter = ',')

def output2(n):
	DeX = abs_length(zs[count - 1]) - abs_length(zs[0])
	print(f"Box size: {spec_obj.box}")
	print(f"Box path length: {DeX}")
	pzs, widths = equiv_widths(n, False)
	print(f"Peak positions: {pzs}")
	print(f"EWs: {widths}")
	np.savetxt(f"flux {n}.txt", (zs, fluxes(n, False, False)))

# Find which arbsorbers are in which z bin
def output3(n):
	zms = [6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7]
	inp = np.loadtxt("raw_2019_data.txt", skiprows = 1)[:, 0]
	for zm in zms:
		zs = redshift_array(zm)
		within = lambda x: x >= zs[0] and x <= zs[count - 1]
		print("{} ({} - {}): {}".format(zm, zs[0], zs[count - 1], len(list(filter(within, inp)))))

# Gamma from ATON
def output4(n):
	Ga12ss = np.transpose(spec_obj_patchy.Gamma_HI)
	print("Mean: {}\nMedian: {}".format(np.mean(Ga12ss), np.median(Ga12ss)))
	plt.plot(zs, Ga12ss[:, n] * 1.0e12)
	plt.xlabel("$z$")
	plt.ylabel("$\\Gamma_{12}$")
	plt.show()

def input1():
	tass = np.loadtxt('../../Optical_Depth.txt')
	ns = [2188, 2369, 2514, 251, 3231]
	for i in range(0, 5):
		plt.semilogy(zs, tass[:, i], 'k')
		plt.semilogy(zs, opticalDepths(ns[i], False, False), 'b--')
		pr = ml.Line2D([], [], color = "k", label = "Prakash's calculation")
		da = ml.Line2D([], [], color = "b", ls = "--", label = "Daniel's calculation")
		plt.legend(handles = [pr, da])
		plt.show()
		plt.semilogy(zs, tass[:, i] / opticalDepths(ns[i], False, False))
		plt.show()

# Plot positions and widths of peaks
def example1(n):
	exaggerate()
	flux_data = fluxes(n, False, True)
	plt.plot(zs, flux_data)
	#plt.title("Extrema detection and EW calculation in an exaggerated spectrum")
	plt.ylim([0.0, 1.1])
	plt.xlabel("$z$")
	plt.ylabel(fluxLabel)
	mins = extrema(flux_data, True)
	maxes = extrema(flux_data, False)
	plt.scatter(zs[mins], flux_data[mins], c = 'r')
	plt.scatter(zs[maxes], flux_data[maxes], c = 'g')
	pzs, ews = equiv_widths(n, True)
	ews *= nu_12_OI * 1.0e-10 / c
	plt.errorbar(zs[mins], flux_data[mins], xerr = ews / 2.0, fmt = 'none', capsize = 10.0)
	plt.show()

# Fake trough example
def example2(n):
	xs = np.linspace(-1.0, 1.0, 200)
	ys = 1.0 - 0.3 * np.exp(-0.5 * (xs / 0.2) ** 2.0)
	noise = 0.01 * nrand.normal(0.0, 1.0, 200)
	plt.plot(xs, ys + noise)
	plt.ylim([0.0, 1.1])
	plt.xticks([])
	plt.xlabel('$z$')
	plt.ylabel('Flux')
	plt.show()
	
# Fake trough EW example
def example3(n):
	xs = np.linspace(-1.0, 1.0, 200)
	sigma, scale = 0.2, 0.3
	y1s = 1.0 - scale * np.exp(-0.5 * (xs / sigma) ** 2.0)
	hwidth = scale * sigma * np.sqrt(2.0 * pi) / 2.0
	y2s = 1.0 + np.heaviside(xs - hwidth, 1.0) - np.heaviside(xs + hwidth, 1.0)
	plt.ylim([0.0, 1.1])
	plt.xticks([])
	plt.xlabel('$z$')
	plt.ylabel('Flux')
	plt.plot(xs, y1s, 'k')
	plt.plot(xs, y2s, 'b')
	plt.show()

# Main
n = int(sys.argv[1]) - 1
example1(n)