import matplotlib.pyplot as plt
import matplotlib.lines as ml
import matplotlib
import sys

from los_fields import *

matplotlib.rcParams["text.usetex"] = True
plt.style.use("custom_plot_style.py")

# Number of sightlines to average over
num_sightlines = 10

# Number of bins
num_bins = 128

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

# dN/dz equivalent width plot
def plot2():
	plt.title('Cumulative incidence rate of $' + oiLabel + '$ absorbers at $z = 5.6$')
	widths = np.array([])
	for n in range(0, num_sightlines):
		widths = np.append(equiv_widths(n, False), widths)
		print(n)
	plt.hist(widths, num_bins, density = True, histtype = "step", cumulative = -1)
	plt.xlabel('$' + oiLabel + '$ equivalent width')
	plt.ylabel('$\\frac{dN}{dX}$')
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

# Plot positions of peaks
def test5(n):
	flux_data = fluxes(n, False)
	plt.plot(zs, flux_data)
	plt.title("Trough detection in an Oxygen I spectrum")
	plt.ylim([0.0, 1.1])
	plt.xlabel("$z$")
	plt.ylabel(fluxLabel)
	mins = extrema(n, False, True)
	maxes = extrema(n, False, False)
	plt.scatter(zs[mins], flux_data[mins], c = 'r')
	plt.scatter(zs[maxes], flux_data[maxes], c = 'g')
	for i in mins:
		prev, next = trough_boundaries(i, mins, maxes)
		plt.plot(zs[prev], flux_data[prev], color = 'k', marker = '<', markersize = 4.0)
		plt.plot(zs[next], flux_data[next], color = 'k', marker = '>', markersize = 4.0)
	plt.show()

def test6(n):
	fig, axes = plt.subplots(5, 1, sharex = True)
	axes[0].set_title(f"Oxygen properties for sightline {n + 1}, using $\\Gamma_{{12}}={Ga_12}$ and $z={z_mid}$")
	axes[0].semilogy(zs, nOIs(n) / nHIs(n) * fHIss[:, n])
	axes[0].set_ylabel('$n_{' + oiLabel + '}/n_H$', fontsize = 18, rotation = "horizontal")
	axes[1].semilogy(zs, DeHss[:, n])
	axes[1].set_ylabel('$\Delta$', fontsize = 18, rotation = "horizontal")
	axes[2].plot(zs, vss[:, n] / 1.0e3)
	axes[2].set_ylabel('$v / \mathrm{kms}^{-1}$', fontsize = 18, rotation = "horizontal")
	axes[3].semilogy(zs, Tss[:, n])
	axes[3].set_ylabel('$T/K$', fontsize = 18, rotation = "horizontal")
	axes[4].semilogy(zs, opticalDepths(n, False))
	axes[4].set_xlabel("$z$")
	axes[4].set_ylabel('$\\tau_{' + oiLabel + '}$', fontsize = 18, rotation = "horizontal")
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

# Check equivalent widths
def check5(n):
	widths = equiv_widths(n, False)
	for w in widths:
		print(w)

def output1():
	ns = [2369, 3231, 251, 2188, 2514]
	for n in ns:
		np.savetxt(f"data {n}.csv", (zs, vss[:, n], Tss[:, n], DeHss[:, n], nOIs(n) / nHIs(n) * fHIss[:, n], opticalDepths(n, False)), delimiter = ',')

# Main
n = 0
if len(sys.argv) > 0:
	n = int(sys.argv[1]) - 1
test5(n)