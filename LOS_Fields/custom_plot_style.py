# This is Dan Allan's setting of Liana Vaccari's plotting style preferences.
#
# modified by Prakash Gaikwad
#
# Updated December 2016
#
# Invoke in Python matplotlib like so:
#
# import matplotlib.pyplot as plt
#
# plt.style.use('/path/to/custom_plot_style.py')
#
# This feature is only available in matplotlib version 1.4 or higher.
 
# Plot area
figure.figsize: 10,8
#figure.facecolor: white
#figure.edgecolor: white
axes.facecolor : white
axes.grid: False
 
# Axes and ticks
axes.linewidth: 2.5
axes.edgecolor: black
xtick.direction: in
ytick.direction: in

xtick.major.size: 12
xtick.major.width: 2.5
xtick.minor.size: 8
xtick.minor.width: 2.5

ytick.major.size: 14
ytick.major.width: 2.5
ytick.minor.size: 8
ytick.minor.width: 2.5
 
# Space xlabels lower than default.
xtick.major.pad: 7
xtick.minor.pad: 7
 
# Font sizes for various elements.
font.size: 35
axes.titlesize: 25
axes.labelsize: 25
xtick.labelsize: 20
ytick.labelsize: 20
legend.fontsize: 20
 
# Use sans-serif LaTeX for all text.
font.family: serif
font.sans-serif: Georgia
 
image.cmap: jet
 
# Plot elements
lines.linewidth:  2
lines.markersize: 8
 
# savefig options (avoid clipping at margins)
savefig.bbox : tight
savefig.pad_inches : 0.01
savefig.dpi       : 200

### Legend
legend.fancybox      : True  # if True, use a rounded box for the
                               # legend, else a rectangle 
                               
                               
#axes.color_cycle    : 348ABD, 7A68A6, A60628, 467821, CF4457, 188487, E24A33
                      # E24A33 : orange
                      # 7A68A6 : purple
                      # 348ABD : blue
                      # 188487 : turquoise
                      # A60628 : red
                      # CF4457 : pink
                      # 467821 : green                               
