import numpy as np
import matplotlib
import panel as pn
pn.extension()

matplotlib.use('agg')

import matplotlib.pyplot as plt

Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)

fig0, ax0 = plt.subplots()
strm = ax0.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)

mpl_pane = pn.pane.Matplotlib(fig0, dpi=144)
mpl_pane

strm.lines.set_cmap(plt.cm.viridis)

mpl_pane.param.trigger('object')
