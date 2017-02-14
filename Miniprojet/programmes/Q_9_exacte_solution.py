# -*- coding: utf-8 -*-

# Load necessary packages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

########################################################
# parameters of the problem
L = 1           # interval between x = [0, L], y = [0, L]
dh = 0.01         # cell size in space for both x and y


def u_exact(x, y):
    return x * (x - L) * y * (y - L)

########################################################
# Initialisation


N = int(L / dh - 1)   # N = number of intervals
x_range = np.linspace(0, L, N + 2)
U = np.ndarray(shape=(N + 2, N + 2))

# Set specific f values
for i in range(N + 2):
    for j in range(N + 2):
        U[i][j] = u_exact(x_range[i], x_range[j])


fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(0, L + dh, dh)
Y = np.arange(0, L + dh, dh)
X, Y = np.meshgrid(X, Y)
Z = U.copy()
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

ax.set_title('La solution exacte de u(x,y)')
fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

fig.savefig("Q_9_exacte.pdf")
