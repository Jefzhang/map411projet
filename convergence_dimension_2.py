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
from numpy import linalg as LA


########################################################
# parameters of the problem
L = 1           # interval between x = [0, L], y = [0, L]
# dh = 0.1        # cell size in space for both x and y


# source term f
def f_source(x, y):
    return -2 * y * (y - L) - 2 * x * (x - L)


def u_exact(x, y):
    return x * (x - L) * y * (y - L)


def u_numerical(dh, N):
    # define matrix A
    DIAG_1 = np.ones(N * N - 1)
    for i in range(N - 1):
        DIAG_1[(i + 1) * N] = 0
    DIAG_2 = np.ones(N * N - N)

    # define matrix A
    A_diag_1 = DIAG_1 / (dh * dh)
    A_diag_2 = DIAG_2 / (dh * dh)
    A = sp.sparse.diags([A_diag_2, A_diag_1, -4 / (dh * dh),
                        A_diag_1, A_diag_2],
                        [-N, -1, 0, 1, N],
                        shape=(N * N, N * N))
    x_range = np.linspace(0, L, N + 2)
    f = np.ndarray(shape=(N * N))

    # Set specific f values
    for i in range(N):
        for j in range(N):
            f[i * N + j] = f_source(x_range[i + 1], x_range[j + 1])

    U = sp.sparse.linalg.spsolve(-A, f)
    U = U.reshape(N, N)
    return U


########################################################
# Initialisation


'''
# Q_9
N = int(L / dh - 1)   # N = number of intervals
# define matrix A
DIAG_1 = np.ones(N * N - 1)
for i in range(N - 1):
    DIAG_1[(i + 1) * N] = 0
DIAG_2 = np.ones(N * N - N)

# define matrix A
A_diag_1 = DIAG_1 / (dh * dh)
A_diag_2 = DIAG_2 / (dh * dh)
A = sp.sparse.diags([A_diag_2, A_diag_1, -4 / (dh * dh), A_diag_1, A_diag_2],
                    [-N, -1, 0, 1, N], shape=(N * N, N * N))

x_range = np.linspace(0, L, N + 2)
f = np.ndarray(shape=(N * N))

# Set specific f values
for i in range(N):
    for j in range(N):
        f[i * N + j] = f_source(x_range[i + 1], x_range[j + 1])

U = sp.sparse.linalg.spsolve(-A, f)
U = U.reshape(N, N)

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(dh, L, dh)
Y = np.arange(dh, L, dh)
X, Y = np.meshgrid(X, Y)
Z = U.copy()
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

ax.set_title('La solution numerique de u(x,y)')
fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

fig.savefig("Q_9.pdf")

'''

########################################################
# oberservation of the order of convergence

dh_list = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.025, 0.04, 0.05, 0.1, 0.2]
error = []

for dh_prime in dh_list:
    N = int(L / dh_prime - 1)
    x_range = np.linspace(0, L, N + 2)
    u_approximation = u_numerical(dh_prime, N)
    u_theoretical = np.ndarray(shape=(N, N))
    for i in range(N):
        for j in range(N):
            u_theoretical[i][j] = u_exact(x_range[i + 1], x_range[j + 1])

    error.append(LA.norm(u_approximation - u_theoretical, 2) * dh_prime)

X = dh_list
Y = error

fig = plt.figure()
plt.plot(X, Y, "o-", label="ligne -")
plt.xlabel('h')
plt.ylabel('Erreur')
plt.title('Convergence de la solution')

fig.savefig("Q_9_evolution.pdf")
