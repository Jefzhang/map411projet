# -*- coding: utf-8 -*-

# Load necessary packages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
from scipy import stats
from scipy.stats import norm as gaussian
from scipy.sparse import linalg

########################################################
# parameters of the problem
L = 50                  # interval between x = [0, L], y = [0, L]
T = 5                   # inerval of time between t = [0, T]
dh = 0.5                # cell size in space for both x and y
dt = 0.1                # cell size in time for t
N = int(L / dh - 1)     # N = number of intervals
Nt = int(T / dt)        # Nt = number of iteration on time
nu = 1                  # nu value


# initial condition of u
def f_initial(x, y, x0, y0, sigma):
    return 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
        -((x - x0) * (x - x0) + (y - y0) * (y - y0)) /
        (2 * sigma * sigma))


# source term f
# x0 = 25, y0=0, sigma = 1
def f_source(x, y):
    return f_initial(x, y, 25, 0, 1)


# precise solution u_exact
def u_exact(x, y):
    return x * (x - L) * y * (y - L)

########################################################
# Initialisation
########################################################


DIAG_1 = np.ones(N * N - 1)
for i in range(N - 1):
    DIAG_1[(i + 1) * N] = 0
DIAG_2 = np.ones(N * N - N)

I_N = sp.sparse.diags([1], [0], shape=(N * N, N * N))

# define matrix A
A_diag_1 = DIAG_1 / (dh * dh)
A_diag_2 = DIAG_2 / (dh * dh)
A = sp.sparse.diags([A_diag_2, A_diag_1, -4 / (dh * dh), A_diag_1, A_diag_2],
                    [-N, -1, 0, 1, N], shape=(N * N, N * N))

# define matrix kc_x and kc_y
kc_x_diag_1 = DIAG_2 / (2 * dh)
kc_x = sp.sparse.diags([-kc_x_diag_1, kc_x_diag_1],
                       [-N, N], shape=(N * N, N * N))

kc_y_diag_1 = DIAG_1 / (2 * dh)
kc_y = sp.sparse.diags([-kc_y_diag_1, kc_y_diag_1],
                       [-1, 1], shape=(N * N, N * N))

# define range of variable x and y
x_range = np.linspace(0, L, N + 2)
U0 = np.ndarray(shape=(N * N))

# set up initial data U0
# u0 same as source term
for i in range(N):
    for j in range(N):
        U0[i * N + j] = f_initial(x_range[i + 1], x_range[j + 1], 25, 25, 1)


# Numeric simulation for Q11
# matrix for U_n+1 Q11
M = I_N + dt / 2 * (kc_x + kc_y) - nu * dt / 2 * A
M_prime = I_N - dt / 2 * (kc_x + kc_y) + nu * dt / 2 * A


# solve linear equation for U
U = U0.copy()
fig = plt.figure()
for n in range(Nt):
    U = sp.sparse.linalg.spsolve(M, M_prime * U)

    ax = fig.gca(projection='3d')
    X = np.arange(dh, L, dh)
    Y = np.arange(dh, L, dh)
    X, Y = np.meshgrid(X, Y)
    Z = U.copy().reshape(N, N)
    ax.clear()
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)

    ax.set_zlim3d(0, 0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')

    if n == 0:
        ax.set_title('u(t=0,x,y)')
        fig.savefig("Q_11_t=0.pdf")
    if n == Nt / 4:
        ax.set_title('u(t=T/4,x,y)')
        fig.savefig("Q_11_t=T-4.pdf")
    if n == Nt / 2:
        ax.set_title('u(t=T/2,x,y)')
        fig.savefig("Q_11_t=T-2.pdf")
    if n == Nt - 1:
        ax.set_title('u(t=T,x,y)')
        fig.savefig("Q_11_t=T.pdf")

    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()

    plt.pause(0.1)

    # fig.savefig("Q_11.pdf")
