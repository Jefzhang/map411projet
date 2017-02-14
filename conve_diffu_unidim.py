import numpy as np
import matplotlib.pyplot as plt # pour plot functons
plt.switch_backend('tkAgg')  # necessary for OS SUSE 13.1 version,
# otherwise, the plt.show() function will not display any window

import pylab
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import coo_matrix, bmat


########################################################
# Parametres du probleme
lg = 50       # intervalle en x=[0,lg]
dx = 0.1      # dx = pas d'espace
dt = 0.025    # dt = pas de temps
Tfinal = 10   # Temps final souhaite
Vitesse = 2   # Vitesse de difusion
Coeffi = 1    #Coefficient de difusion
xmean = 20
sigma = 1

#define the functions necessaire
def funInit(x,x0,sigma):
    a = np.exp(-(x - x0)**2/(2*sigma**2))
    return a/sigma/np.sqrt(2*np.pi)

# Initialisation
nx =  int(lg/dx)-1  # nx = nombre d'intervals-1 , =N
x = np.linspace(0,lg,nx+2)
#print(len(x))

# Initialize u0
u0 = np.zeros(len(x))
for k in range(1,nx):
    u0[k] = funInit(x[k],xmean,sigma)

u1 = np.zeros(len(x))   # Pour Q8, la concentration t=0 est nulle
fsour = np.zeros(len(x))
for k in range(1,nx):
    fsour[k] = funInit(x[k],xmean,sigma)
########################################################
# Schemas numeriques

# Initialize u by the initial data u0
ucrankni = u0.copy()

# Construction de la matrice (creuse) pour le schema explicite cnetree
Kc = sp.sparse.diags([-1/(2*dx),0,+1/(2*dx)],[-1,0,1],shape=(nx,nx))
A = sp.sparse.diags([1/(2*dx),-1/dx,1/(2*dx)],[-1,0,1],shape=(nx,nx))


########################################################
# modifier Kc et A


Kc = Kc.asformat('csc')       #convert the diag format to csc sparse matrix, then we can get the items that we want to change
Kc[nx-1,nx-1] += 1/(2*dx)
A = A.asformat('csc')
A[nx-1,nx-1] += 1/(2*dx)






Term = sp.sparse.identity(nx) + Vitesse*dt*Kc/2# - Coeffi*dt*A/2
Mata = Term - Coeffi*dt*A/2
Term = sp.sparse.identity(nx) - Vitesse*dt*Kc/2
Matb = Term + Coeffi*dt*A/2

# Nombre de pas de temps effectues
nt = int(Tfinal/dt)
Tfinal = nt*dt # on corrige le temps final (si Tfinal/dt n'est pas entier

# Time loop
plt.ion()
for n in range(1,nt+1):
    #ucrankni[1:nx+1] = sp.sparse.linalg.spsolve(Mata,Matb*ucrankni[1:nx+1])


    if n%2 ==0:
        u1[1:nx+1] = sp.sparse.linalg.spsolve(Mata,Matb*u1[1:nx+1]+fsour[1:nx+1])
    else:
        u1[1:nx+1] = sp.sparse.linalg.spsolve(Mata,Matb*u1[1:nx+1])



    if n%5 == 0:
        plt.figure(1)
        plt.clf()

        '''
        plt.subplot(111)
        plt.plot(x,u0,'b',x,ucrankni,'r')
        plt.xlabel('$x$')
        plt.title('Schema Crank-Nicholson')
        if(n == nt/2):
            plt.savefig('diffsortiemo.png')
        '''


        plt.subplot(111)
        plt.plot(x,u1,'r')
        plt.xlabel('$x$')
        plt.title('Schema Crank-Nicholson avec le terme source')
        if(n == nt/2):
            plt.savefig('CNsource.png')


        plt.pause(0.1)
