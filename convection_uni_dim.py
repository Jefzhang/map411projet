import numpy as np
import matplotlib.pyplot as plt # pour plot functons
plt.switch_backend('tkAgg')  # necessary for OS SUSE 13.1 version,
# otherwise, the plt.show() function will not display any window

import pylab
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg


########################################################
# Parametres du probleme
lg = 50       # intervalle en x=[0,lg]
dx = 0.1      # dx = pas d'espace
dt = 0.025    # dt = pas de temps
Tfinal = 5   # Temps final souhaite
Vitesse = 1   # Vitesse de difusion
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
print(len(x))

# Initialize u0
u0 = np.zeros(len(x))
for k in range(1,nx):
    u0[k] = funInit(x[k],xmean,sigma)

# Plot initial condition
plt.ion()
plt.plot(x,u0,'o')
#plt.show()

# We also write the intial condition into a file
# of *.png or *.pdf format
fig, ax = plt.subplots(nrows=1,ncols=1)
ax.plot(x,u0) # trace u0 en fonction de x
fig.savefig('u0convection.png')
plt.close(fig)


########################################################
# Schemas numeriques

# Initialize u by the initial data u0
uexpcen = u0.copy()
#print(len(uexpcen))

# Construction de la matrice (creuse) pour le schema explicite cnetree
Kc = sp.sparse.diags([1/(2*dx),0,-1/(2*dx)],[-1,0,1],shape=(nx,nx))
Aexpcen = sp.sparse.identity(nx) - Vitesse*dt*Kc
#print(Aexpcen.shape)

# Nombre de pas de temps effectues
nt = int(Tfinal/dt)
Tfinal = nt*dt # on corrige le temps final (si Tfinal/dt n'est pas entier


# Time loop
for n in range(1,nt+1):
    uexpcen[1:nx+1] = Aexpcen * uexpcen[1:nx+1]

   # Print solution
    if n%5 == 0:
        plt.figure(2)
        plt.clf()
        #plt.subplot(121)
        plt.plot(x,u0,'b',x,uexpcen,'r')
        plt.xlabel('$x$')
        plt.title('Schema explicite centree')
        plt.pause(0.1)
