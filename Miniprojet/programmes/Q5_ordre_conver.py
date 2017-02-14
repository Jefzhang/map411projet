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
#dx = 0.1      # dx = pas d'espace
#dt = 0.025    # dt = pas de temps
Tfinal = 5   # Temps final souhaite
Vitesse = 1   # Vitesse de difusion
Coeffi = 1    #Coefficient de difusion
xmean = 20
sigma = 1


#define the functions necessaire
def funInit(x,x0,sigma):
    a = np.exp(-(x - x0)**2/(2*sigma**2))
    return a/sigma/np.sqrt(2*np.pi)

pasdx = np.linspace(0.05,0.5,20)
err1 = np.zeros(len(pasdx))
err2 = np.zeros(len(pasdx))


for i in range(1,len(pasdx)+1):
    dx = pasdx[i-1]
    dt = 0.25 * dx / Vitesse

    # Initialisation
    nx =  int(lg/dx)-1  # nx = nombre d'intervals-1 , =N
    x = np.linspace(0,lg,nx+2)
    #print(len(x))

    # Initialize u0
    u0 = np.zeros(len(x))
    for k in range(1,nx):
        u0[k] = funInit(x[k],xmean,sigma)

    uexpdecen = u0.copy()    #initial value of schema explicite decentree
    ucrankni = u0.copy()     #initial value of schema Crank-Nicholson

    Kc = sp.sparse.diags([-1/(2*dx),0,+1/(2*dx)],[-1,0,1],shape=(nx,nx))
    Aexpcen = sp.sparse.identity(nx) - Vitesse*dt*Kc
    #print(Aexpcen.shape)
    Kd = sp.sparse.diags([-1/dx,1/dx],[-1,0],shape=(nx,nx))
    Aexpdecen = sp.sparse.identity(nx) - Vitesse*dt*Kd

    Acrank1 = sp.sparse.identity(nx) + Vitesse*dt* Kc/2
    Acrank2 = sp.sparse.identity(nx) - Vitesse*dt* Kc/2
    nt = int(Tfinal / dt)
    Tfinal = dt * nt
    for n in range(1,nt+1):
        uexpdecen[1:nx+1] = Aexpdecen * uexpdecen[1:nx+1]
        ucrankni[1:nx+1] = sp.sparse.linalg.spsolve(Acrank1,Acrank2*ucrankni[1:nx+1])
    uext = np.zeros(len(x))
    for k in range(1,nx):
        uext[k] = funInit(x[k],xmean+Vitesse*Tfinal,sigma)
    err1[i-1] = sp.linalg.norm(uexpdecen-uext,2)/np.sqrt(nx)
    err2[i-1] = sp.linalg.norm(ucrankni-uext,2)/np.sqrt(nx)

pasdx = np.log(pasdx)
err1 = np.log(err1)
err2 = np.log(err2)


plt.figure()
decen, =plt.plot(pasdx,err1,'b',label = "Decentre amont")
crank, =plt.plot(pasdx,err2,'r',label = "Crank-Nicholson")
plt.legend([decen,crank],["Decentre amont","Crank-Nicholson"])
plt.xlabel('$\ln(x)$')
plt.ylabel('$\ln(err)$')
plt.title('Ordre de convergence des schemas')
plt.savefig('ordre.png')
plt.show()
