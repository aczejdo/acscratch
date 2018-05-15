import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import scipy.special as sp
from mpl_toolkits.mplot3d import Axes3D
import argparse          
from argparse import RawTextHelpFormatter
import struct


d=14 #seperation between "coils"
h=2 #width of tanh curve
N=8#resolution Even number
x_ = np.linspace(-5., 5., N+1)
z_ =np.linspace(-10,10,N+1)
y_ = np.linspace(-5., 5., N+1)

xg, yg, zg = np.meshgrid(x_,y_,z_, indexing='ij')
Bmax=4
Bmin=1
Bset=[Bmin,Bmax]	





def Bfield(x,y,z):
	
	[Bmin,Bmax]=Bset
	r=np.sqrt(x**2+y**2)
	phi=np.arctan2(y,x)
	b=0.5*(Bmax-Bmin)
	Bz=b*np.tanh(-z*h-d)*np.tanh(-z*h+d)+0.5*(Bmax+Bmin)
	Br=-(b*h*np.tanh(d+h*z)*((np.cosh(d-h*z))**-2)-b*h*np.tanh(d-h*z)*((np.cosh(d+h*z)**-2)))
	Bx=Br*np.cos(phi)
	By=Br*np.sin(phi)
	

	return(Bx,By,Bz)   
Bxg,Byg,Bzg=Bfield(xg,yg,zg)

#==================
#Vector Map 
#==================
print('vector mapping')
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(xg,yg,zg,Bxg,Byg,Bzg)
ax.dist=8                                   # camera distance
ax.set_ylabel('x')
ax.set_xlabel('y')
plt.show()
