import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import scipy.special as sp
from mpl_toolkits.mplot3d import Axes3D
import argparse          
from argparse import RawTextHelpFormatter




parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("Integrator",type=int,
					help="    rk4     : 0\n"
                            "   rk45single    :1 \n"
                            "   symplectic  : 2 ruth3")
parser.add_argument("Timestep",type=int,
					help="Order of Magnitude:\n"
						"   10^3    : 3 small \n"
						"   10^4   : 4 Medium \n"
						"   10^5   : 5 Large \n"
						"   10^6   : 6 V-large")
args        = parser.parse_args()
fINT           = args.Integrator
dt     = args.Timestep



G=fINT
W=dt-3

deltatvec=np.array([10**-3,10**-4,10**-5,10**-6])
intvec=['rk4','rk45single','symplectic']
dn=deltatvec.size #4
intn=3 #magic number size of vec





vname =('run%i%i'%(G,W))
ename =('run%i%i'%(G,3))
vplot=np.fromfile(vname,dtype=float, count=-1, sep='')
vsize=int(vplot.shape[0]/6)

p_v=np.zeros((6,vsize))
p_v[0,:]=vplot[0:vsize]
p_v[1,:]=vplot[2*vsize:3*vsize]
p_v[2,:]=vplot[4*vsize:5*vsize]
p_v[3,:]=vplot[vsize:2*vsize]
p_v[4,:]=vplot[3*vsize:4*vsize]
p_v[5,:]=vplot[5*vsize:6*vsize]

genvec=np.linspace(0,vsize-1,vsize)





exact=vplot=np.fromfile(ename,dtype=float, count=-1, sep='')
p_e=np.zeros((6,vsize))
p_e[0,:]=exact[0:vsize]=exact[0:vsize]
p_e[1,:]=exact[2*vsize:3*vsize]
p_e[2,:]=exact[4*vsize:5*vsize]
p_e[3,:]=exact[vsize:2*vsize]
p_e[4,:]=exact[3*vsize:4*vsize]
p_e[5,:]=exact[5*vsize:6*vsize]

fig = plt.figure()
ax = fig.add_subplot(231)
ax.set_title('Xpos')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.plot(p_e[0,:]-p_v[0,:])

ax = fig.add_subplot(232)
ax.set_title('Ypos')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.plot(p_e[1,:]-p_v[1,:])

ax = fig.add_subplot(233)
ax.set_title('Zpos')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.plot(p_e[2,:]-p_v[2,:])

ax = fig.add_subplot(234)
ax.set_title('Xvel')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.plot(p_e[3,:]-p_v[3,:])

ax = fig.add_subplot(235)
ax.set_title('Yvel')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.plot(p_e[4,:]-p_v[4,:])

ax = fig.add_subplot(236)
ax.set_title('Zvel')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.plot(p_e[5,:]-p_v[5,:])

plt.show()




fig=plt.figure()
#================
#Total Motion
#================
ax = fig.add_subplot(3,2,1,projection='3d') 
title=("%s %s"%(vname,'3d'))
print(title)
print(p_v)

ax.set_title(title)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.plot(p_v[1,:], p_v[2,:], p_v[3,:], label='parametric curve')

#===================
#Radius to Z-position
#===================
ax = fig.add_subplot(3,2,2) 
title=("radius wrt z")
ax.set_title(title)
print(title)

ax.plot(p_v[2,:] ,np.sqrt(p_v[0,:]**2+p_v[1,:]**2))


#===================
#Kinetic Energy
#===================
ax = fig.add_subplot(3,2,3) 
title=("KE")
ax.set_title(title)
print(title)
ax.plot(genvec,0.5*0.1*(p_v[0,:]**2+p_v[1,:]**2+p_v[2,:]**2))


#===================
#Angular Momentum Origin
#===================
ax = fig.add_subplot(3,2,4)
title=("L wrt Origin")
ax.set_title(title)
print(title)
Lorigin=np.zeros((vsize))
for i in range(0,vsize):
	Lorigin[i]=np.sum(np.cross(0.1*p_v[0:3,i],p_v[3:6,i]))
ax.plot(genvec,Lorigin)



#===================
#Angular Momentum Z-axis
#===================
ax = fig.add_subplot(3,2,5) 
title=("L wrt z")
ax.set_title(title)
print(title)
Lzaxis=np.zeros((vsize))
for i in range(0,vsize):
	Lzaxis[i]=np.sum(np.cross(0.1*p_v[0:2,i],p_v[3:5,i]))
ax.plot(genvec,Lzaxis)


#===================
#Z-position 
#===================

ax = fig.add_subplot(3,2,6) 
title=("Z-position wrt iter")
ax.set_title(title)
print(title)
ax.plot(genvec,p_v[3,:])


plt.show()	
