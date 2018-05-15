import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import scipy.special as sp
from mpl_toolkits.mplot3d import Axes3D
import argparse          
from argparse import RawTextHelpFormatter
import struct

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("mode",type=int,
					help="    read or write    : 0 or 1 \n")
args        = parser.parse_args()
mode         = args.mode

d=3 #seperation between "coils"
h=1 #width of tanh curve
N=6#resolution Even number
x_ = np.linspace(-5., 5., N+1)
z_ =np.linspace(-10,10,N+1)
y_ = np.linspace(-5., 5., N+1)

xg, yg, zg = np.meshgrid(x_,y_,z_, indexing='ij')
Bmax=100
Bmin=10
Bset=[Bmin,Bmax]	
	

E=[0,0,0]
rhs=np.zeros((6))

Tfin=20*np.pi


m=1/10
q=-1/100


def Bfield(x,y,z,h,d,Bset):
	
	[Bmin,Bmax]=Bset
	r=np.sqrt(x**2+y**2)
	phi=np.arctan2(y,x)
	b=0.5*(Bmax-Bmin)
	Bz=b*np.tanh(-z*h-d)*np.tanh(-z*h+d)+0.5*(Bmax+Bmin)
	Br=-(b*h*np.tanh(d+h*z)*((np.cosh(d-h*z))**-2)-b*h*np.tanh(d-h*z)*((np.cosh(d+h*z)**-2)))
	Bx=Br*np.cos(phi)
	By=Br*np.sin(phi)
	

	return(Bx,By,Bz)   
#===========================
def euler(y,fRHS,dt,h,d,Bset,m,null1,null2):
	x=y+fRHS(y,h,d,Bset,m)*dt
	return(x,null2)
def symplectic(pv,fRHS,dt,h,d,Bset,m,numbers,null2):
	RHS=fRHS(pv,h,d,Bset,m)
	accel=RHS[3:6]
	v=RHS[0:3]
	p=pv[0:3]
	coeffs=numbers
	for ai,bi in coeffs.T:
		v += bi * accel * dt
		p += ai * v* dt
	pv[0:3]=p
	pv[3:6]=v
	
	return(pv,null2)
def rk4 (pv,fRHS,dt,h,d,Bset,m,null1,null2):
        
	k1=fRHS(pv,h,d,Bset,m)

	pv2=pv+0.5*dt*k1
	k2=fRHS(pv2,h,d,Bset,m)

	
	pv3=pv+0.5*dt*k2

	k3=fRHS(pv3,h,d,Bset,m)
	
	

	pv4=pv+dt*k3
	k4=fRHS(pv4,h,d,Bset,m)
	
	x=pv+(dt/6)*(k1+2*k2+2*k3+k4)
	
	return(x,null2)	
def rk45single(pv,fRHS,dt,h,d,Bset,m,null,yerr): #fRHS,x0,y0,dx
	dx=dt
	a         = np.array([0.0,0.2,0.3,0.6,1.0,0.875]) # weights for x
	b         = np.array([[0.0           , 0.0        , 0.0          , 0.0             , 0.0         ],
								[0.2           , 0.0        , 0.0          , 0.0             , 0.0         ],
								[0.075         , 0.225      , 0.0          , 0.0             , 0.0         ],
								[0.3           , -0.9       , 1.2          , 0.0             , 0.0         ],
								[-11.0/54.0    , 2.5        , -70.0/27.0   , 35.0/27.0       , 0.0         ],
								[1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]])
	c         = np.array([37.0/378.0,0.0,250.0/621.0,125.0/594.0,0.0,512.0/1771.0])
	dc        = np.array([2825.0/27648.0,0.0,18575.0/48384.0,13525.0/55296.0,277.0/14336.0,0.25])
	dc        = c-dc
	n         = pv.size
	dy        = np.zeros(n)        # updates (arguments in f(x,y))
	dydx      = np.zeros((6,n))    # derivatives (k1,k2,k3,k4,k5,k6)
	yout      = pv                 # result
	yerr      = np.zeros(n)        # error
	dydx[0,:] = dx*fRHS(pv,h,d,Bset,m)  # first guess
	for i in range(1,6):           # outer loop over k_i 
		dy[:]     = 0.0
		for j in range(i):         # inner loop over y as argument to fRHS(x,y)
			dy = dy + b[i,j]*dydx[j,:]
		dydx[i,:] = dx*fRHS(pv+a[i],h,d,Bset,m) #x0+a[i]*dx,y0+dy,a[i]*dx
	for i in range(0,6):           # add up the k_i times their weighting factors
		yout = yout + c [i]*dydx[i,:]
	yerr = yerr + dc[i]*dydx[i,:] 

	return (yout,yerr)
#=============================
def fRHS(pv,h,d,Bset,m):
	x	=	pv[0]
	y	=	pv[1]
	z	=	pv[2]
	V	=	pv[3:6]
	Bx,By,Bz	=	Bfield(x,y,z,h,d,Bset)
	B				=	[Bx,By,Bz]
	F				=	q*(E+np.cross(V,B))
	
	rhs[0:3]	=	V
	rhs[3:6]	=	F/m

	return(rhs)
#==========
#integrator coefficients
#=============================================
c = np.math.pow(2.0, 1.0/3.0)
ruth4 = np.array([[0.5, 0.5*(1.0-c), 0.5*(1.0-c), 0.5],
                  [0.0,         1.0,          -c, 1.0]]) / (2.0 - c)
ruth3 = np.array([[2.0/3.0, -2.0/3.0, 1.0], [7.0/24.0, 0.75, -1.0/24.0]])
leap2 = np.array([[0.5, 0.5], [0.0, 1.0]])
#=============================================
def run(integrator,deltat):
	integratorfunc=integrator
	dt=deltat
	print(integratorfunc.__name__)
	print(dt)
	#+++++++++++++++++++++++++++++++++++++++++++
	dumptime=10**-3
	plotres=int(np.floor(Tfin/dumptime))
	#++++++++++++++++++++++++++++++++++++++++++++

	
	#Imax=int(Tfin/dt)
	Imax=plotres
	perc=0.1*Imax
	

	pos=np.zeros((3,Imax))
	vel=np.zeros((3,Imax))
	pv=np.zeros((6,Imax))

	pos[:,0]=2*np.random.rand(3)-1
	vel[:,0]=4*np.random.rand(3)-2

	#=========INITIAL CONDITIONS===========================
	r_0=0.7
	z_0=0.5
	a1,a2,B_0=Bfield(r_0,0,z_0,h,d,Bset)
	vperp=r_0*np.abs(q)*B_0/m
	print(r_0,"radius",vperp,"perpendicular vel")
	print(B_0)

	#============
	pos[:,0]=[r_0,0,z_0]
	vel[:,0]=[vperp,0,0]
	pv[0:3,0]=pos[:,0]
	pv[3:6,0]=vel[:,0]


	print(pv[:,0])

	
	"""
	Imod=int(np.floor(Imax))
	percentvec=np.linspace(0,Imod,10)
	iter0=1
	
	
	for I in range(1,Imod):
		
		#Percentage
			if  Imod % percentvec[iter0] ==0:
				print ('{}% complete'.format(int(100 * I / Imod)))
				iter0=iter0+1
	"""	 

	tempres=int(np.floor(dumptime/dt))+1
	temppv=np.zeros((6,tempres))
	temppv[:,0]=pv[:,0]
	
	yerr=0.001
	print(plotres," plotres ",tempres," tempres ")
	for j in range(1,plotres):
		print(j)
		for i in range(1,tempres):
			temppv[:,i],yerr=integratorfunc(temppv[:,i-1],fRHS,dt,h,d,Bset,m,ruth3,yerr)
		temppv[:,0]=temppv[:,-1]
		pv[:,j]=temppv[:,-1]
	pv[:,-1]=temppv[:,-1]
			

	print(pv.shape)	
	return(pv)


#==============
#Vector field
#=============
Bxg,Byg,Bzg=Bfield(xg,yg,zg,h,d,Bset)

#==================
#Vector Map 
#==================
"""
print('vector mapping')
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(xg,yg,zg,Bxg,Byg,Bzg)
ax.dist=8                                   # camera distance
ax.set_ylabel('x')
ax.set_xlabel('y')
plt.show()
"""




deltatvec=np.array([10**-3,10**-4,10**-5,10**-6])
intvec=[rk4,rk45single,symplectic]

#=====================
#Singe Plotter
#====================
"""
fig = plt.figure()
ax = fig.gca(projection='3d')
pv=run(intvec[2],deltatvec[1])
ax.plot(pv[0,:], pv[1,:], pv[2,:], label='parametric curve')
plt.show()
"""

#==============
#Plotting loops
#==============

deltatvec=np.array([10**-3,10**-4,10**-5,10**-6])
intvec=[rk4,rk45single,symplectic]
dn=deltatvec.size
intn=3 #magic number size of vec
fig = plt.figure() #figsize=plt.figaspect(0.5) inside
n=1
i=0
guide=np.array([1,4,7,10,13,2,5,8,11,14,3,6,9,12,15])

if (mode==0):
	#Writer
	#I JUST EDITED THIS TO MAKE AN ULTRA HIGH RES RUN DELETE SOME OF THESE LINES
	G=2 #SET TO 0 
	#for G in range(1,1):
	for W in range(0,1):
		print(G, " G ",W," W ")
		pv=run(intvec[G],10**-7) # REPLACE 107 with deltavec[W]
		print(pv.shape, " pv shape" )
		vname =('run%i%i'%(9,9)) # REPLACE 9 9 WITH G AND W 
		pv.tofile(vname, sep="", format="%s")
		del pv

#updated reader

elif (mode==1):
	for GG in range(0,3): #change to intn
		for WW in range(0,4): #change val to dn
			ax = fig.add_subplot(5,3,guide[n-1],projection='3d') 
			vname =('run%i%i'%(GG,WW))
			vplot=np.fromfile(vname,dtype=float, count=-1, sep='')
			vsize=int(vplot.shape[0]/6)
			title=("%s %i"%(vname,vsize))
			ax.set_title(title)
			p_v=np.zeros((3,vsize))
			p_v[0,:]=vplot[0:vsize]
			p_v[1,:]=vplot[2*vsize:3*vsize]
			p_v[2,:]=vplot[5*vsize:6*vsize]
			ax.plot(p_v[0,:], p_v[1,:], p_v[2,:], label='parametric curve')	
			n=n+1	
	plt.show()	
else:
	print('done')
#==============
#Energy runoff
#==============
"""
vsum=np.sqrt(p_v[3,:]**2+p_v[4,:]**2+p_v[5,:]**2)
sumnum=np.linspace(0,100,len(vsum))
fig =plt.figure
plt.plot(sumnum,vsum)
plt.show()
"""
