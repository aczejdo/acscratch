import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.fftpack import fftn
from scipy.fftpack import ifftn
from scipy.fftpack import fftfreq
import time

def fourier(f,fBNC,tol):
    J   =   f.shape[0]
    Jp  =   2*J 

    RHS =   np.zeros((Jp))
 
    RHS[J//2 : 3*J//2]   =   f
    #================================
    Fcoeff  =   fftn(RHS)
    freq    =   np.zeros((Jp))
    for n in range(0,Jp//2):
        freq[n]  = n
    for n in range(Jp//2,Jp):
        freq[n]  = -Jp - n
    
    
    #print(freq.shape, "freq shape \n", Jp, "Jp \n" , freq, "freq")
  
    
    Fcoeff[0]=0
    
    #X AXIS
    for i in range(1,Jp):
            eigi  =   np.cos(2 * freq[i] * np.pi / Jp ) - 1
            eig              =    (eigi)
            
            Fcoeff[i]    =   (dt**2)* Fcoeff[i] / eig
    #Fcoeff=np.nan_to_num(Fcoeff)
   
   
    
    
    #==============================
    u=ifftn(Fcoeff)
    #==============================
    u=u.real
    lhs=u[J//2 : 3*J//2]
    
    return(lhs)


def jacobi(f,fBNC,tol,**kwargs): #acually jacobi but maybe not
    #insize  2^n
    #outsize 2^n 
  
    maxit = 10000
    s = f.shape
    J         = s[0]
    u1        = np.zeros((J+2)) # initial guess.
    u2        = np.zeros((J+2))
    for key in kwargs:
        if (key == 'maxit'):
            maxit = kwargs[key]
        if (key == 'u0'):
            u1[1:J+1] = kwargs[key]
            u2[1:J+1] = kwargs[key]
            
    u1        = fBNC(f,u1)
    diff      = 1e30
    it        = 0

    while((diff > tol) & (it < maxit)):
        u2[:]   = u1[:]
        u1[:]   = fBNC(f,u2)
        #bt = it
        #if (bt%1000==10):
            #print(bt)
            #print(u1[0],u1[-1])
        #    bt=bt-1000
        u1[1:J+1] = 0.5 * (u1[0:J]+u1[2:J+2] - (dt**2)*f)    
        #for i in range(1,J+1):
        #   u1[i] = 0.5 * (u2[i+1] + u2[i-1] - (dt**2)*f[i-1])        
        it        = it+1
        
        
        diff      = get_rmsdiff(u1,u2)
        
    #print('[jacobi]: it=%5i diff/tol=%13.5e' % (it,diff/tol))
    return u1[1:J+1]
 
def get_rmsdiff(u1,u2):
    J = u1.shape[0]
    return np.sqrt(np.mean((u1[1:J+1]-u2[1:J+1])**2))
    
def mg_restrict(u):
    J     = u.shape[0]
    i     = (np.arange(J//2))
    uf    =  np.zeros((J//2))
    uf    = 0.5*(u[2*i]+u[2*i+1])

    return uf
  
def mg_prolong(u,fBNC):    
    #2^n in 
    #2^n out
    J               = u.shape[0] #size of J
    u1              = np.zeros((J+2)) #init
 
    u1[1:J+1] = u #leaves the frame for BC's
    u1              = fBNC(u1,u1) # note that the first argument is a dummy argument 
    i               = (np.arange(J,dtype=int)) #makes grid for prolonging
    mi              = u1[1:J+2]-u1[0:J+1] #slope
    uf              = np.zeros((2*J)) #new fine matrix


    uf[2*i+1] = u1[1:J+1] + 0.25*mi[1:J+1]  #forwards
    uf[2*i  ] = u1[1:J+1] - 0.25*mi[0:J]    #backwards
    
    return uf    
       
def mg_residual(u,f,fBNC):
    #2^n in
    #2^n+2 out
    J               = f.shape[0] 
    u1              = np.zeros((J+2))
    u1[1:J+1]       = u 
    u1              = fBNC(f,u1)  
    residual               = f - (u1[0:J]+u1[2:J+2]-2.0*u1[1:J+1])/(dt**2) 
    return residual     

def mg_vcycle(f,fBNC,npre,npst,level,**kwargs):
    for key in kwargs:
        if(key=='u0'):
            u0=kwargs[key]
        elif(key=='maxxit'):
            maxxit=kwargs[key] 
    J=f.shape[0]
    
    print('at', J)
    if(level>1):
        if(J==N):
            vh   = jacobi(f,fBNC,tol,u0=u0,maxxit=npre)
        else:
            vh   = jacobi(f,bnc_nocharge,tol,u0=u0,maxxit=npre)
        rh   = mg_residual(vh,f,fBNC)
        print('restrict to', J//2)
        f2h  = mg_restrict(rh)
        v2h = np.zeros((J//2))
        v2h = mg_vcycle(f2h,bnc_nocharge,npre,npst,level-1,u0=v2h)
        print('prolong to', J) 
        vh   = vh + mg_prolong(v2h,fBNC)
        u   = jacobi(f,fBNC,tol,u0=vh,maxxit=npst)
        
    if (level==1):
        print('lowest level', J)
        u   = jacobi(f,fBNC,tol,u0=u0,maxxit=npst)
    
    return(u)

    
def multigrid(f,fBNC,tol):
    #should be goodfor 3D
    npre = 10
    npst = 80
    J    = f.shape[0]
    ui = np.zeros((J))
    if (J % 2 == 1):
        print('[multigrid]: J must be even: J=%4i' % (J))
        exit()
    
    level=np.log2(J)
    
    u    = mg_vcycle(f,fBNC,npre,npst,level,u0=ui, verbose=1)
    #print(np.amin(u))

    for i in range(10):
        u = mg_vcycle(f,fBNC,npre,npst,level,u0=u,verbose=0)
        #print(np.max(u))
        print(i)
    return u
    
def bnc_none(f,u):
    return(u) 
def bnc_monopole(f,u):
    J  =  u.shape[0]-2
    if(J<N-2):
        u = bnc_nocharge(f,u)
    else:
        u[0  ]  = UU[0  ]
        u[J+1]  = UU[J+1]

    return u  
def bnc_nocharge(f,u):
    J        = u.shape[0]-2
    u[0    ] = -u[1]
    u[-1  ] = -u[-2]
    return u

def bnc_dir(f,u):
    u[0] = 10
    u[-1] = 10
    return u
 
def Usphere(xarr,bf,N):
    # in n 
    # out n
    RHS=np.zeros((N+2))
    rad=0.5 #arbitrary
    m=50
    b=(Bmax-Bmin)*0.5
    B=(Bmax+Bmin)*0.5
    for i in range(0,N+1):
        xrhs=xarr[i]
      
        R=np.sqrt(xrhs**2)
        
        R= (-np.tanh(m*R-m*rad))*b+B
        RHS[i]=R

    NRHS=RHS[1:-1]
    return(NRHS) 
    
def init(problem,N):
    x = np.linspace(-ghostmax, ghostmax, N+2)
    if(problem == 'sphere'):
        bc=bnc_monopole
        #bc= bnc_dir
        #bc = bnc_nocharge
        print('calculating monopole boundaries')
        global UU
        UU    =np.zeros((N+2))
        G     = -1
        displacement     = G/np.abs(x)
        UU[:] = displacement
        print("done!")
        
        rhs = Usphere(x,0,N)
    else:
        print('problem not in 1d code')
        exit()
        
    return(rhs,x,bc)

    
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)  
parser.add_argument("prob",type=str,
                    help="source field:\n"
                        "    bottle   : magnetic bottle radial on z axis\n"
                        "    cosine   : cosine in xy plane \n"  
                        "    sphere   : solid spherical potential") 
parser.add_argument("points",type=int,
                    help="number of spatial support points in each dimension")
parser.add_argument("tol",type=float,
                    help="solver tolerance") 

                    
args        = parser.parse_args()
N           = args.points
problem     = args.prob                   
tol             = args.tol

bound=10.         #radius
dt=(2.*bound)/N    #dt for cell-cell-centered
ghostmax=bound+0.5*dt  #max for ghost cells

d=10. #seperation between "coils"
h=2. #width of tanh curve

Bmax=5. #B field strength
Bmin=0. 
#Bset=[Bmin,Bmax]

nrhs,x,bc  = init(problem,N)   

 
if(bc == bnc_monopole):
    print()
else: 
    print('not doing monopole boundaries for mg')
    
    
X=x[1:N+1]
    

#for i in range(0,N):

fig=plt.figure()
ax=fig.gca()
ax.plot(X,nrhs)
ax.set_ylabel('rhs')
ax.set_xlabel('x')
plt.title('RHS')
plt.show()


    
    

#====================================================
#t=time.time()
#====================================================    
    
    
#===========================================================
#===========================================================
#===========================================================
#===========================================================
#===========================================================
print('doing multigrid')
sol     =   multigrid(nrhs,bc,1E-8)
sol2     =   jacobi(nrhs,bc,1E-8)
print('done!')
print('doing fourier')
sol1    =   fourier(nrhs,bnc_none,0.00000001)
print('done!')
dif     =   sol -  sol1

print(np.min(sol),'mg min')
print(np.min(sol1),'fr min')
#===========================================================
#===========================================================
#===========================================================
#===========================================================
#===========================================================


#elapsed=time.time() - t
#print(elapsed, 'time')
#exit()


#for i in range(0,N):
fig=plt.figure()

ax=fig.add_subplot(321)
ax.set_ylabel('sol')
ax.set_xlabel('x')
plt.title('MG sol')
ax.plot(X,sol)



ax=fig.add_subplot(323)
ax.set_ylabel('sol')
ax.set_xlabel('x')
plt.title('JC sol')
ax.plot(X,sol2)

ax=fig.add_subplot(325)
ax.set_ylabel('sol')
ax.set_xlabel('x')
plt.title('Fourier sol')
ax.plot(X,sol1)


ax=fig.add_subplot(322)
ax.set_ylabel('res')
ax.set_xlabel('x')
plt.title('MG F')
ax.plot(X,dif)

ax=fig.add_subplot(324)
ax.set_ylabel('res')
ax.set_xlabel('x')
plt.title('JC F')
ax.plot(X,sol2-sol1)

ax=fig.add_subplot(326)
ax.set_ylabel('res')
ax.set_xlabel('x')
plt.title('MG JC')
ax.plot(X,sol2-sol)

plt.show()

print(1/x[0],1/x[-1], 'monopole')
print(sol[0],sol[-1], 'mg bounds')
print(sol1[0],sol1[-1], 'fr bounds')