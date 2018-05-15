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
from copy import deepcopy
def fourier(f,fBNC,tol):
    J   =   f.shape[0]
    Jp  = 2*J 
    #Jp  = 3*J 
    RHS =   np.zeros((Jp,Jp))
 
    RHS[J//2 : 3*J//2   ,J//2 : 3*J//2]   =   f
    #================================
    Fcoeff  =   fftn(RHS)
    freq    =   np.zeros((Jp))
    for n in range(0,Jp//2):
        freq[n]=n
    for n in range(Jp//2,Jp):
        freq[n]=-Jp-n
    
    
    #print(freq.shape, "freq shape \n", Jp, "Jp \n" , freq, "freq")
  
    
    Fcoeff[0,0]=0
    
    #X AXIS
    for i in range(1,Jp):
            eigi  =   np.cos(2*freq[i] * np.pi / (Jp) ) - 1
            
            eig              =    2*(eigi)
            Fcoeff[i,0]    =   (dt**2)* Fcoeff[i,0] / eig
    #Y AXIS
    for j in range(1,Jp):
            eigj  =   np.cos( 2*freq[j] * np.pi / (Jp) ) - 1
            
            eig              =    2*(eigj)
            Fcoeff[0,j]    =   (dt**2)* Fcoeff[0,j] / eig
    
   
    #FULL
    for i in range(1,Jp):
        
        eigi =  np.cos(2* freq[i] * np.pi / (Jp) )  - 1
 
        for j in range(1,Jp):
            
            eigj =  np.cos( 2*freq[j] * np.pi / (Jp) )  - 1       
           
            eig              =    2*(eigi  +  eigj)
              
            Fcoeff[i,j]    =   (dt**2)* Fcoeff[i,j] / eig
    
    #Fcoeff=np.nan_to_num(Fcoeff)
   
   
    
    
    #==============================
    u=ifftn(Fcoeff)
    #==============================
    u=u.real
    #u=fBNC(f,u)
    rhs=u[J//2 : 3*J//2,  J//2 : 3*J//2]
    #rhs=u[J:2*J,J:2*J,J:2*J]*(8/Jp**3)
    
    return(rhs)


def jacobi(f,fBNC,tol,**kwargs): #acually jacobi
    maxit = 1000000
    s = f.shape
    if (s[0] != s[1]):
        print('[jacobi]: need square matrix.')
        exit()
    J         = s[0]
    u1        = np.zeros((J+2,J+2)) # initial guess.
    u2        = np.zeros((J+2,J+2))
    for key in kwargs:
        if (key == 'maxit'):
            maxit = kwargs[key]
        if (key == 'u0'):
            u1[1:J+1,1:J+1] = kwargs[key]
            u2[1:J+1,1:J+1] = kwargs[key]
    u1        = fBNC(f,u1)
    diff      = 1e30
    it        = 0
    #im,jm,km = np.meshgrid(np.arange(J,dtype=int)+1,np.arange(J,dtype=int)+1,np.arange(J,dtype=int)+1)
    while((diff > tol) & (it < maxit)):
        u2[:,:]   = u1[:,:]
        u2[:,:]   = fBNC(f,u2)

        u1[1:J+1,1:J+1] = (0.25) * (u2[0:J,1:J+1]+u2[2:J+2,1:J+1] +u2[1:J+1,0:J]+u2[1:J+1,2:J+2]- (dt**2)*f)   
        it        = it+ 1
        diff      = get_rmsdiff(u1,u2)
        diff      = get_rmsdiff(u1,u2)
    #print('[jacobi]: it=%5i diff/tol=%13.5e' % (it,diff/tol))
        
    return u1[1:J+1,1:J+1]
  
def get_rmsdiff(u1,u2):
    J = (u1.shape)[0]
    return np.sqrt(np.mean((u1[1:J+1,1:J+1]-u2[1:J+1,1:J+1])**2))
    
def mg_restrict(u):
        #should be goodfor 3D
        #cell-centered

    J     = u.shape[0]
    i,j   = np.meshgrid(np.arange(J//2),np.arange(J//2))
    u    = 0.25*(u[2*i,2*j]+u[2*i,2*j+1]+u[2*i+1,2*j]+u[2*i+1,2*j+1]) #$0.125 vs 0.25
    return u
  
def mg_prolong(u,fBNC):
    #probably good for 3d now
    #cell-centered
    
    J               = u.shape[0] #size of J
    u1              = np.zeros((J+2,J+2)) #init
    u1[1:J+1,1:J+1] = u #leaves the frame for BC's
    u1              = fBNC(u1,u1) # note that the first argument is a dummy argument as of 27-9 fBNC is not 3D yet
    #i               = np.arange(J,dtype=int)#makes grid for prolonging
    #j               = np.arange(J,dtype=int)#makes grid for prolonging
    i,j             = np.meshgrid(np.arange(J,dtype=int),np.arange(J,dtype=int))#makes grid for prolonging
    
    #print(i,j)
    mi              = (u1[1:J+2,1:J+1]-u1[0:J+1,1:J+1]) #avg x
    mj              = (u1[1:J+1,1:J+2]-u1[1:J+1,0:J+1]) #avg y
    uf              = np.zeros((2*J,2*J)) #new fine matrix
    #print(mi,'\n' ,mi[0:J,:])
    #print(mj,'\n' ,mj[:,0:J])
    #print(uf)
    #print(uf[2*i,2*j])
    
    # [ijk1] [ij1k1]
    # [i1j51][i1j1k1]    
    #/              /
    #[ijk] [ij1k]  /
    #[i1jk][i1j1k]/
    
   
    uf[2*i  ,2*j  ] = u - 0.25*mj[:,0:J  ] - 0.25*mi[0:J  ,:] #even i even j 
    uf[2*i  ,2*j+1] = u - 0.25*mj[:,0:J  ] + 0.25*mi[1:J+1,:] #even i odd  j
    uf[2*i+1,2*j  ] = u + 0.25*mj[:,1:J+1] - 0.25*mi[0:J  ,:] #odd  i even j
    uf[2*i+1,2*j+1] = u + 0.25*mj[:,1:J+1] + 0.25*mi[1:J+1,:] #odd  i odd  j
    return uf    
       
def mg_residual(u,f,fBNC):
    #should be goodfor 3D
    #should be cell-centered
    J               = f.shape[0]
    u1              = np.zeros((J+2,J+2))
    u1[1:J+1,1:J+1] = u
    u1              = fBNC(f,u1)  
    r               = f[:,:] - (u1[0:J,1:J+1]+u1[2:J+2,1:J+1]+u1[1:J+1,0:J]+u1[1:J+1,2:J+2]-4.0*u1[1:J+1,1:J+1])/(dt**2)
    return r     


def mg_vcycle(f,fBNC,npre,npst,level,**kwargs):
    for key in kwargs:
        if(key=='u0'):
            u0=kwargs[key]
        elif(key=='maxxit'):
            maxxit=kwargs[key] 
        #elif(key=='verbose'):
            #verbose=kwargs[key]
    J=f.shape[0]
    
   
    #print('at', J)
    if(level>1):
        if(J==N):
            vh   = jacobi(f,fBNC,tol,u0=u0,maxxit=npre)
        else:
            vh   = jacobi(f,bnc_nocharge,tol,u0=u0,maxxit=npre)
        rh   = mg_residual(vh,f,fBNC)
       
         #print('restrict to', J//2)
        f2h  = mg_restrict(rh)
        v2h = np.zeros((J//2,J//2))
        v2h = mg_vcycle(f2h,bnc_nocharge,npre,npst,level-1,u0=v2h)
        
        #print('prolong to', J) 
        vh   = vh + mg_prolong(v2h,fBNC)
        u   = jacobi(f,fBNC,tol,u0=vh,maxxit=npst)
        
    if (level==1):
        #print('lowest level', J)
        u   = jacobi(f,fBNC,tol,u0=u0,maxxit=npst)
    
    return(u)
    
def multigrid(f,fBNC,tol):
    #should be goodfor 3D
    npre = 10
    npst = 80
    J    = f.shape[0]
    ui = np.zeros((J,J))
    if (J % 2 == 1):
        print('[multigrid]: J must be even: J=%4i' % (J))
        exit()
    #TEMPORARY LINE BELOW 
    level=np.log2(J)
    
    u    = mg_vcycle(f,fBNC,npre,npst,level,u0=ui, verbose=1)
    #print(np.amin(u))

    for i in range(20):
        u = mg_vcycle(f,fBNC,npre,npst,level,u0=u,verbose=0)
        #print(np.max(u))
        print(i)
    return u    
    
def bnc_none(f,u):
    return(u) 
def bnc_monopole(f,u):
    J  =  u.shape[0]-2
    if(J==N):
        u[0  ,0:J+2]  = UU[0  ,0:J+2]
        u[J+1,0:J+2]  = UU[J+1,0:J+2]
        u[1:J+2,0  ]  = UU[1:J+2,0  ]
        u[1:J+1,J+1]  = UU[1:J+1,J+1]
    else:
        u=bnc_nocharge(f,u)
    return u  
def bnc_nocharge(f,u):
    J                = u.shape[0]-2
    
    u[0:J+2,0    ] = -u[0:J+2,1]
    u[0:J+2,J+1  ] = -u[0:J+2,J]
    u[0    ,0:J+2] = -u[1,0:J+2]
    u[J+1  ,0:J+2] = -u[J,0:J+2]
    return u
    
def Usphere(xarr,yarr,bf,N):
    RHS=np.zeros((N+2,N+2))
    rad=2 #arbitrary
    m=50
    b=(Bmax-Bmin)*0.5
    B=(Bmax+Bmin)*0.5
    for i in range(0,N+1):
        xrhs=xarr[i]
        for j in range(0,N+1):
            yrhs=yarr[j]
      
            R=np.sqrt(xrhs**2+yrhs**2)
            
            R= (-np.tanh(m*R-m*rad))*b+B
            RHS[i,j]=R

    NRHS=RHS[1:-1,1:-1]
    return(NRHS)       
    
def init(problem,N):
    x = np.linspace(-ghostmax, ghostmax, N+2)
    y = np.linspace(-ghostmax, ghostmax, N+2)
    if(problem == 'sphere'):
        #bc=bnc_monopole
        bc = bnc_monopole
        rhs = Usphere(x,y,0,N)
    else:
        print('problem not in 2d code')
    return(rhs,x,y,bc)

 
def analytic(x,y,rad,extm):
    r = np.sqrt(x**2 + y** 2)
    uan = np.log(r)
    
    return(uan)

    
    
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

bound=10.
dt=(2*bound)/N
ghostmax=bound+0.5*dt

d=10. #seperation between "coils"
h=2. #width of tanh curve

Bmax=5.
Bmin=0.
#Bset=[Bmin,Bmax]

nrhs,x,y,bc  =init(problem,N)   

 
if(bc == bnc_monopole):
    print('calculating monopole boundaries')
    
    J               = N
    UU=np.zeros((J+2,J+2))
    G = -1

    for i in range(0,J+2):
            
            r  =  G/np.sqrt(x[i]**2+y[0]**2)
            UU[i,0] = r  #side 3 
            
    for i in range(0,J+2):
            r  =  G/np.sqrt(x[i]**2+y[J+1]**2)
            UU[i,J+1] = r
         
    for j in range(1,J+1):
         
        r  =  G/np.sqrt(x[0]**2+y[j]**2)
        UU[0,j] = r  #side 5
       
    for j in range(1,J+1):

        r = G/np.sqrt(x[J+1]**2+y[j]**2)
        UU[J+1,j] = r
        del r

    print("done!")
else: 
    print('not doing monopole boundaries for mg')
    
    
X=x[1:N+1]
Y=y[1:N+1]
xg, yg= np.meshgrid(X,Y,indexing='ij')


fig=plt.figure()

#for i in range(0,N):

ax=fig.gca(projection='3d')
ax.plot_surface(xg,yg,nrhs, cmap='bone')
ax.set_ylabel('x')
ax.set_xlabel('y')
ax.set_zlabel('rhs amp')
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
sol     =   multigrid(nrhs,bc,tol) 
print('done!')
print('doing fourier')
sol1    =   fourier(nrhs,bnc_none,tol)
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
ax=fig.gca(projection='3d')
ax.plot_surface(xg,yg,sol,cmap='bone')
ax.set_ylabel('x')
ax.set_xlabel('y')
ax.set_zlabel('sol amp')
plt.title('Sol Multigrid')
plt.show()
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(xg,yg,sol1, cmap='bone')
ax.set_ylabel('x')
ax.set_xlabel('y')
ax.set_zlabel('sol amp')
plt.title('Sol Fourier')
plt.show()
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(xg,yg,dif, cmap='bone')
ax.set_ylabel('x')
ax.set_xlabel('y')
ax.set_zlabel('sol amp')
plt.title('Muligrid-Fourier')
plt.show()

print(sol[:,0]-1/X)
print(sol[:,0],1/X)