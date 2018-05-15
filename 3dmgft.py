import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import animation
from scipy.fftpack import fftn
from scipy.fftpack import ifftn
from scipy.fftpack import fftfreq
import time




def jacobi(f,fBNC,tol,**kwargs): #actually jacobi
    maxit = 10000
    s = f.shape
    if (s[0] != s[1]):
        print('[jacobi]: need square matrix.')
        exit()
    J         = s[0]
    u1        = np.zeros((J+2,J+2,J+2)) # initial guess.
    u2        = np.zeros((J+2,J+2,J+2))
    for key in kwargs:
        if (key == 'maxit'):
            maxit = kwargs[key]
        if (key == 'u0'):
            u1[1:J+1,1:J+1,1:J+1] = kwargs[key]
            u2[1:J+1,1:J+1,1:J+1] = kwargs[key]
    u1        = fBNC(f,u1)
    diff      = 1e30
    it        = 0
    #im,jm,km = np.meshgrid(np.arange(J,dtype=int)+1,np.arange(J,dtype=int)+1,np.arange(J,dtype=int)+1)
    while(it < maxit):
        u2[:,:,:]   = u1[:,:,:]
        u2[:,:,:]   = fBNC(f,u2)
        
       #u1[im,jm,km] = (1/6)*(u2[im-1,jm,km]+u2[im+1,jm,km]+u2[im,jm-1,km]+u2[im,jm+1,km]+u2[im,jm,km+1]+u2[im,jm,km-1] - (dt**2)*f[im-1,jm-1,km-1])
        u1[1:J+1,1:J+1,1:J+1] = (1/6) * (u2[0:J,1:J+1,1:J+1]+u2[2:J+2,1:J+1,1:J+1] +u2[1:J+1,0:J,1:J+1]+u2[1:J+1,2:J+2,1:J+1]+u2[1:J+1,1:J+1,0:J]+u2[1:J+1,1:J+1,2:J+2]- (dt**2)*f)   
        it        = it+ 1
        #diff      = get_rmsdiff(u1,u2)
        #print('[jacobi]: it=%5i diff/tol=%13.5e' % (it,diff/tol))
        
    return u1[1:J+1,1:J+1,1:J+1]
    
def fourier(f,fBNC,tol):
    J   =   f.shape[0]
    Jp  = 2*J 
    #Jp  = 3*J 
    RHS =   np.zeros((Jp,Jp,Jp))
 
    RHS[J//2 : 3*J//2   ,J//2 : 3*J//2  ,J//2 : 3*J//2]   =   f
    #RHS[J:2*J,J:2*J,J:2*J]=f
    #================================
    Fcoeff  =   fftn(RHS)
    freq    =   np.zeros((Jp))
    for n in range(0,Jp//2):
        freq[n]=n
    for n in range(Jp//2,Jp):
        freq[n]=-Jp-n
    
    
    #print(freq.shape, "freq shape \n", Jp, "Jp \n" , freq, "freq")
  
    
    Fcoeff[0,0,0]=0
    
    #X AXIS
    for i in range(1,Jp):
            eigi  =   np.cos(2*freq[i] * np.pi / (Jp) ) - 1
            
            eig              =    2*(eigi)
            Fcoeff[i,0,0]    =   (dt**2)* Fcoeff[i,0,0] / eig
    #Y AXIS
    for j in range(1,Jp):
            eigj  =   np.cos( 2*freq[j] * np.pi / (Jp) ) - 1
            
            eig              =    2*(eigj)
            Fcoeff[0,j,0]    =   (dt**2)* Fcoeff[0,j,0] / eig
    
    #Z AXIS
    for k in range(1,Jp):
            eigk  =   np.cos(2* freq[k] * np.pi / (Jp) ) - 1
            
            eig              =    2*(eigk)
            Fcoeff[0,0,k]    =   (dt**2)* Fcoeff[0,0,k] / eig            
    
    #XY PLANE
    for i in range(1,Jp):
        eigi =  np.cos(2* freq[i] * np.pi / (Jp) )  - 1
        for j in range(1,Jp):
            eigj =  np.cos(2* freq[j] * np.pi / (Jp) )  - 1
            
            eig              =    2*(eigi  +  eigj)
            Fcoeff[i,j,0]    =   (dt**2)* Fcoeff[i,j,0] / eig
            
    #XZ PLANE
    for i in range(1,Jp):
        eigi =  np.cos(2* freq[i] * np.pi / (Jp) )  - 1
        for k in range(1,Jp):
            eigk  =   np.cos(2* freq[k] * np.pi / (Jp) ) - 1
            
            eig              =    2*(eigi  +  eigk)
            Fcoeff[i,0,k]    =   (dt**2)* Fcoeff[i,0,k] / eig
    
    #YZ PLANE
    for j in range(1,Jp):
        eigj =  np.cos(2* freq[j] * np.pi / (Jp) )  - 1
        for k in range(1,Jp):
            eigk  =   np.cos(2* freq[k] * np.pi / (Jp) ) - 1
            
            eig              =    2*(eigj  +  eigk)
            Fcoeff[0,j,k]    =   (dt**2)* Fcoeff[0,j,k] / eig    
     
    #FULL
    for i in range(1,Jp):
        
        eigi =  np.cos(2* freq[i] * np.pi / (Jp) )  - 1
 
        for j in range(1,Jp):
            
            eigj =  np.cos( 2*freq[j] * np.pi / (Jp) )  - 1       
            for k in range(1,Jp):
            
                eigk  =   np.cos(2* freq[k] * np.pi / (Jp) ) - 1
                
                eig              =    2*(eigi  +  eigj  +  eigk)
              
                Fcoeff[i,j,k]    =   (dt**2)* Fcoeff[i,j,k] / eig
    
    #Fcoeff=np.nan_to_num(Fcoeff)
   
   
    
    
    #==============================
    u=ifftn(Fcoeff)
    #==============================
    u=u.real
    #u=fBNC(f,u)
    rhs=u[J//2 : 3*J//2,  J//2 : 3*J//2,    J//2 : 3*J//2]
    #rhs=u[J:2*J,J:2*J,J:2*J]*(8/Jp**3)
    
    return(rhs)
    
def get_rmsdiff(u1,u2):
    J = u1.shape[0]
    return np.sqrt(np.mean((u1[1:J+1,1:J+1,1:J+1]-u2[1:J+1,1:J+1,1:J+1])**2))
    
def mg_restrict(u):
        #should be goodfor 3D
        #cell-centered

    J     = u.shape[0]
    i,j,k   = np.meshgrid(np.arange(J//2),np.arange(J//2),np.arange(J//2))
    u    = 0.125*(u[2*i,2*j,2*k]+u[2*i,2*j,2*k+1]+u[2*i,2*j+1,2*k]+u[2*i,2*j+1,2*k+1]+u[2*i+1,2*j,2*k]+u[2*i+1,2*j,2*k+1]+u[2*i+1,2*j+1,2*k]+u[2*i+1,2*j+1,2*k+1])
    return u
  
def mg_prolong(u,fBNC):
    #probably good for 3d now
    #cell-centered
    
    J               = u.shape[0] #size of J
    u1              = np.zeros((J+2,J+2,J+2)) #init
    u1[1:J+1,1:J+1,1:J+1] = u #leaves the frame for BC's
    u1              = fBNC(u1,u1) # note that the first argument is a dummy argument as of 27-9 fBNC is not 3D yet
    i,j,k             = np.meshgrid(np.arange(J,dtype=int),np.arange(J,dtype=int),np.arange(J,dtype=int)) #makes grid for prolonging
   
    mi              = 0.5*(u1[1:J+2,1:J+1,1:J+1]-u1[0:J+1,1:J+1,1:J+1]) #avg x
    mj              = 0.5*(u1[1:J+1,1:J+2,1:J+1]-u1[1:J+1,0:J+1,1:J+1])   #avg y
    mk              = 0.5*(u1[1:J+1,1:J+1,1:J+2]-u1[1:J+1,1:J+1,0:J+1]) #3d avergages
    """ 
    mi              = 0.5*(u1[2:J+2,1:J+1,1:J+1]-u1[0:J,1:J+1,1:J+1]) #avg x
    mj              = 0.5*(u1[1:J+1,2:J+2,1:J+1]-u1[1:J+1,0:J,1:J+1])   #avg y
    mk              = 0.5*(u1[1:J+1,1:J+1,2:J+2]-u1[1:J+1,1:J+1,0:J]) #3d avergages
    """
    uf              = np.zeros((2*J,2*J,2*J)) #new fine matrix
    
    # [ijk1] [ij1k1]
    # [i1j51][i1j1k1]    
    #/              /
    #[ijk] [ij1k]  /
    #[i1jk][i1j1k]/
    uf[2*i  ,2*j  ,2*k] = u + (0.25)*mj[:,1:J+1,:] + (0.25)*mi[1:J+1,:,:] + (0.25)*mk[:,:,1:J+1]
    uf[2*i  ,2*j+1,2*k] = u + (0.25)*mj[:,1:J+1,:] + (0.25)*mi[1:J+1,:,:] - (0.25)*mk[:,:,0:J]
    uf[2*i+1,2*j  ,2*k] = u + (0.25)*mj[:,1:J+1,:] - (0.25)*mi[0:J,  :,:] + (0.25)*mk[:,:,1:J+1] 
    uf[2*i+1,2*j+1,2*k] = u + (0.25)*mj[:,1:J+1,:] - (0.25)*mi[0:J,  :,:] - (0.25)*mk[:,:,0:J] 

    uf[2*i  ,2*j  ,2*k+1] = u - (0.25)*mj[:,0:J,:] + (0.25)*mi[1:J+1,:,:] + (0.25)*mk[:,:,1:J+1] 
    uf[2*i  ,2*j+1,2*k+1] = u - (0.25)*mj[:,0:J,:] + (0.25)*mi[1:J+1,:,:] - (0.25)*mk[:,:,0:J] 
    uf[2*i+1,2*j  ,2*k+1] = u - (0.25)*mj[:,0:J,:] - (0.25)*mi[0:J,  :,:] + (0.25)*mk[:,:,1:J+1] 
    uf[2*i+1,2*j+1,2*k+1] = u - (0.25)*mj[:,0:J,:] - (0.25)*mi[0:J,  :,:] - (0.25)*mk[:,:,0:J] 
   
    """
    uf[2*i  ,2*j  ,2*k] = u + (0.25)*mj + (0.25)*mi + (0.25)*mk 
    uf[2*i  ,2*j+1,2*k] = u + (0.25)*mj + (0.25)*mi - (0.25)*mk 
    uf[2*i+1,2*j  ,2*k] = u + (0.25)*mj - (0.25)*mi + (0.25)*mk 
    uf[2*i+1,2*j+1,2*k] = u + (0.25)*mj - (0.25)*mi - (0.25)*mk 

    uf[2*i  ,2*j  ,2*k+1] = u - (0.25)*mj + (0.25)*mi + (0.25)*mk 
    uf[2*i  ,2*j+1,2*k+1] = u - (0.25)*mj + (0.25)*mi - (0.25)*mk 
    uf[2*i+1,2*j  ,2*k+1] = u - (0.25)*mj - (0.25)*mi + (0.25)*mk 
    uf[2*i+1,2*j+1,2*k+1] = u - (0.25)*mj - (0.25)*mi - (0.25)*mk 
    """
    return uf    
       
def mg_residual(u,f,fBNC):
    #should be goodfor 3D
    #should be cell-centered
    J               = f.shape[0]
    u1              = np.zeros((J+2,J+2,J+2))
    u1[1:J+1,1:J+1,1:J+1] = u
    u1              = fBNC(f,u1)  
    r               = f[:,:,:] - (u1[0:J,1:J+1,1:J+1]+u1[2:J+2,1:J+1,1:J+1]+u1[1:J+1,0:J,1:J+1]+u1[1:J+1,2:J+2,1:J+1]+u1[1:J+1,1:J+1,0:J]+u1[1:J+1,1:J+1,2:J+2]-6.0*u1[1:J+1,1:J+1,1:J+1])/(dt**2)
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
        v2h = np.zeros(f2h.shape)
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
    npst = 20
    J    = f.shape[0]
    ui = np.zeros((J,J,J))
    if (J % 2 == 1):
        print('[multigrid]: J must be even: J=%4i' % (J))
        exit()
    #TEMPORARY LINE BELOW 
    level=int(np.log2(J))
    
    u    = mg_vcycle(f,fBNC,npre,npst,level,u0=ui, verbose=1)
    #print(np.amin(u))
    print('first cycle')

    for i in range(10):
        u = mg_vcycle(f,fBNC,npre,npst,level,u0=u,verbose=0)
        #print(np.max(u))
        print(i)
    return u    


    
def bnc_nocharge(f,u):
    J                = u.shape[0]-2
    
    u[0:J+2,0:J+2,0  ] = -u[0:J+2,0:J+2,1  ] 
    u[0:J+2,0:J+2,J+1] = -u[0:J+2,0:J+2,J  ]
    
    u[0:J+2,0  ,1:J+1] = -u[0:J+2,1  ,1:J+1]
    u[0:J+2,J+1,1:J+1] = -u[0:J+2,J  ,1:J+1]
    
    u[0  ,1:J+1,1:J+1] = -u[1  ,1:J+1,1:J+1]      
    u[J+1,1:J+1,1:J+1] = -u[J  ,1:J+1,1:J+1]      
    return u
   
    
def bnc_monopole(f,u):
    J               = u.shape[0]-2
    if(J==N):
        temp =  J+1
        for i in range(0,J+2):
            for j in range(0,J+2):
                u[i,j,0] = UU[i,j] # xy plane zf full square
                            
        temp = 0        
        for i in range(0,J+2):
            for j in range(0,J+2):
                u[i,j,J+1] = UU[i,j]  #xy plane z0 full square
                      
        temp = J+1
        for j in range(0,J+2):
            for k in range(1,J+1):
                u[J+1,j,k] = UU[j,k]  #yz plane xf rec
                       
        temp = 0
        for j in range(0,J+2):
            for k in range(1,J+1):
                u[0,j,k] = UU[j,k]         #yz plane x0 rec
                
        temp = J+1       
        for i in range(1,J+1):
            for k in range(1,J+1):
                u[i,J+1,k] = UU[i,k]  #xz plane yf small sqr
                       
        temp = 0    
        for i in range(1,J+1):
            for k in range(1,J+1):
                u[i,0,k] = UU[i,k] # xz plane y0 small sqr
    else:
        temp =  J+1
        for i in range(0,J+2):
            for j in range(0,J+2):
                u[i,j,J+1] = -u[i,j,J] # xy plane zf full square
                            
        temp = 0        
        for i in range(0,J+2):
            for j in range(0,J+2):
                u[i,j,0] = -u[i,j,1]  #xy plane z0 full square
                      
        temp = J+1
        for j in range(0,J+2):
            for k in range(1,J+1):
                u[J+1,j,k] = -u[J,j,k]  #yz plane xf rec
                       
        temp = 0
        for j in range(0,J+2):
            for k in range(1,J+1):
                u[0,j,k] = -u[1,j,k]         #yz plane x0 rec
                
        temp = J+1       
        for i in range(1,J+1):
            for k in range(1,J+1):
                u[i,J+1,k] =  -u[i,J,k] #xz plane yf small sqr
                       
        temp = 0    
        for i in range(1,J+1):
            for k in range(1,J+1):
                u[i,0,k] = - u[i,1,k] # xz plane y0 small sqr
        
    return(u)    
    
def bnc_none(f,u):
    return(u)  
def bnc_periodic(f,u):

    J               = u.shape[0]-2
    D1 = u[0:J+2,0:J+2,J+1]   #side 1 flat bottom #side 1 flat bottom
    D2 = u[0:J+2,0:J+2,0  ]   #side 2 flat top
    D3 = u[J+1    ,0:J+2,1:J+1]  #side 3 
    D4 = u[0  ,0:J+2,1:J+1]  #side 4
    D5 = u[1:J+1,J+1    ,1:J+1]  #side 5
    D6 = u[1:J+1,0  ,1:J+1]
    
    
    
    
    u[0:J+2,0:J+2,    0] = D1 #side 1 flat bottom
    u[0:J+2,0:J+2,J+1  ] = D2 #side 2 flat top
    u[0    ,0:J+2,1:J+1] = D3 #side 3 
    u[J+1  ,0:J+2,1:J+1] = D4 #side 4
    u[1:J+1,0    ,1:J+1] = D5 #side 5
    u[1:J+1,J+1  ,1:J+1] = D6 #side 6
    return u    
def bnc_grad(f,u):
    
    J               = u.shape[0]-2
    D1 = np.zeros((J+2,J+2)) #side 1 flat bottom
    D2 = np.zeros((J+2,J+2)) #side 2 flat top
    D3 = np.zeros((J+2,J))   #side 3 
    D4 = np.zeros((J+2,J))   #side 4
    D5 = np.zeros((J,J))     #side 5
    D6 = np.zeros((J,J))     #side 6
    
    
    m1=     (u[0:J+2,0:J+2,    2]  -  u[0:J+2,0:J+2,    1])
    m2=     (u[0:J+2,0:J+2,J    ]  -  u[0:J+2,0:J+2,  J-1])
    m3=     (u[2    ,0:J+2,1:J+1]  -  u[1    ,0:J+2,1:J+1])
    m4=     (u[J    ,0:J+2,1:J+1]  -  u[J-1  ,0:J+2,1:J+1])
    m5=     (u[1:J+1,2    ,1:J+1]  -  u[1:J+1,1    ,1:J+1])
    m6=     (u[1:J+1,J    ,1:J+1]  -  u[1:J+1,J-1  ,1:J+1])
    
    
    
    
    
    u[0:J+2,0:J+2,    0] = u[0:J+2,0:J+2,    1]  -  m1 #side 1 flat bottom full
    u[0:J+2,0:J+2,J+1  ] = u[0:J+2,0:J+2,J    ]  +  m2 #side 2 flat top full
    u[0    ,0:J+2,1:J+1] = u[1    ,0:J+2,1:J+1]  -  m3 #side 3 front face wide
    u[J+1  ,0:J+2,1:J+1] = u[J    ,0:J+2,1:J+1]  +  m4 #side 4 back face wide
    u[1:J+1,0    ,1:J+1] = u[1:J+1,1    ,1:J+1]  -  m5 #side 5 left face small
    u[1:J+1,J+1  ,1:J+1] = u[1:J+1,J    ,1:J+1]  +  m6 #side 6 right face small
    
    
    return u 
def bnc_zero(f,u):
    J               = u.shape[0]-2
    D1 = np.zeros((J+2,J+2)) #side 1 flat bottom
    D2 = np.zeros((J+2,J+2)) #side 2 flat top
    D3 = np.zeros((J+2,J))   #side 3 
    D4 = np.zeros((J+2,J))   #side 4
    D5 = np.zeros((J,J))     #side 5
    D6 = np.zeros((J,J))     #side 6
    
    u[0:J+2,0:J+2,    0] = D1 #side 1 flat bottom
    u[0:J+2,0:J+2,J+1  ] = D2 #side 2 flat top
    u[0    ,0:J+2,1:J+1] = D3 #side 3 
    u[J+1  ,0:J+2,1:J+1] = D4 #side 4
    u[1:J+1,0    ,1:J+1] = D5 #side 5
    u[1:J+1,J+1  ,1:J+1] = D6 #side 6
    return u    
def Bfield(x,y,z):
    [Bmin,Bmax]=Bset
    r=np.sqrt(x**2+y**2)
    phi=np.arctan2(y,x)
    height=0.5*(Bmax-Bmin)
    Bz=height*np.tanh(-z*h-d)*np.tanh(-z*h+d)+0.5*(Bmax+Bmin)
    z=z+0.000000000001
    Br=-(height*h*np.tanh(d+h*z)*((np.cosh(d-h*z))**-2)-height*h*np.tanh(d-h*z)*((np.cosh(d+h*z)**-2)))
    Br=Br*r
    Bx=Br*np.cos(phi)
    By=Br*np.sin(phi)	
    return(Bx,By,Bz)
 
def cosfield(x,y,z,bf,N):
    RHS=np.zeros((N+2,N+2,N+2))

    for i in range(1,N+1):
        for j in range (1,N+1):
            for k in range(1,N+1):
                
                A=np.zeros((7,3)) #full stencil x,y,z
                A[0]=bf(x[i],y[j],z[k])   # 0, 0, 0
                A[1]=bf(x[i+1],y[j],z[k]) # 1, 0, 0
                A[2]=bf(x[i-1],y[j],z[k]) #-1, 0, 0
                A[3]=bf(x[i],y[j+1],z[k]) # 0, 1, 0
                A[4]=bf(x[i],y[j-1],z[k]) # 0,-1, 0
                A[5]=bf(x[i],y[j],z[k+1]) # 0, 0, 1
                A[6]=bf(x[i],y[j],z[k-1]) # 0, 0,-1

                             
                d2bx2=(A[1,:]-2*A[0,:]+A[2,:])*(dt**-2)
                
                d2by2=(A[3,:]-2*A[0,:]+A[4,:])*(dt**-2)
    
               
                d2B=d2by2+d2bx2
                
                Tension=-A[0,2]*d2B[2]
                
                dbzx=(A[1,2]-A[2,2])/(2*dt)
                dbzy=(A[3,2]-A[4,2])/(2*dt)
                Pressure=(dbzy)**2 +(-dbzx)**2 
                RHS[i,j,k]=Tension-Pressure
    

    return(RHS)
 
def Usphere(xarr,yarr,zarr,bf,N):
    RHS=np.zeros((N+2,N+2,N+2))
    rad=2.5 #arbitrary
    m=50
    b=(Bmax-Bmin)*0.5
    B=(Bmax+Bmin)*0.5
    for i in range(0,N+1):
        xrhs=xarr[i]
        for j in range(0,N+1):
            yrhs=yarr[j]
            for k in range(0,N+1):
                zrhs=zarr[k]
      
                R=np.sqrt(xrhs**2+yrhs**2+zrhs**2)
                
                R= (-np.tanh(m*R-m*rad))*b+B
                #Bx= (-np.tanh(m*R-m*rad**2))*b+B
                #By= (-np.tanh(m*R-m*rad**2))*b+B
                #Bz= (-np.tanh(m*R-m*rad**2))*b+B
                #R=np.sqrt(Bx**2+By**2+Bz**2)
                
                RHS[i,j,k]=R

    NRHS=RHS[1:-1,1:-1,1:-1]
    return(NRHS)
 
def field_create(x,y,z,bf,N):

        #should be goodfor 3D

    RHS=np.zeros((N+2,N+2,N+2))
    for i in range(1,N+1):
        for j in range (1,N+1):
            for k in range(1,N+1):
                
                A=np.zeros((7,3)) #full stencil x,y,z
                A[0]=bf(x[i],y[j],z[k])   # 0, 0, 0 
                A[1]=bf(x[i+1],y[j],z[k]) # 1, 0, 0
                A[2]=bf(x[i-1],y[j],z[k]) #-1, 0, 0
                A[3]=bf(x[i],y[j+1],z[k]) # 0, 1, 0
                A[4]=bf(x[i],y[j-1],z[k]) # 0,-1, 0
                A[5]=bf(x[i],y[j],z[k+1]) # 0, 0, 1
                A[6]=bf(x[i],y[j],z[k-1]) # 0, 0,-1
                
                
               
                
                d2bx2=(A[1,:]-2*A[0,:]+A[2,:])*(dt**-2)
                
                d2by2=(A[3,:]-2*A[0,:]+A[4,:])*(dt**-2)
                
                d2bz2=(A[5,:]-2*A[0,:]+A[6,:])*(dt**-2)
               
                d2B=d2bz2+d2by2+d2bx2
                #print(d2B)
                
                Tension=-A[0,0]*d2B[0]-A[0,1]*d2B[1]-A[0,2]*d2B[2]
                
                dbxy=(A[3,0]-A[4,0])/(2*dt)
                dbxz=(A[5,0]-A[6,0])/(2*dt)
               
                
                dbyx=(A[1,1]-A[2,1])/(2*dt)
                dbyz=(A[5,1]-A[6,1])/(2*dt)
                
                dbzx=(A[1,2]-A[2,2])/(2*dt)
                dbzy=(A[3,2]-A[4,2])/(2*dt)
                
                Pressure=(dbzy-dbyz)**2 +(dbxz-dbzx)**2 + (dbyx-dbxy)**2
                
                
                RHS[i,j,k]=Tension-Pressure
                
                NRHS=RHS[1:-1,1:-1,1:-1]

    return(NRHS)    
        
def Bzcos(x,y,z):
    Bx=0
    By=0
    w=16
    Bz=np.cos(2*np.pi*x/w)*np.cos(2*np.pi*y/w)
    return(Bx,By,Bz)

    
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)  
parser.add_argument("prob",type=str,
                    help="source field:\n"
                        "    bottle   : magnetic bottle radial on z axis\n"
                        "    cosine   : cosine in xy plane \n"  
                        "    sphere   : solid spherical potential") 
parser.add_argument("points",type=int,
                    help="number of spatial support points in each dimension")
   
parser.add_argument("zcoordinate",type=int,
                    help=" xy slice on z axis between 0 and N:\n")
                    
parser.add_argument("tol",type=float,
                    help=" tolerance:\n")

                    
args        = parser.parse_args()
N           = args.points
problem     = args.prob       
zcoord      = args.zcoordinate             
tol             = args.tol

bound=10.
dt=(2*bound)/N
ghostmax=bound+0.5*dt

d=10. #seperation between "coils"
h=2. #width of tanh curve

Bmax=5.
Bmin=0.
Bset=[Bmin,Bmax]



def init(problem,N):
    x = np.linspace(-ghostmax, ghostmax, N+2)
    y = np.linspace(-ghostmax, ghostmax, N+2)
    z = np.linspace(-ghostmax, ghostmax, N+2)
    if (problem == 'bottle'):
        bc=bnc_nocharge
        rhs=field_create(x,y,z,Bfield,N)
    elif (problem == 'cosine'):
        bc  = bnc_periodic
        rhs = field_create(x,y,z,Bzcos,N)
    elif (problem == 'sphere'):
        bc=bnc_monopole
        rhs = Usphere(x,y,z,0,N)
    return(rhs,x,y,z,bc)
nrhs,x,y,z,bc  =init(problem,N)

if(bc == bnc_monopole):
    print('calculating monopole boundaries')
    
    J               = N
    UU=np.zeros((J+2,J+2))
    G = -1
    for i in range(0,J+2):
        for j in range(0,J+2):
        
            r  =  G/np.sqrt(x[i]**2+y[j]**2+z[0]**2)
            UU[i,j] = r #side 1 flat bottom
            
    print("done!")
else: 
    print('not doing monopole boundaries for mg')

       

    
X=x[1:N+1]
Y=y[1:N+1]
Z=z[1:N+1]
xg, yg= np.meshgrid(X,Y,indexing='ij')


fig=plt.figure()

#for i in range(0,N):

ax=fig.gca(projection='3d')
ax.plot_surface(xg,yg,nrhs[:,:,zcoord], cmap='bone')
ax.set_ylabel('x')
ax.set_xlabel('y')
ax.set_zlabel('rhs amp')
plt.title('RHS')
plt.show()
	
    
    

#====================================================

t=time.time()
#====================================================    
    
    
#===========================================================
#===========================================================
#===========================================================
#===========================================================
#===========================================================
print('doing fourier')
sol1    =   fourier(nrhs,bnc_none,tol)
print('done!')
print('doing multigrid')
sol     =   multigrid(nrhs,bc,tol) 
print('done!')

dif     =   sol -  sol1
mgmin = np.min(sol)
frmin = np.min(sol1)
print(mgmin,'mg min')
print(frmin,'fr min')

#================Adjust Fourier offset
offset=mgmin-frmin
print("adjusting for fourier %0.4f offset"%offset)
sol1=sol1+offset


#===========================================================
#===========================================================
#===========================================================
#===========================================================
#===========================================================



elapsed=time.time() - t
print(elapsed, 'time')
#exit()

fig = plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(xg,yg,nrhs[:,:,zcoord], cmap='bone')
ax.set_ylabel('x')
ax.set_xlabel('y')
ax.set_zlabel('rhs amp')
plt.title('RHS')
plt.show()
	


fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(xg,yg,sol[:,:,zcoord], cmap='bone')
ax.set_ylabel('x')
ax.set_xlabel('y')
ax.set_zlabel('sol amp')
plt.title('Sol Multigrid')
plt.show()


fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(xg,yg,sol1[:,:,zcoord], cmap='bone')
ax.set_ylabel('x')
ax.set_xlabel('y')
ax.set_zlabel('sol amp')
plt.title('Sol Fourier')
plt.show()

fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(xg,yg,dif[:,:,zcoord], cmap='bone')
ax.set_ylabel('x')
ax.set_xlabel('y')
ax.set_zlabel('sol amp')
plt.title('Muligrid-Fourier')
plt.show()



def slicer(i,zslice, line,solution, name):
    zslice = solution[:,:,i]
    ax.clear()
    name = name,i
    plt.title(name)
    plt.xlim((-bound,bound))
    plt.ylim((-bound, bound))
    ax.set_zlim((np.min(solution), 1.5*np.max(solution)))
    line =ax.plot_surface(xg,yg,zslice, cmap='bone')
    return line,

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
zslice = nrhs[:,:,0]
line =ax.plot_surface(xg,yg,zslice, cmap='bone')
name = 'rhs'
ani = animation.FuncAnimation(fig, slicer, fargs= (zslice,line,nrhs,name) , frames = N, blit=False)
plt.show()   
    
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
zslice = sol[:,:,0]
name = 'multigrid'
line =ax.plot_surface(xg,yg,zslice, cmap='bone')
ani = animation.FuncAnimation(fig, slicer, fargs= (zslice,line,sol,name) , frames = N, blit=False)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
zslice = sol[:,:,0]
name = 'fourier'
line =ax.plot_surface(xg,yg,zslice, cmap='bone')
ani = animation.FuncAnimation(fig, slicer, fargs= (zslice,line,sol1,name) , frames = N, blit=False)
plt.show()
 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
zslice = sol[:,:,0]
line =ax.plot_surface(xg,yg,zslice, cmap='bone')
name = 'mg - fourier diff'
ani = animation.FuncAnimation(fig, slicer, fargs= (zslice,line,dif,name) , frames = N, blit=False)
plt.show() 