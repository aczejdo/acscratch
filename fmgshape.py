import numpy as np

#fmg
def S(x):
    J=x.shape[0]
    print("smoothing! on %i"%J)
    return(x)
def R(x):
    J=x.shape[0]
    print("restricting! from %i to %i"%(J,J//2))
    return(np.zeros(np.array(x.shape)//2))
def P(x):
    J=x.shape[0]
    print("prolonging! from %i to %i"%(J,J*2))
    return(np.zeros(np.array(x.shape)*2))
    
    
    
    
def fmg(x,level):
    J=x.shape[0]
    #1
    if(level==1):
        e2h=S(x)

    else:
        eh=S(x)
        f2h=R(x)
        e2h=np.zeros(f2h.shape)
        e2h =fmg(f2h,level-1)
        
    #2

    eh=P(e2h)
    #3
    u=vcyc(eh,level+1)
        

    
    
    return(u)
def vcyc(f,level):
    
    J=f.shape[0]
    

    if(level>1):
        if(J==N):
            eh   = S(f)
        else:
            eh   = S(f)
        rh   = eh
        f2h  = R(rh)
        e2h = np.zeros(f2h.shape)
        e2h = vcyc(f2h,level-1)
        eh   = P(e2h)
        u   = S(f)
        
    if (level==1):
        u   = S(f)
    
    return(u)
N=16
X = np.zeros((N,N))
Level = np.log2(N)
X = fmg(X,Level)
