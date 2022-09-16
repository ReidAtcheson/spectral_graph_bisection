import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import pdb


#Computes graph laplacian for sparsity structure of input
#sparse matrix
def graph_laplacian(A):
    #Symmetrize the input graph
    A=A+A.T
    m,n=A.shape

    rs=[]
    cs=[]
    vals=[]
    A=sp.csr_matrix(A)
    cids=A.indices
    offs=A.indptr

    for r,(beg,end) in enumerate(zip(offs[0:m],offs[1:m+1])):
        deg=0.0
        for c in cids[beg:end]:
            if not r==c:
                rs.append(r)
                cs.append(c)
                vals.append(-1.0)
                deg=deg+1.0
        rs.append(r)
        cs.append(r)
        vals.append(deg)
    return sp.coo_matrix((vals,(rs,cs)))


#Computes fiedler vector via inverse iteration using
#conjugate gradients as underlying solver
#Each step of iteration we project out the nullspace of G

def fiedler(G,tol=1e-6,maxiter=20,seed=42,verbosity=0):
    m,n=G.shape
    assert(m==n)
    rng=np.random.default_rng(seed)
    k=5
    X=rng.uniform(-1,1,size=(m,k))
    w,V=spla.lobpcg(G,X,Y=(np.ones(m)/np.sqrt(m)).reshape((m,1)),tol=tol,maxiter=maxiter,largest=False,verbosityLevel=verbosity)
    ids=np.argsort(w)
    w=w[ids]
    V=V[:,ids]
    return w[0],V[:,0]



#Uses fiedler vector to compute a 3-way partition [b0,s,b1] where s is an exact separator
#between b0 and b1
def split(G,tol=1e-6,maxiter=20,seed=42,verbosity=0):
    G=sp.csr_matrix(G)
    m,n=G.shape
    assert(m==n)
    t,f=fiedler(G,tol=tol,maxiter=maxiter,seed=seed,verbosity=verbosity)
    f=f/np.linalg.norm(f,ord=np.inf)
    b0=set(filter(lambda i : f[i]<0.0,range(0,m)))
    b1=set(filter(lambda i : f[i]>=0.0,range(0,m)))
    s0=set()
    s1=set()
    for i in b0:
        cids=G.indices
        offs=G.indptr
        beg,end=offs[i],offs[i+1]
        for c in cids[beg:end]:
            if c in b1:
                s0.add(c)

    b1 = b1.difference(s0)
    for i in b1:
        cids=G.indices
        offs=G.indptr
        beg,end=offs[i],offs[i+1]
        for c in cids[beg:end]:
            if c in b0:
                s1.add(c)


    b0 = b0.difference(s1)
    s = s0.union(s1)


    return (list(b0),list(s),list(b1))






def ndisect(G,ids,maxm=32,tol=1e-6,maxiter=20,seed=42,verbosity=0):
    G=sp.lil_matrix(G)
    m,n=G.shape
    assert(m==n)
    if m<=maxm:
        return ids
    else:
        b0,s,b1 = split(G,tol=tol,maxiter=maxiter,seed=seed,verbosity=verbosity)
        G0=graph_laplacian(G[np.ix_(b0,b0)])
        G1=graph_laplacian(G[np.ix_(b1,b1)])

        return (
                ndisect(G0,[ids[i] for i in b0],maxm=maxm,tol=tol,maxiter=maxiter,seed=seed,verbosity=verbosity),
                [ids[i] for i in s],
                ndisect(G1,[ids[i] for i in b1],maxm=maxm,tol=tol,maxiter=maxiter,seed=seed,verbosity=verbosity)
                )


def level_flatten(nd):
    stack=[]
    parents=[]
    offs=[]
    parent=-1
    queue=deque([(nd,parent)])

    while queue:
        n,p=queue.popleft()
        if isinstance(n,list):
            stack.append(n)
            parents.append(p)
        else:
            left,sep,right=n
            stack.append(sep)
            parents.append(p)
            queue.append((left,len(stack)-1))
            queue.append((right,len(stack)-1))

    off=0
    offs.append(off)
    for n in stack:
        off=off+len(n)
        offs.append(off)
    return stack,parents,offs


def assemble(A,ids,parents,offs):
    m,n=A.shape
    assert(m==n)
    B = sp.lil_matrix((m,n))
    for n,p,j in zip(reversed(ids),reversed(parents),reversed(range(0,len(ids)))):
        start=offs[j]
        stop=offs[j+1]
        rn=range(start,stop)
        #First assemble local block
        B[np.ix_(rn,rn)]=A[np.ix_(n,n)]
        #Now assemble off-diagonal blocks
        while not p==-1:
            rp=range(offs[p],offs[p+1])
            mp=ids[p]
            B[np.ix_(rn,rp)]=A[np.ix_(n,mp)]
            B[np.ix_(rp,rn)]=A[np.ix_(mp,n)]
            p=parents[p]
    return B





