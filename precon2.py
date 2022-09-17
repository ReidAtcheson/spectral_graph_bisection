import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sbisect


seed=2398743
rng=np.random.default_rng(seed)

nsamples=100


dpits=[]
offdiags=[]
ranks=[]
for _ in range(0,nsamples):
    maxsep=rng.choice(range(5,64))
    maxm=32
    m=rng.choice(range(32*32,128*128))
    ndiags=6
    offs = rng.choice(range(-int(np.sqrt(m)),int(np.sqrt(m))),size=ndiags,replace=False)
    A=sp.diags([rng.choice([1e-2,1e-1,1.0,1e1,1e2])*rng.uniform(-1,1,m) for _ in offs],offs,shape=(m,m))
    diag=abs(A)@np.ones(m)+2.0
    A=sp.lil_matrix(A+sp.diags([diag],[0],shape=(m,m)))

    G=sbisect.graph_laplacian(A)
    p0,s,p1=sbisect.split(G,maxiter=1000,maxsep=maxsep)
    p=p0+p1+s


    Aph = sp.bmat(
            [
                [A[np.ix_(p0,p0)],            None,A[np.ix_(p0,s)]],
                [None,            A[np.ix_(p1,p1)],A[np.ix_(p1,s)]],
                [A[np.ix_(s,p0)], A[np.ix_(s,p1)] ,A[np.ix_(s,s)]]
            ]
            )

    eps0=A[np.ix_(p0,p1)]
    eps1=A[np.ix_(p1,p0)]

    r0=sp.csgraph.structural_rank(eps0)
    r1=sp.csgraph.structural_rank(eps1)
    r=(r0+r1)/m

    offdiag=(sp.linalg.norm(eps0) + sp.linalg.norm(eps1))/sp.linalg.norm(A)

    M=spla.splu(sp.csc_matrix(Aph))


    Ap=A[np.ix_(p,p)]

    b=rng.uniform(-1,1,size=m)
    bp=b[p]





    it=0
    res=[]
    def callback(xk):
        global res
        xp=xk.copy()
        xp[p]=xk
        r=b-A@xp
        res.append(np.linalg.norm(r,ord=np.inf))


    def evalA(x):
        global it
        it=it+1
        return Ap@x



    spla.gmres(spla.LinearOperator((m,m),matvec=evalA),bp,M=spla.LinearOperator((m,m),matvec=M.solve),callback=callback,callback_type="x",tol=1e-14,restart=10,maxiter=100)

    digits_per_it=np.abs(np.log(res[-1]/res[0]))/it
    print(f"dpit = {digits_per_it},   offdiag = {offdiag},    rank = {r}")

    dpits.append(digits_per_it)
    offdiags.append(offdiag)
    ranks.append(r)





plt.scatter(dpits,offdiags)
plt.title("Digits per evaluation of A against offdiag norms")
plt.xlabel("Digits recovered per evaluation of A")
plt.ylabel("Offdiag norms")
plt.savefig("offdiags.svg")
plt.close()

plt.scatter(dpits,ranks)
plt.title("Digits per iteration against structural ranks")
plt.xlabel("Digits per iteration")
plt.ylabel("Structural ranks")
plt.savefig("sranks.svg")
plt.close()
