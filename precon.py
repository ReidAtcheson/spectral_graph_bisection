import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sbisect


mx=64
my=64
m=mx*my
seed=2398743
diag=4.0
maxsep=32
rng=np.random.default_rng(seed)

offs = [-mx,-1,0,1,mx]
A=sp.lil_matrix(sp.diags([rng.uniform(-1,1,m) for _ in offs],offs,shape=(m,m)) + diag*sp.diags([rng.uniform(0.3,1,m)],[0],shape=(m,m)))



G=sbisect.graph_laplacian(A)
nd = sbisect.ndisect(G,list(range(0,m)),maxm=32,tol=1e-6,maxiter=200,verbosity=0,maxsep=maxsep)
ids,parents,offs=sbisect.level_flatten(nd)
Aph=sbisect.assemble(A,ids,parents,offs)
Ab=sbisect.assemble_block(A,ids,parents,offs)
p=[]
for n in ids:
    p=p+n
Ap=A[np.ix_(p,p)]


b=rng.uniform(-1,1,size=m)

it=0
res=[]
def callback(xk):
    global it
    it=it+1
    r=b-Ap@xk
    res.append(np.linalg.norm(r))
    print(f"it={it}   res = {np.linalg.norm(r)}")


Mb=spla.splu(Ab)
M=spla.splu(Aph)

spla.gmres(Ap,b,callback=callback,callback_type="x",maxiter=100,M = spla.LinearOperator((m,m),matvec=Mb.solve),maxiter=100)
plt.semilogy(res)
res=[]
it=0


spla.gmres(Ap,b,callback=callback,callback_type="x",M = spla.LinearOperator((m,m),matvec=M.solve),maxiter=100)
plt.semilogy(res)
plt.title("Comparing two preconditioners on GMRES")
plt.xlabel("GMRES Iterations")
plt.ylabel("Residual")
plt.savefig("gmres.svg")














