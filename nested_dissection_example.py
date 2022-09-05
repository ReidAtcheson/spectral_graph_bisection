import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import sbisect



mx=128
my=128
m=mx*my
ndiags=5
A=sp.diags([np.ones(m) for _ in range(ndiags)],[-mx,-1,0,1,mx],shape=(m,m))
G=sbisect.graph_laplacian(A)
t,f=sbisect.fiedler(G,maxiter=200)
ids0=[]
ids1=[]
for i,v in enumerate(f):
    if v<0:
        ids0.append(i)
    else:
        ids1.append(i)
p=ids0+ids1


A=sp.lil_matrix(A)
plt.spy(A,markersize=1)
plt.savefig("spyA.svg")
plt.close()

plt.spy(A[np.ix_(p,p)],markersize=1)
plt.savefig("spyPAP.svg")
plt.close()


def idx(ix,iy):
    return ix+mx*iy



xs = []
ys = []


fh=f/np.linalg.norm(f,ord=np.inf)
p=np.argsort(fh)
nparts=4


for k in range(0,len(p),len(p)//4):
    pb = k
    pe = min(len(p),k+len(p)//4)
    xs.append([])
    ys.append([])
    for pi in p[pb:pe]:
        ix = pi % mx
        iy = (pi - ix)//mx

        xs[-1].append(ix)
        ys[-1].append(iy)


plt.scatter(xs[0],ys[0],color="orange")
plt.scatter(xs[1],ys[1],color="green")
plt.scatter(xs[2],ys[2],color="red")
plt.scatter(xs[3],ys[3],color="black")
plt.savefig("grid_part.svg")
plt.close()




def bisect(p,maxsep,maxpart=128):
    if len(p)<maxpart:
        return p
    else:
        b2=len(p)//2
        return (
                bisect(p[0:b2-maxsep//2],maxsep//2),
                p[b2-maxsep//2 : b2 + maxsep//2],
                bisect(p[b2+maxsep//2:len(p)],maxsep//2)
                )




pb = sbisect.ndisect(G,list(range(0,m)),maxm=1024,tol=1e-6,maxiter=200,verbosity=0)
colors=['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c']
colors = plt.get_cmap("tab10").colors
print(colors)
stack=[pb]
i=0
p=[]
while stack:
    left,sep,right = stack.pop()
    pl = [sep]
    if isinstance(left,list):
        pl.append(left)
    else:
        stack.append(left)

    if isinstance(right,list):
        pl.append(right)
    else:
        stack.append(right)

    for pi in pl:
        xs=[]
        ys=[]
        for n in pi:
            ix = n % mx
            iy = (n - ix)//mx
            xs.append(ix)
            ys.append(iy)
        plt.scatter(xs,ys,color=colors[i%len(colors)])
        i=i+1



plt.savefig("ndisect.svg")
plt.close()



print(sum(np.abs(fh)<1e-1 /2))
plt.plot(np.sort(fh))
plt.savefig("vector_entries.svg")




