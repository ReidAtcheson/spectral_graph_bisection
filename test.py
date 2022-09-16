import unittest
import sbisect
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pdb


class TestSbisect(unittest.TestCase):
    #Test nullspace properties of graph laplacian with random sparse matrix
    #Explicitly exclude diagonal of matrix
    def test_graph_laplacian_nodiag_nullspace(self):
        seed=2398743
        rng=np.random.default_rng(seed)
        ndiags=10
        m=256
        offs = np.sort(rng.choice(list(range(-m+1,0))+list(range(1,m)),size=ndiags,replace=False))
        A=sp.diags([rng.uniform(-1,1,m) for _ in range(ndiags)],offs,shape=(m,m))
        G=sbisect.graph_laplacian(A)

        #G should have a 1-dimensional nullspace
        s = la.svdvals(G.toarray())
        self.assertEqual(sum(s<max(s)*1e-14),1)

        #The nullspace of G should be the vector of all 1s
        e=np.ones(m)
        Ge=G@e
        self.assertLess(la.norm(Ge),1e-14)

    #Test nullspace properties of graph laplacian with random sparse matrix
    #Make sure properties independent of existence of a diagonal
    def test_graph_laplacian_diag_nullspace(self):
        seed=2398743
        rng=np.random.default_rng(seed)
        ndiags=10
        m=256
        offs = np.sort(rng.choice(list(range(-m+1,0))+list(range(1,m)),size=ndiags,replace=False))
        A=sp.diags([rng.uniform(-1,1,m) for _ in range(ndiags)],offs,shape=(m,m)) + sp.diags([rng.uniform(-1,1,m)],[0],shape=(m,m))
        G=sbisect.graph_laplacian(A)

        #G should have a 1-dimensional nullspace
        s = la.svdvals(G.toarray())
        self.assertEqual(sum(s<max(s)*1e-14),1)

        #The nullspace of G should be the vector of all 1s
        e=np.ones(m)
        Ge=G@e
        self.assertLess(la.norm(Ge),1e-14)


    #Graph laplacian should be symmetric
    def test_graph_laplacian_symmetric(self):
        seed=2398743
        rng=np.random.default_rng(seed)
        ndiags=10
        m=256
        offs = np.sort(rng.choice(list(range(-m+1,0))+list(range(1,m)),size=ndiags,replace=False))
        A=sp.diags([rng.uniform(-1,1,m) for _ in range(ndiags)],offs,shape=(m,m)) + sp.diags([rng.uniform(-1,1,m)],[0],shape=(m,m))
        G=sbisect.graph_laplacian(A)

        self.assertLess( sp.linalg.norm(G-G.T), 1e-14 )

    #Compare "matrix-free" fiedler vector calculation
    #to dense eigenvalue calculation
    def test_fiedler_vector(self):
        seed=2398743
        rng=np.random.default_rng(seed)
        ndiags=5
        m=256
        offs = [-32,-1,0,1,32]
        A=sp.diags([rng.uniform(-1,1,m) for _ in range(ndiags)],offs,shape=(m,m)) 
        G=sbisect.graph_laplacian(A)
        (t,f)=sbisect.fiedler(G,tol=1e-4,maxiter=200,verbosity=-1)

        es,V = la.eigh(G.toarray())

        self.assertLess( abs(es[1]-t)/es[1], 1e-4 )

    def test_split_partition(self):
        seed=2398743
        rng=np.random.default_rng(seed)
        ndiags=5
        m=256
        offs = [-32,-1,0,1,32]
        A=sp.diags([rng.uniform(-1,1,m) for _ in range(ndiags)],offs,shape=(m,m)) 
        G=sbisect.graph_laplacian(A)

        b0,s,b1 = sbisect.split(G)

        #Check this is a partition
        self.assertEqual(sorted(b0+s+b1),list(range(0,m)))

    def test_split_separator(self):
        seed=2398743
        rng=np.random.default_rng(seed)
        ndiags=5
        m=256
        offs = [-32,-1,0,1,32]
        A=sp.diags([rng.uniform(-1,1,m) for _ in range(ndiags)],offs,shape=(m,m)) 
        G=sbisect.graph_laplacian(A)

        b0,s,b1 = sbisect.split(G)

        #Check that `s` separates `b0` and `b1`
        x=np.zeros(m)
        x[b0]=rng.uniform(-1,1,size=len(b0))
        y=G@x
        self.assertEqual(list(y[b1]),list(np.zeros(len(b1))))

        x=np.zeros(m)
        x[b1]=rng.uniform(-1,1,size=len(b1))
        y=G@x
        self.assertEqual(list(y[b0]),list(np.zeros(len(b0))))

    def test_nested_dissection_partition(self):
        seed=2398743
        rng=np.random.default_rng(seed)
        ndiags=5
        m=256
        offs = [-32,-1,0,1,32]
        A=sp.diags([rng.uniform(-1,1,m) for _ in range(ndiags)],offs,shape=(m,m)) 
        G=sbisect.graph_laplacian(A)
        nd = sbisect.ndisect(G,list(range(0,m)),maxm=32,tol=1e-6,maxiter=200,verbosity=0)
        #Test that `nd forms a partition

        stack=[nd]
        ids=[]
        while stack:
            p=stack.pop()
            if isinstance(p,list):
                ids=ids+p
            else:
                b0,s,b1=p
                ids=ids+s
                stack.append(b0)
                stack.append(b1)

        self.assertEqual(sorted(ids),list(range(0,m)))

    def test_nested_dissection_nonempty_separator(self):
        seed=2398743
        rng=np.random.default_rng(seed)
        ndiags=5
        m=256
        offs = [-32,-1,0,1,32]
        A=sp.diags([rng.uniform(-1,1,m) for _ in range(ndiags)],offs,shape=(m,m)) 
        G=sbisect.graph_laplacian(A)
        nd = sbisect.ndisect(G,list(range(0,m)),maxm=32,tol=1e-6,maxiter=200,verbosity=0)
        #Test that `nd forms a partition

        stack=[nd]
        ids=[]
        while stack:
            p=stack.pop()
            if isinstance(p,list):
                ids=ids+p
            else:
                b0,s,b1=p
                self.assertGreater(len(s),0)



    def test_nested_dissection_assemble(self):
        seed=2398743
        rng=np.random.default_rng(seed)
        ndiags=5
        m=256
        offs = [-32,-1,0,1,32]
        A=sp.diags([rng.uniform(-1,1,m) for _ in range(ndiags)],offs,shape=(m,m))
        A=A+sp.diags([8*np.ones(m)],[0],shape=(m,m))
        A=sp.lil_matrix(A)
        G=sbisect.graph_laplacian(A)
        nd = sbisect.ndisect(G,list(range(0,m)),maxm=32,tol=1e-6,maxiter=200,verbosity=0)
        ids,parents,offs=sbisect.level_flatten(nd)
        Ap=sbisect.assemble(A,ids,parents,offs)
        #Ap should be the same as if we simply created the permutation p directly
        #and formed A[p,p]
        p=[]
        for n in ids:
            p=p+n
        Aph=A[np.ix_(p,p)]

        #plt.spy(Ap)
        #plt.savefig("Ap.svg")
        #plt.close()
        #plt.spy(Aph)
        #plt.savefig("Aph.svg")
        #plt.close()

        self.assertEqual(sp.linalg.norm( (Ap-Aph) ),0.0)












unittest.main()
