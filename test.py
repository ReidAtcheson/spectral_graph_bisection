import unittest
import sbisect
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


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








unittest.main()
