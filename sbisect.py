import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def graph_laplacian(A):
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



