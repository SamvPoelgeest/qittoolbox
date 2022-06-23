from inspect import trace
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.sparse.linalg as sparse_linalg
from math import sqrt
from ..qfunctions.qfunctions import q0_bracket, theta_q0
from ..logging.logger import log
from ..linalg.tensors import tens, basis
from ..qit.channels import get_kraus_ops_from_stinespring, minimize_num_kraus_ops

from itertools import product as cartesian_product

_GLOBAL_N = 3
_memoization_Jones_Wenzl = {}
_memoization_intertwiner_T = {}

def get_T_intertwiner(r: int, N:int=_GLOBAL_N, memoize:bool=True) -> sparse.spmatrix:
    """
    Computes the [BC16] T_r intertwiner in Hom(1, u^{otimes 2}) as matrix.
    INPUT:  
        r: integer,
        N: integer, dimensionality
        memoize: bool=True, if True then memoization is used, data is stored in the global _memoization_intertwiner_T. Carefully
            assess how much memory this costs!
    OUTPUT:
        sparse.spmatrix representing T_r. This is a column vector.
    """
    global _memoization_intertwiner_T

    if r == 0:
        log(f'[get_T_intertwiner] r == 0, returning [1x1] matrix. This might lead to untested behaviour.','warning')
        return sparse.eye(1)
    
    if r == 1:
        T1 = sum( tens( basis(i,size=N), basis(i,size=N) ) for i in range(N) )
        return T1

    if memoize:
        if (r,N) in _memoization_intertwiner_T.keys():
            log(f'[get_T_intertwiner] Using memoization to find T_r for (r,N) = {(r,N)}.','debug')
            return _memoization_intertwiner_T[(r,N)]

    # Use functional description of T_r: T_r = sum_i e_i otimes e_j, where
    # the sum is over all i: [r] -> [N], 
    # and j: [r] -> [N] is determined by j(s) = i(r-s+1)
    out = sparse.dok_matrix( (N**(2*r), 1) )
    for i in cartesian_product( range(N), repeat=r ):
        j = tuple(i[r-1-x] for x in range(len(i)))
        out += tens( tens(*tuple( basis(x,size=N) for x in i ) ), tens(*tuple( basis(y,size=N) for y in j ) ) )

    if memoize:
        _memoization_intertwiner_T[(r,N)] = out

    return out


def get_Jones_Wenzl_projection(k: int, N:int=_GLOBAL_N, memoize:bool=True) -> sparse.spmatrix:
    """
    Computes the [BC16] Jones-Wenzl projection p_k.
    INPUT:
        k: integer,
        N: int=_GLOBAL_N, dimensionality
        memoize: bool=True, if True then memoization is used, data is stored in the global _memoization_Jones_Wenzl. Carefully
            assess how much memory this costs!

    OUTPUT:
        sparse.spmatrix representing p_k
    """
    global _memoization_Jones_Wenzl
    if k == 0:
        log(f'[get_Jones_Wenzl_projection] k == 0, returning a [1x1] matrix. This might lead to untested behaviour.','warning')
        return sparse.eye(1)
    
    p1 = sparse.eye(N)
    if k == 1:
        return p1

    if memoize:
        if (k,N) in _memoization_Jones_Wenzl.keys():
            log(f'[get_Jones_Wenzl_projection] Using memoization to find p_k for (k,N) = {(k,N)}.','debug')
            return _memoization_Jones_Wenzl[(k,N)]

    #Use the Wenzl recursion formula otherwise
    iH1 = sparse.eye(N)
    iH1km2 = sparse.eye( N**(k-2) )
    pkm1 = get_Jones_Wenzl_projection(k-1,N=N,memoize=memoize)

    T1 = get_T_intertwiner(1,N=N,memoize=memoize)
    T1mat = T1 @ T1.getH()

    newproj = tens(iH1,pkm1)
    
    if memoize:
        out = newproj - q0_bracket(k-1,N=N)/q0_bracket(k,N=N) * newproj @ tens(T1mat,iH1km2) @ newproj
        _memoization_Jones_Wenzl[(k,N)] = out
        return out
    else:
        return newproj - q0_bracket(k-1,N=N)/q0_bracket(k,N=N) * newproj @ tens(T1mat,iH1km2) @ newproj

def get_A_intertwiner(k: int, l: int, m: int, N:int=_GLOBAL_N, memoize:bool=True) -> sparse.spmatrix:
    """"
    Computes the [BC16] intertwiner A(k,l,m), the unnormalized version of alpha(k,l,m).
    INPUT:
        k: integer, 
        l: integer,
        m: integer,
        N: int=_GLOBAL_N, dimensionality,
        memoize: bool=True, if True then memoization is used for T intertwiner and Jones-Wenzl projections.

    OUTPUT:
        sparse.spmatrix representing A(k,l,m)
    """
    #k = l + m -2r
    r = float(l+m-k)/2.0
    if not r.is_integer():
        log(f'[get_A_intertwiner] r=(l+m-k)/2 = ({l}+{m}-{k})/2 = {r} is not an integer. Aborting.','fatal')
        return sparse.dok_matrix((0,0))
    
    r = int(r)

    p_k = get_Jones_Wenzl_projection(k, N=N, memoize=memoize)
    T_r = get_T_intertwiner(r,N=N, memoize=memoize)
    p_l = get_Jones_Wenzl_projection(l,N=N, memoize=memoize)
    p_m = get_Jones_Wenzl_projection(m,N=N, memoize=memoize)

    iH1lmr = sparse.eye(N**(l-r))
    iH1mmr = sparse.eye(N**(m-r))

    return tens(p_l,p_m) @ tens(iH1lmr,T_r,iH1mmr) @ p_k
    

def get_alpha_isometry(k: int, l: int, m: int, N:int=_GLOBAL_N, memoize:bool=True) -> sparse.spmatrix:
    """
    Computes the [BC16] isometry alpha(k,l,m), the normalized version of the A(k,l,m) map.
    INPUT:
        k: integer,
        l: integer,
        m: integer,
        N: int=_GLOBAL_N, dimensionality
        memoize: bool=True, if True then memoization is used for T intertwiner and Jones-Wenzl projections.

    OUTPUT:
        sparse.spmatrix representing alpha(k,l,m) 
    """

    return sqrt(  q0_bracket(k+1,N=N) / theta_q0(k,l,m,N=N) ) * get_A_intertwiner(k,l,m,N=N,memoize=memoize)

def get_kraus_ops(k: int, l: int, m: int, N:int=_GLOBAL_N,  trace_out_first: bool=True, method: str='stinespring', tolerance:float=1e-14) -> 'list[sparse.spmatrix]':
    """
    Gets the Kraus operators belonging to the [BC16] quantum channel. 
    INPUT:
        k: integer,
        l: integer,
        m: integer,
        N: int=_GLOBAL_N, dimensionality
        trace_out_first: bool=True, set to True if you wish to trace out the first part of the tensor product, otherwise set to False.
        method: str, in ['stinespring','choi']. Stinespring is presumably faster, but may lead to more Kraus operators than necessary.
            Using Choi matrix decomposition might be slower, but should yield a smaller number of Kraus operators.
        tolerance: float=1e-14, in case of method=='choi', this specifies when we throw out eigenvalues for being within tolerance dist to 0.
    OUTPUT:
        list of sparse.spmatrices that represent the Kraus operators.
    """
    if method == 'stinespring':
        isometry = get_alpha_isometry(k,l,m,N=N)
        dim_trace = N**l if trace_out_first else N**m
        return get_kraus_ops_from_stinespring(isometry, dim_trace, trace_out_first)
    elif method == 'choi':
        kraus_ops = get_kraus_ops(k,l,m,N=N,trace_out_first=trace_out_first,method='stinespring')
        return minimize_num_kraus_ops(kraus_ops, tolerance=tolerance)
    else:
        log(f'[get_kraus_ops] : Method {method} not implemented. Aborting','fatal')
        return []

def get_choi_matrix(k:int, l:int, m:int, N:int=_GLOBAL_N, trace_out_first:bool=True) -> 'tuple[sparse.spmatrix,str]':
    """
    Gets the Choi matrix belonging to the O_N^+ channel using its algebraic expression in terms of the intertwiners alpha_klm

    INPUT:
        k: integer,
        l: integer,
        m: integer,
        N: int=_GLOBAL_N, dimensionality
        trace_out_first: bool=True, set to True if you wish to trace out the first part of the tensor product, otherwise set to False.

    OUTPUT:
        tuple with items:
            [0] sparse.spmatrix representing the Choi matrix of the (k,l,m,trace_out_first=True/False)-O_N^+ quantum channel
            [1] str, either 'left' or 'right', denoting whether the Choi matrix is left- or right-handed.
    """
    if trace_out_first:
        alpha_lmk = get_alpha_isometry(l,m,k,N=N)
        return q0_bracket(k+1,N=N) / q0_bracket(l+1,N=N) * alpha_lmk @ alpha_lmk.getH() , 'left'
    else:
        alpha_mkl = get_alpha_isometry(m,k,l,N=N)
        return q0_bracket(k+1,N=N) / q0_bracket(m+1,N=N) * alpha_mkl @ alpha_mkl.getH() , 'right'
