
import scipy.sparse as sparse
from ..logging.logger import log
from ..linalg.tensors import tens, basis
from ..qfunctions.qfunctions import q0_bracket, theta_q0
from ..qit.channels import get_kraus_ops_from_stinespring, minimize_num_kraus_ops
from math import sqrt
from numpy import around as numpy_around
import time

_GLOBAL_N = 4
_memoization_Jones_Wenzl_NC = {}
_memoization_intertwiner_T_NC = {}
_memoization_nested_cap_NC = {}

def _round_sparse_matrix(mat:sparse.spmatrix, round_lvl:int=15) -> sparse.spmatrix:
    """
    Rounds the non-zero elements of a sparse matrix using `round_lvl` amount of digits. Then, eliminates those elements that have rounded to 0.0

    INPUT:
        mat: sparse.spmatrix, matrix to be rounded
        round_lvl: int=15, number of digits to which we will round.

    OUTPUT:
        sparse.spmatrix, rounded version of mat. The matrix `mat` will not be affected.
    """
    _start_time = time.time()
    more_sparse = mat.copy()
    more_sparse.data = numpy_around(more_sparse.data,round_lvl)
    more_sparse.eliminate_zeros()
    log(f'[_round_sparse_matrix] :: from nnz {mat.nnz} to nnz {more_sparse.nnz}, this is a {round((mat.nnz-more_sparse.nnz)/mat.nnz*100,2)} % reduction.'+\
            f' This took {round(time.time()-_start_time,2)} sec.','debug')
    return more_sparse

def get_nested_cap_NC(r: int, N:int=_GLOBAL_N, memoize:bool=True) -> sparse.spmatrix:
    """
    Computes the nested cap partition \cap^r in Hom(1, u^{2r}) as matrix.

    INPUT:
        r: integer,
        N: int=_GLOBAL_N, dimensionality
        memoize: bool=true, if True then memoization is used, data is stored in the global _memoization_nested_cap_NC. Carefully
            assess how much memory this costs!

    OUTPUT:
        sparse.spmatrix representing the nested cap cap^r.
    """

    #Prevent recursion hell in get_nest_cap_NC early on, r should be >= 0.
    if r < 0:
        log(f'[get_nested_cap_NC] r < 0, this is impossible. Aborting.','fatal')
        return None

    if r == 0:
        log(f'[get_nested_cap_NC] r == 0, this is the empty partition in NC(0,0), are you sure this is correct?. Returning [1x1] identity.','info')
        return sparse.eye(1)
    
    if memoize:
        if (r,N) in _memoization_nested_cap_NC:
            log(f'[get_nested_cap_NC] Using memoization for (r,N) = ({r},{N}).','debug')
            return _memoization_nested_cap_NC[(r,N)]
    
    if r == 1:
        out = sum( basis([x,x], size=[N,N]) for x in range(N) )
    else:
        cap_rm1 = get_nested_cap_NC(r-1,N=N,memoize=memoize)
        out = sum( tens( basis(x,size=N), cap_rm1, basis(x,size=N) ) for x in range(N) )
    
    if memoize:
        _memoization_nested_cap_NC[(r,N)] = out
    
    return out

def get_T_intertwiner_NC(k: int, l:int, m:int, N:int=_GLOBAL_N, memoize:bool=True) -> sparse.spmatrix:
    """
    Computes the intertwiner T in Hom(u^k, u^l otimes u^m) as matrix.

    INPUT:
        k: integer such that |l-m| <= k <= l+m.
        l: integer,
        m: integer,
        N: int=_GLOBAL_N, dimensionality.
        memoize: bool=True, if True then memoization is used, data is stored in the global _memoization_intertwiner_T_NC. Carefully
            assess how much memory this costs!

    OUTPUT:
        sparse.spmatrix representing the intertwiner T
    """
    global _memoization_intertwiner_T_NC
    r = l+m-k

    if r < 0 or r > 2*min(l,m):
        log(f'[get_T_intertwiner_NC] r < 0 or r > 2*min(l,m) for k,l,m,r={k,l,m,r}, this is impossible. Aborting.','fatal')
        return None

    if r == 0:
        log(f'[get_T_intertwiner_NC] r == 0, so this simply becomes an identity on k = {k} = l+m = {l+m} strings.','debug')
        return sparse.eye( N**k )

    if memoize:
        if (k,l,m,N) in _memoization_intertwiner_T_NC:
            log(f'[get_T_intertwiner_NC] Using memoiziation for (k,l,m,N) = ({k},{l},{m},{N}).','debug')
            return _memoization_intertwiner_T_NC[(k,l,m,N)]

    if r%2 == 0:
        left = sparse.eye(N**(l-r//2))
        right = sparse.eye(N**(m-r//2))

        center = get_nested_cap_NC(r//2, N=N, memoize=memoize)
        out = tens(left, center, right)

    else:
        rprime = (r-1)//2
        left = sparse.eye( N**(l-rprime-1) )
        right = sparse.eye( N**(m-rprime-1) )

        small_center = get_nested_cap_NC(rprime, N=N, memoize=memoize)
        center = sum( tens( basis(x,size=N), small_center, basis(x,size=N) ) @ basis(x,size=N,bra=True) for x in range(N) )
        out = tens(left, center, right)
    
    if memoize:
        _memoization_intertwiner_T_NC[(k,l,m,N)] = out
    
    return out



def get_Jones_Wenzl_projection_NC(k: int, N:int=_GLOBAL_N, memoize:bool=True, round_lvl:int=-1) -> sparse.spmatrix:
    """
    Computes the inverse image in NC(k) of the Jones-Wenzl projection p_{2k} from TL_{2k}(delta). Here, delta^2 = N.

    INPUT:
        k: integer,
        N: int=_GLOBAL_N, dimensionality. Note that delta^2 = N, so the loop param of TL_{2k}(delta) is sqrt(N)
        memoize: bool=True, if True thne memoization is used, data is stored in the global _memoizations_Jones_Wenzl_NC. Carefully
            assess how much memory this costs!
        round_lvl: int=-1, if set to a positive integer, rounds the resulting sparse.spmatrix to `round_lvl` number of digits, removes resulting 0.0 values. 

    OUTPUT:
        sparse.spmatrix representing the inverse image in NC(k) of p_{2k} in TL_{2k}(delta).  
    """

    global _memoization_Jones_Wenzl_NC
    
    if k == 0:
        log('[get_Jones_Wenzl_projection_NC] k == 0, returning a [1x1] matrix. This might lead to untested behaviour.', 'info')
        return sparse.eye(1)

    if memoize:
        if (k,N) in _memoization_Jones_Wenzl_NC.keys():
            log(f'[get_Jones_Wenzl_projection_NC] Using memoization for (k,N) = ({k},{N}).','debug')
            if round_lvl > -1:
                return _round_sparse_matrix(_memoization_Jones_Wenzl_NC[(k,N)],round_lvl=round_lvl)
            else:
                return _memoization_Jones_Wenzl_NC[(k,N)]

    delta = sqrt(N)

    # Identity strand
    id_part = sparse.eye(N)
    
    # Disconnected strands in NC(1,1) with components { {1}, {1'} }
    ghz =  sum( basis(x,size=N) for x in range(N)  )
    disconnected_part = ghz @ ghz.getH()

    # All connected strands in NC(2,2) with components { {1,2,1',2'} }
    connected_part = sum( tens(basis(x,size=N), basis(x,size=N) ) @ tens(basis(x,size=N,bra=True) , basis(x,size=N,bra=True) ) for x in range(N) )

    # The prefix `hat_` denotes that the operator lives in NC(k), otherwise they live in TL_{2k}(delta)
    hat_p1 = id_part - (delta**-2) * disconnected_part # so this is the inverse image of p2

    if k == 1:
        if memoize:
            _memoization_Jones_Wenzl_NC[(k,N)] = hat_p1
        return hat_p1

    hat_pkm1 = get_Jones_Wenzl_projection_NC(k-1,N=N,memoize=memoize)
    hat_pkm1_tens_id = tens(hat_pkm1, id_part)
    disconnected_part_last = tens( sparse.eye(N**(k-1)) , disconnected_part )
    connected_part_last = tens( sparse.eye(N**(k-2)), connected_part)

    qkm2 = q0_bracket(2*k-2,N=delta, round_ans=False)
    qkm1 = q0_bracket(2*k-1,N=delta, round_ans=False)
    qk   = q0_bracket(2*k,  N=delta, round_ans=False)

    hat_A = tens(hat_pkm1, id_part)
    hat_B = -1 * delta * qkm2 / qkm1 * hat_pkm1_tens_id @ connected_part_last @ hat_pkm1_tens_id
    hat_C = -1 * (delta**-1) * qkm1/qk * disconnected_part_last @ hat_pkm1_tens_id
    hat_D = 1 * qkm1/qk * qkm2/qkm1 * hat_pkm1_tens_id @ connected_part_last @ hat_pkm1_tens_id @ disconnected_part_last
    hat_E = 1 * qkm1/qk * qkm2/qkm1 * disconnected_part_last @ hat_pkm1_tens_id @ connected_part_last @ hat_pkm1_tens_id
    hat_F = -1 * delta * qkm1 / qk * ( qkm2/qkm1 )**2 * hat_pkm1_tens_id @ connected_part_last @ hat_pkm1_tens_id @ disconnected_part_last @ connected_part_last @ hat_pkm1_tens_id

    hat_pk = hat_A + hat_B + hat_C + hat_D + hat_E + hat_F

    if memoize:
        _memoization_Jones_Wenzl_NC[(k,N)] = hat_pk

    if round_lvl > -1:
        return _round_sparse_matrix(hat_pk,round_lvl=round_lvl)
    else:
        return hat_pk

def get_A_intertwiner_NC(k:int, l:int, m:int, N:int=_GLOBAL_N, memoize:bool=True) -> sparse.spmatrix:
    """"
    Computes the S_N^+ intertwiner A(k,l,m), the unnormalized version of alpha(k,l,m).
    INPUT:
        k: integer, 
        l: integer,
        m: integer,
        N: int=_GLOBAL_N, dimensionality,
        memoize: bool=True, if True then memoization is used for T intertwiner and Jones-Wenzl projections.

    OUTPUT:
        sparse.spmatrix representing A(k,l,m)
    """
    
    p_k = get_Jones_Wenzl_projection_NC(k,N=N,memoize=memoize)
    T_intertwiner = get_T_intertwiner_NC(k,l,m,N=N,memoize=memoize)
    p_l = get_Jones_Wenzl_projection_NC(l,N=N,memoize=memoize)
    p_m = get_Jones_Wenzl_projection_NC(m,N=N,memoize=memoize)
    return tens(p_l,p_m) @ T_intertwiner @ p_k

def get_alpha_isometry_NC(k: int, l:int, m:int, N:int=_GLOBAL_N, memoize:bool=True) -> sparse.spmatrix:
    """
    Computes the S_N^+ isometry alpha(k,l,m), the normalized version of the A(k,l,m) map.
    INPUT:
        k: integer,
        l: integer,
        m: integer,
        N: int=_GLOBAL_N, dimensionality
        memoize: bool=True, if True then memoization is used for T intertwiner and Jones-Wenzl projections.

    OUTPUT:
        sparse.spmatrix representing alpha(k,l,m) 
    """

    #Bugfix: if r defined by k = l + m - r is odd, this should contain an additional sqrt(delta) = sqrt(sqrt(N)) term.
    r = l+m-k
    odd_term = 1 if r % 2 == 0 else N**(1.0/4.0)
    return odd_term * sqrt( q0_bracket(2*k+1,N=sqrt(N),round_ans=False) / theta_q0(2*k,2*l,2*m,N=sqrt(N),round_ans=False) ) * get_A_intertwiner_NC(k,l,m,N=N,memoize=memoize)

def get_kraus_ops(k:int, l:int, m:int, N:int=_GLOBAL_N, trace_out_first: bool=True, method:str='stinespring', tolerance:float=1e-14, memoize:bool=True) -> 'list[sparse.spmatrix]':
    """
    Gets the Kraus operators belonging to the S_N^+ quantum channel. 
    INPUT:
        k: integer,
        l: integer,
        m: integer,
        N: int=_GLOBAL_N, dimensionality
        trace_out_first: bool=True, set to True if you wish to trace out the first part of the tensor product, otherwise set to False.
        method: str, in ['stinespring','choi']. Stinespring is presumably faster, but may lead to more Kraus operators than necessary.
            Using Choi matrix decomposition might be slower, but should yield a smaller number of Kraus operators.
        tolerance: float=1e-14, in case of method=='choi', this specifies when we throw out eigenvalues for being within tolerance dist to 0.
        memoize: bool=True, if True then memoization is used for T intertwiner and Jones-Wenzl projections.

    OUTPUT:
        list of sparse.spmatrices that represent the Kraus operators.
    """

    if method == 'stinespring':
        isometry = get_alpha_isometry_NC(k,l,m,N=N,memoize=memoize)
        dim_trace = N**l if trace_out_first else N**m
        return get_kraus_ops_from_stinespring(isometry, dim_trace, trace_out_first)
    elif method == 'choi':
        kraus_ops = get_kraus_ops(k,l,m,N=N,trace_out_first=trace_out_first,method='stinespring')
        return minimize_num_kraus_ops(kraus_ops, tolerance=tolerance)
    else:
        log(f'[get_kraus_ops] : Method {method} not implemented. Aborting','fatal')
        return []

def get_choi_matrix_NC(k:int, l:int, m:int, N:int=_GLOBAL_N, trace_out_first:bool=True) -> 'tuple[sparse.spmatrix,str]':
    """
    Gets the Choi matrix belonging to the S_N^+ channel using its algebraic expression in terms of the intertwiners alpha_klm

    INPUT:
        k: integer,
        l: integer,
        m: integer,
        N: int=_GLOBAL_N, dimensionality
        trace_out_first: bool=True, set to True if you wish to trace out the first part of the tensor product, otherwise set to False.

    OUTPUT:
         tuple with items:
            [0] sparse.spmatrix representing the Choi matrix of the (k,l,m,trace_out_first=True/False)-S_N^+ quantum channel
            [1] str, either 'left' or 'right', denoting whether the Choi matrix is left- or right-handed.
    """
    
    if trace_out_first:
        alpha_lmk = get_alpha_isometry_NC(l,m,k,N=N)
        return q0_bracket(2*k+1,N=sqrt(N)) / q0_bracket(2*l+1,N=sqrt(N)) * alpha_lmk @ alpha_lmk.getH() , 'left'
    else:
        alpha_mkl = get_alpha_isometry_NC(m,k,l,N=N)
        return q0_bracket(2*k+1,N=sqrt(N)) / q0_bracket(2*m+1,N=sqrt(N)) * alpha_mkl @ alpha_mkl.getH() , 'right'