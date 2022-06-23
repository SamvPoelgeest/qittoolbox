import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from ..logging.logger import log
from .tensors import basis
from math import prod

def get_isometry_from_range_space_proj(proj: sparse.spmatrix, rank: int) -> sparse.spmatrix:
    """
    Uses sparse SVD decomposition to decompose a projection matrix `proj` with shape (n,n) 
        with a known `rank < n`, returns a matrix `A` of shape (n,r) such that
        `A.T @ A = id_r` and `A @ A.T = proj` . 
    """
    if rank > proj.shape[0]:
        log(f'[get_isometry_from_range_space_proj] Error, rank = {rank} > proj.shape[0] = {proj.shape[0]}.','fatal')
        return None
    if rank == proj.shape[0]:
        log(f'[get_isometry_from_range_space_proj] Warning, rank = {rank} = proj.shape[0], so we are returning sparse eye(rank). Check whether this is what you want!','warning')
        return sparse.eye(rank)
    partial_isom, s , _ = sparse_linalg.svds(proj, k=rank, return_singular_vectors="u")
    return partial_isom

def get_orthonormal_basis(vects: 'list[sparse.spmatrix]', threshold: float=1e-14) -> 'list[sparse.spmatrix]':
    """
    Implements the Gram-Schmidt process to make an orthonormal basis (ONB) from the vector list. Linearly dependent
    input is allowed, will filter out those vectors if their norm dives below the threshold.

    INPUT:
        vects: list of sparse.spmatrices, the vectors that span the subspace for which we wish to find an ONB.
        threshold: float=1e-14, if new vector in Gram-Schmidt algorithm has norm < threshold, we assume it is linearly dependent.

    OUTPUT:
        list of sparse.spmatrices representing the ONB.
    """
    def proj(v: sparse.spmatrix, u: sparse.spmatrix ) -> sparse.spmatrix:
        #dot results in a [1x1] sparse matrix, so cast to a Dictionary Of Keys and read out the first element.
        return u.getH().dot(v).todok()[0,0] * u
    
    #Find the first non-zero vector
    for first_vect_idx, first_vect in enumerate(vects):
        if sparse_linalg.norm(first_vect,ord='fro') > threshold:
            break
    else:
        log(f'[get_orthonormal_basis] : All input vectors have a norm smaller than threshold={threshold}. Dumping the vectors...','fatal')
        log('\n'.join(str(x) for x in vects),'fatal')
        return []

    #Create the ONB
    onb = [first_vect/sparse_linalg.norm(first_vect,ord='fro')]
    for vect_idx, vect in enumerate(vects):
        is_linear_combination = False
        new_vect = vect.copy()

        for onb_vect in onb:
            new_vect -= proj(vect,onb_vect)
            if sparse_linalg.norm(new_vect,ord='fro') < threshold:
                is_linear_combination = True
                break
                
        if is_linear_combination:
            continue

        new_vect = new_vect / sparse_linalg.norm(new_vect,ord='fro')

        if any(sparse_linalg.norm(proj(new_vect,x),ord='fro') > threshold for x in onb):
            log(f'[get_orthonormal_basis] : ONB is no longer orthogonal if we add new_vect. Norms of all projections of new_vect onto current ONB:','fatal')
            log('\n'.join(str(sparse_linalg.norm(proj(new_vect,x),ord='fro')) for x in onb ),'fatal')
            return []

        onb.append(new_vect)
    
    return onb

def get_isomorphism_to_range_space(mat: sparse.spmatrix, threshold: float=1e-14) -> sparse.spmatrix:
    """
    Given an operator mat, its Range Ran(mat) = Span{ mat @ basis(i) }_{1 leq i leq dim H} where basis(i) are canonical basis elements
    in the input Hilbert space H. Then, we can find an isomorphism K --> Ran(mat), where K is a Hilbert space with dim K = rank(mat),
    by finding an orthonormal basis for Ran(mat) and then mapping the canonical basis elements in K to that ONB of Ran(mat).

    INPUT:
        mat: sparse.spmatrix representing the operator.
        threshold: float=1e-14, passed to get_orthonormal_basis.

    OUTPUT:
        sparse.spmatrix representing the basis transformation K --> Ran(mat).
    """
    size_in = mat.shape[1]
    out_vects = [ mat @ basis(i,size=size_in) for i in range(size_in) ]
    onb = get_orthonormal_basis(out_vects,threshold=threshold)

    rank = len(onb)
    return sum( x @ basis(i,size=rank, bra=True) for i,x in enumerate(onb) )




