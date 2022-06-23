import scipy.sparse as sparse
import scipy.linalg as linalg
from numpy import eye as dense_eye
from ..logging.logger import log

def _get_infimum_two_projections(proj1: sparse.spmatrix, proj2: sparse.spmatrix) -> sparse.spmatrix:
    """
    Returns the infimum of two orthogonal projections, defined as the orthogonal projection onto the intersection of the range
        spaces of the two orthogonal projections. Uses the formula from
        W.N. Anderson and R.J. Duffin, 1969, Series and Parallel Addition of Matrices.
        See also R. Piziak, P.L. Odell and R. Hahn, Constructing Projections on Sums and Intersections.

    INPUT:
        proj1,proj2: sparse.spmatrix, representing the two orthogonal projections

    OUTPUT:
        sparse.spmatrix representing the infimum of proj1 and proj2.
    """

    if not sparse.issparse(proj1) == sparse.issparse(proj2):
        log(f'[_get_infimum_two_projections] Passed one sparse and one dense matrix, outputting a sparse matrix anyway','info')

    s = proj1 + proj2
    if sparse.issparse(s):
        #Note that pinvh only exists for dense matrices.
        s = s.todense()
        pinv = linalg.pinvh(s)

    if sparse.issparse(proj1) or sparse.issparse(proj2):
        sparse_format = proj1.getformat() if sparse.issparse(proj1) else proj2.getformat()
        out = 2 * proj1 @ sparse.coo_matrix(pinv).asformat(sparse_format) @ proj2
        
        if not sparse.issparse(out):
            return sparse.coo_matrix(out)
        else:
            return out

    else:
        return 2 * proj1 @ pinv @ proj2

def _get_supremum_two_projections(proj1: sparse.spmatrix, proj2: sparse.spmatrix) -> sparse.spmatrix:
    """
    Returns the supremum of two orthogonal projections, defined as the orthogonal projection onto the union of the range
        spaces of the two orthogonal projections. Uses the formula from
        W.N. Anderson and R.J. Duffin, 1969, Series and Parallel Addition of Matrices.

    INPUT:
        proj1,proj2: sparse.spmatrix, representing the two orthogonal projections

    OUTPUT:
        sparse.spmatrix representing the suprmemum of proj1 and proj2.
    """

    if not sparse.issparse(proj1) == sparse.issparse(proj2):
        log(f'[_get_supremum_two_projections] Passed one sparse and one dense matrix, this might lead to performance issues','info')

    if sparse.issparse(proj1) or sparse.issparse(proj2):
        sparse_format = proj1.getformat() if sparse.issparse(proj1) else proj2.getformat()
        eye = sparse.eye(proj1.shape[0], format=sparse_format)
    else:
        eye = dense_eye(proj1.shape[0])
    return eye - _get_infimum_two_projections(eye-proj1,eye-proj2)

def get_infimum_projections(*arg: 'list[sparse.spmatrix]') -> sparse.spmatrix:
    """
    Returns the infimum of a collection of orthogonal projections, defined as the orthogonal projection onto the intersection of the range
        spaces of the collection of orthogonal projections. Uses the formula from
        W.N. Anderson and R.J. Duffin, 1969, Series and Parallel Addition of Matrices.

    INPUT:
        *arg: list of sparse.spmatrix's, representing the collection of orthogonal projections

    OUTPUT:
        sparse.spmatrix representing the infimum of the collection of orthogonal projections.
    """

    if len(arg) == 0:
        log(f'[get_infimum_projections] len(arg) = 0, so I do not understand the input. ','fatal')
        return None
    if len(arg) == 1:
        log(f'[get_infimum_projections] len(arg) = 1. Simply returning the first input.', 'warning')
        return arg[0]

    res = _get_infimum_two_projections(arg[0],arg[1])
    length = len(arg)
    for k in range(2,length):
        res = _get_infimum_two_projections(res,arg[k])
    return res

def get_supremum_projections(*arg: 'list[sparse.spmatrix]') -> sparse.spmatrix:
    """
    Returns the suprmemum of a collection of orthogonal projections, defined as the orthogonal projection onto the union of the range
        spaces of the collection of orthogonal projections. Uses the formula from
        W.N. Anderson and R.J. Duffin, 1969, Series and Parallel Addition of Matrices.

    INPUT:
        *arg: list of sparse.spmatrix's, representing the collection of orthogonal projections

    OUTPUT:
        sparse.spmatrix representing the suprmemum of the collection of orthogonal projections.
    """

    if len(arg) == 0:
        log(f'[get_supremum_projections] len(arg) = 0, so I do not understand the input. ','fatal')
        return None
    if len(arg) == 1:
        log(f'[get_supremum_projections] len(arg) = 1. Simply returning the first input.', 'warning')
        return arg[0]

    res = _get_supremum_two_projections(arg[0],arg[1])
    length = len(arg)
    for k in range(2,length):
        res = _get_supremum_two_projections(res,arg[k])
    return res