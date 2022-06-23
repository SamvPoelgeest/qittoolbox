import scipy.sparse as sparse
from scipy.sparse import linalg as sparse_linalg
from ..logging.logger import log
from math import ceil, factorial, floor
from ..linalg.tensors import basis, tens, maximally_entangled_psi_plus
from scipy.special import binom, jacobi
from itertools import permutations
from itertools import product as itertools_product
import time

_INSTALLED_INTERFACES = {'picos':False, 'cvxpy':False}

try:
    import picos as pc
    _INSTALLED_INTERFACES['picos'] = True
except:
    pass
try:
    import cvxpy as cp
    from cvxpy.atoms.affine.trace import trace as cp_trace
    from cvxpy.atoms.affine.partial_transpose import partial_transpose as cp_partial_transpose
    from cvxpy.atoms.affine.partial_trace import partial_trace as cp_partial_trace
    from cvxpy.atoms.affine.kron import kron as cp_kron
    _INSTALLED_INTERFACES['cvxpy'] = True
except:
    pass


def symmetric_subspace_tuple_generator(d:int, n:int) -> 'tuple[int]':
    """
    Generates all tuples (t1,...,td) such that t(i) >= 0 and sum_i t(i) = n. Starts with (0,...,n), then (0,...,1,n-1), etc.

    INPUT:
        d: int, length of the tuple (t1,...,t(d))
        n: int, sum of the tuple elements sum_i t(i) = n.

    OUTPUT:
        yields a tuple of ints (t1,...,t(d)) such that sum_i t(i) = n.

    Almost verbatim copy of https://stackoverflow.com/questions/29170953/how-to-generate-a-set-of-all-tuples-of-given-length-and-sum-of-elements 
    """
    if d == 1:
        yield (n,)
        return
    
    for i in range(n+1):
        for rest_result in symmetric_subspace_tuple_generator(d-1,n-i):
            yield (i,) + rest_result

def get_permutation_operator( dim: int, sigma: 'list[int]' )-> sparse.spmatrix:
    """
    Produces a permutation matrix acting on the tensor product of Hilbert spaces H_0 \otimes .... \otimes H_{n-1},
        where dim H_i = `dim` and n = len(dims) , which permutes a vector | i_0, ..., i_{n-1} > to the vector
        | j_0, ... , j_{n-1} > where j(k) = i( sigma^{-1}(k)  ) .

    INPUT:
        dim: int, dimensionality of the Hilbert spaces
        sigma: list of ints, permutation list.

    OUTPUT:
        sparse.spmatrix of size ( prod(dims) , prod(dims) ) representing the permutation operator.

    See: https://codereview.stackexchange.com/questions/241565/efficient-numpy-array-manipulation-to-convert-an-identity-matrix-to-a-permutatio
    """
    #Note that |i_0, ... , i_n >  --> | j_0, ..., j_n > with j(k) = i( sigma^{-1}(k) ) is equivalent to
    # saying that i(k) = j( sigma(k) ).
    if not 0 in sigma:
        log('[get_permutation_operator] sigma does not seem to be 0-based, so removing 1 from each value...','debug')
        sigma = [x-1 for x in sigma]
    
    if not set(sigma) == set(range(len(sigma))):
        log(f'[get_permutation_operator] sigma does not seem to be a valid permutation list. set(sigma) = {set(sigma)}, expected {set(range(len(sigma)))}','fatal')
        return None

    # Algorithm: loop over all rows stored in I, compute their decomposition into tensor components |i_0,...,i_{n-1}> captured by `row_dics`,
    # then compute the corresponding column index by permuting these indices according to sigma,
    # then re-encode this column index list into a column index using the kronecker-product indexing rule,
    # and then output a sparse matrix with 1's at these particular locations. 
    n = len(sigma)
    prod_dim = dim**n
    vals = [1]*prod_dim
    I = range(prod_dim)
    J = []
    for row_idcs in itertools_product(range(dim), repeat=n):
        col_idcs = [ row_idcs[perm_idx] for perm_idx in sigma ]

        # row_idx = sum( r * dim**(n-1-idx) for idx,r in enumerate(row_idcs) )
        col_idx = sum( c * dim**(n-1-idx) for idx,c in enumerate(col_idcs) )
        J.append(col_idx)

    return sparse.coo_matrix((vals, (I,J)))

def get_symmetric_projection(k: int, d: int ) -> sparse.spmatrix:
    """
    Gets the symmetric projection operator onto the symmetric subspace of k tensor copies of C^d.

    INPUT:
        k: int, number of tensor copies. Should be >= 1.
        d: int, dimensionality of the fundamental Hilbert space C^d.

    OUTPUT:
        sparse.spmatrix of shape ( d**k, d**k ) representing the projection operator on (C^d)^(otimes k). 

    NOTE:
        This method quickly becomes quite slow for increasing dimension.
    """
    if k == 1:
        log('[get_symmetric_projection] Asked to get the symmetric projection on k = 1 copies of the Hilbert space, this is the identity.','info')
        return sparse.eye(d)
    
    perms = permutations(range(k))
    k_fac = factorial(k)
    outp = sparse.coo_matrix((d**k,d**k))

    for perm in perms:
        outp += get_permutation_operator(d, perm)
    
    return outp / k_fac

def symmetric_extension(rho:sparse.spmatrix, dim_A:int, dim_B: int, k:int=2, ppt:bool=True, extend_B:bool=None, interface:str='cvxpy', solver:str='mosek', **kwargs) -> 'tuple[bool,sparse.spmatrix]':
    """
    Attempts to find the k-symmetric extension of the input state rho.

    INPUT:
        rho: sparse.spmatrix, density matrix on $H_A \otimes H_B$ , so must have rho.shape = (dim_A*dim_B,dim_A*dim_B).
        dim_A: int, dimension of Hilbert space H_A
        dim_B: int, dimension of Hilbert space H_B
        k: int=2, the depth of the symmetric extension. k = 2 corresponds with tensoring on H_B once. Must have k >= 2.
        ppt: bool=True, if set to True also checks the PPT conditions, this strengthens the symmetric extension testing but is slower.
        extend_B: bool=None, set to True if you want to extend dim_B, set to False if you want to extend H_A instead, keep `None` to
            let this function decide what is most memory-efficient (preferred).
        interface: str='cvxpy', string should be in the list of _INSTALLED_INTERFACES and that flag should be set to True.
        solver: string='mosek', passed to the `solve` method of the interface, must be string associated with solver installed on system.
        kwargs: specific keyword arguments for particular interfaces and solvers.

    OUTPUT:
        tuple:
            [0]: boolean, set to True if the extension was found, otherwise False
            [1]: sparse.spmatrix, output of the solver. 
    """
    global _INSTALLED_INTERFACES
    if interface not in _INSTALLED_INTERFACES or not _INSTALLED_INTERFACES[interface]:
        log(f'[symmetric_extension] Error: _INSTALLED_INTERFACE does not list `{interface}` as installed. Instead, it is: \n{_INSTALLED_INTERFACES}','fatal')
        return None

    if k < 2:
        log(f'[symmetric_extension] Error: k = {k} < 2. I cannot do _less_ than tensor on H_B once.','fatal')
        return None

    if not rho.shape == (dim_A*dim_B,dim_A*dim_B):
        log(f'[symmetric_extension] Error: Wrong shape. rho.shape ={rho.shape} whilst dim_A = {dim_A}, dim_B = {dim_B}, product = {dim_A*dim_B}.','fatal')
        return None

    log(f'[symmetric_extension] ========================= [STARTING {k}-SYMMETRIC EXTENSION] ====================','info')
    _overall_t = time.time()

    # First step: determine dimensionalities
    if extend_B is None:
        extend_B = (dim_B < dim_A)
    dim_symmetric_out = int(binom(dim_B+k-1, k)) if extend_B else int(binom(dim_A+k-1, k))
    dim_reduced_symmetric_out = dim_B if extend_B else dim_A
    total_dim_out = dim_A * dim_symmetric_out if extend_B else dim_symmetric_out * dim_B
    log(f'[symmetric_extension] Computed dimensionalities as: \n' +\
            f'dim_A = {dim_A}, dim_B = {dim_B}, extend_B={extend_B}, dim_symmetric_out = {dim_symmetric_out}, k = {k}, total_dim_out = {total_dim_out}','info')


    # Second step: set up the extended density matrix.
    if interface == 'picos':
        rho_const = pc.Constant("rho", rho)
        prob = pc.Problem()
        rho_extended = pc.HermitianVariable("rho_extended", (total_dim_out, total_dim_out))
    elif interface == 'cvxpy':
        constraints = []
        rho_extended = cp.Variable(shape=(total_dim_out,total_dim_out),name="rho_extended", hermitian=True)

    # Third step: first set of constraints: rho_extended must be a density matrix
    if interface == 'picos':
        prob.add_constraint(rho_extended>>0)
        prob.add_constraint(pc.trace(rho_extended)==1)
    elif interface == 'cvxpy':
        constraints.append( rho_extended >> 0 )
        constraints.append( cp_trace(rho_extended) == 1 )

    # Fourth step: compute the permutation operator needed
    _t = time.time()
    log(f'[symmetric_extension] Computing the permutation operator... For large dimensions this may take some time.','info')
    sym_proj = get_symmetric_projection(k, dim_B) if extend_B else get_symmetric_projection(k, dim_A)
    sym_proj_partial_isom, s , _ = sparse_linalg.svds(sym_proj, k=dim_symmetric_out, return_singular_vectors="u")
    P_op = tens( sparse.eye(dim_A), sym_proj_partial_isom ) if extend_B else tens( sym_proj_partial_isom, sparse.eye(dim_B))

    log(f'[symmetric_extension] Computed the permutation operator in {(time.time()-_t):.2e}s.','info')

    # Fifth step: set up partial trace requirement
    dimensions = [dim_A, dim_B, dim_B**(k-1) ] if extend_B else [(dim_A)**(k-1), dim_A, dim_B]
    axis_totrace = 2 if extend_B else 0

    if interface == 'picos':
        # Note that SciPy sparse matrices do not currently respect the __array_priority__ attribute according to https://picos-api.gitlab.io/picos/numscipy.html, so P_op * rho_extended * P_op.T does not work!
        rho_extended_embedded = rho_extended.__rmul__(P_op) * P_op.T #picos uses * as matmul, @ as kron
    
        partially_traced_rho = pc.partial_trace(rho_extended_embedded, subsystems=axis_totrace, dimensions=dimensions)
        prob.add_constraint(partially_traced_rho == rho_const )
    elif interface == 'cvxpy':
        rho_extended_embedded = P_op @ rho_extended @ P_op.T 
        partially_traced_rho = cp_partial_trace(rho_extended_embedded , dimensions, axis=axis_totrace )
        constraints.append( partially_traced_rho == rho  )

    # Possible 5.1th step: ppt requirement
    if ppt:
        ppt_dims = [ dim_A * dim_B**(ceil(k/2)) , dim_B**(floor(k/2)) ] if extend_B else \
                    [ dim_A**(floor(k/2)), dim_A**(ceil(k/2))*dim_B ]
        ppt_axis_totranspose = 0 if extend_B else 1
        if interface == 'picos':
            prob.add_constraint( pc.partial_transpose(rho_extended_embedded,subsystems=ppt_axis_totranspose,dimensions=ppt_dims) >> 0)
        elif interface == 'cvxpy':
            constraints.append( cp_partial_transpose(rho_extended_embedded, ppt_dims, axis=ppt_axis_totranspose) >> 0 )

    # Sixth step: solve the problem
    if interface == 'cvxpy':
        prob = cp.Problem( cp.Minimize(0) , constraints )

    if interface == 'picos':
        if 'verbosity' in kwargs:
            prob.options['verbosity'] = kwargs['verbosity']

        log(f'[symmetric_extension] Current problem: \n{prob}','info') #Don't log in case of cvxpy, this causes huge output.

    if interface == 'picos':
        try:
            prob.solve(solver=solver)
        except Exception as e:
            log(f'[symmetric_extension] Picos threw an exception: {e}','error')
    elif interface == 'cvxpy':
        prob.solve(solver=solver.upper(), verbose=True)

    log(f'[symmetric_extension] Done! This took {(time.time()-_overall_t):.2e}s.','info')
    log(f'[symmetric_extension] Currently, prob.status = {prob.status}','info')
    log(f'[symmetric_extension] ========================= [END OF {k}-SYMMETRIC EXTENSION] ====================','info')
    return prob

def symmetric_inner_extension(rho:sparse.spmatrix, dim_A:int, dim_B: int, k:int=2, ppt:bool=True, extend_B:bool=None, interface:str='cvxpy', solver:str='mosek', **kwargs) -> 'tuple[bool,sparse.spmatrix]':
    """
    Attempts to find the k-symmetric inner extension of the input state rho.

    INPUT:
        rho: sparse.spmatrix, density matrix on $H_A \otimes H_B$ , so must have rho.shape = (dim_A*dim_B,dim_A*dim_B).
        dim_A: int, dimension of Hilbert space H_A
        dim_B: int, dimension of Hilbert space H_B
        k: int=2, the depth of the symmetric extension. k = 2 corresponds with tensoring on H_B once. Must have k >= 2.
        ppt: bool=True, if set to True also checks the PPT conditions, this strengthens the symmetric extension testing but is slower.
        extend_B: bool=None, set to True if you want to extend dim_B, set to False if you want to extend H_A instead, keep `None` to
            let this function decide what is most memory-efficient (preferred).
        interface: str='cvxpy', string should be in the list of _INSTALLED_INTERFACES and that flag should be set to True.
        solver: string='mosek', passed to the `solve` method of the interface, must be string associated with solver installed on system.
        kwargs: specific keyword arguments for particular interfaces and solvers.

    OUTPUT:
        tuple:
            [0]: boolean, set to True if the extension was found, otherwise False
            [1]: sparse.spmatrix, output of the solver. 
    """
    global _INSTALLED_INTERFACES
    if interface not in _INSTALLED_INTERFACES or not _INSTALLED_INTERFACES[interface]:
        log(f'[symmetric_inner_extension] Error: _INSTALLED_INTERFACE does not list `{interface}` as installed. Instead, it is: \n{_INSTALLED_INTERFACES}','fatal')
        return None

    if k < 2:
        log(f'[symmetric_inner_extension] Error: k = {k} < 2. I cannot do _less_ than tensor on H_B once.','fatal')
        return None

    if not rho.shape == (dim_A*dim_B,dim_A*dim_B):
        log(f'[symmetric_inner_extension] Error: Wrong shape. rho.shape ={rho.shape} whilst dim_A = {dim_A}, dim_B = {dim_B}, product = {dim_A*dim_B}.','fatal')
        return None


    log(f'[symmetric_extension] ========================= [STARTING {k}-INNER-SYMMETRIC EXTENSION] ====================','info')
    _overall_t = time.time()

    # First step: determine dimensionalities
    if extend_B is None:
        extend_B = (dim_B < dim_A)
    dim_symmetric_out = int(binom(dim_B+k-1, k)) if extend_B else int(binom(dim_A+k-1, k))
    d_out = dim_B if extend_B else dim_A
    total_dim_out = dim_A * dim_symmetric_out if extend_B else dim_symmetric_out * dim_B
    log(f'[symmetric_inner_extension] Computed dimensionalities as: \n' +\
            f'dim_A = {dim_A}, dim_B = {dim_B}, extend_B={extend_B}, dim_symmetric_out = {dim_symmetric_out}, k = {k}, total_dim_out = {total_dim_out}','info')


    # Second step: set up the extended density matrix.
    if interface == 'picos':
        rho_const = pc.Constant("rho", rho)
        prob = pc.Problem()
        rho_extended = pc.HermitianVariable("rho_extended", (total_dim_out, total_dim_out))
    elif interface == 'cvxpy':
        constraints = []
        rho_extended = cp.Variable(shape=(total_dim_out,total_dim_out),name="rho_extended", hermitian=True)

    # Third step: first set of constraints: rho_extended must be a density matrix
    if interface == 'picos':
        prob.add_constraint(rho_extended>>0)
        prob.add_constraint(pc.trace(rho_extended)==1)
    elif interface == 'cvxpy':
        constraints.append( rho_extended >> 0 )
        constraints.append( cp_trace(rho_extended) == 1 )

    # Fourth step: compute the permutation operator needed
    _t = time.time()
    log(f'[symmetric_inner_extension] Computing the permutation operator... For large dimensions this may take some time.','info')
    sym_proj = get_symmetric_projection(k, dim_B) if extend_B else get_symmetric_projection(k, dim_A)
    sym_proj_partial_isom, s , _ = sparse_linalg.svds(sym_proj, k=dim_symmetric_out, return_singular_vectors="u")
    P_op = tens( sparse.eye(dim_A), sym_proj_partial_isom ) if extend_B else tens( sym_proj_partial_isom, sparse.eye(dim_B))

    log(f'[symmetric_inner_extension] Computed the permutation operator in {(time.time()-_t):.2e}s.','info')

    # Fifth step: set up partial trace requirement
    dimensions = [dim_A, dim_B, dim_B**(k-1) ] if extend_B else [(dim_A)**(k-1), dim_A, dim_B]
    axis_totrace = 2 if extend_B else 0

    id_other = sparse.eye(dim_B) if extend_B else sparse.eye(dim_A)

    if interface == 'picos':
        # Note that SciPy sparse matrices do not currently respect the __array_priority__ attribute according to https://picos-api.gitlab.io/picos/numscipy.html, so P_op * rho_extended * P_op.T does not work!
        rho_extended_embedded = rho_extended.__rmul__(P_op) * P_op.T #picos uses * as matmul, @ as kron
        partially_traced_rho = pc.partial_trace(rho_extended_embedded, subsystems=axis_totrace, dimensions=dimensions)
        axis_totrace_small = 1 if extend_B else 0
        smaller_rho = pc.partial_trace(partially_traced_rho, subsystems=axis_totrace_small, dimensions=[dim_A,dim_B])
        # Note that SciPy sparse matrices do not currently respect the __array_priority__ attribute according to https://picos-api.gitlab.io/picos/numscipy.html, so id_other @ smaller_rho does not work!
        smaller_rho_tens_id = smaller_rho @ id_other if extend_B else smaller_rho.__rmatmul__(id_other)
    elif interface == 'cvxpy':
        rho_extended_embedded = P_op @ rho_extended @ P_op.T 
        partially_traced_rho = cp_partial_trace(rho_extended_embedded , dimensions, axis=axis_totrace )
        axis_totrace_small = 1 if extend_B else 0
        smaller_rho = cp_partial_trace(partially_traced_rho, [dim_A,dim_B], axis=axis_totrace_small)
        smaller_rho_tens_id = cp_kron(smaller_rho,id_other) if extend_B else cp_kron(id_other,smaller_rho)
    
    # Possible 5.1th step: ppt requirement
    if ppt:
        eps_n = dim_B / (2*(dim_B-1)) * min( 1-x for x in jacobi(floor(k/2)+1,dim_B-2,k%2).r ) if extend_B else \
                dim_A / (2*(dim_A-1)) * min( 1-x for x in jacobi(floor(k/2)+1,dim_A-2,k%2).r ) #.r gives the roots of a numpy polynomial
        
        if interface == 'picos':
            prob.add_constraint( (1-eps_n) * partially_traced_rho + eps_n / d_out * smaller_rho_tens_id == rho_const )
        elif interface == 'cvxpy':
            constraints.append( (1-eps_n) * partially_traced_rho + eps_n / d_out * smaller_rho_tens_id == rho )
    else:
        if interface == 'picos':
            prob.add_constraint( k/(k+d_out) * partially_traced_rho + 1 / (k+d_out) * smaller_rho_tens_id == rho_const )
        elif interface == 'cvxpy':
            constraints.append( k/(k+d_out) * partially_traced_rho + 1/(k+d_out) * smaller_rho_tens_id == rho  )

    # Sixth step: solve the problem
    if interface == 'cvxpy':
        prob = cp.Problem( cp.Minimize(0) , constraints )

    if interface == 'picos':
        if 'verbosity' in kwargs:
            prob.options['verbosity'] = kwargs['verbosity']

        log(f'[symmetric_inner_extension] Current problem: \n{prob}','info')  #Don't log in case of cvxpy, this causes huge output.

    if interface == 'picos':
        try:
            prob.solve(solver=solver)
        except Exception as e:
            log(f'[symmetric_inner_extension] Picos threw an exception: {e}','error')
    elif interface == 'cvxpy':
        prob.solve(solver=solver.upper(), verbose=True)

    log(f'[symmetric_inner_extension] Done! This took {(time.time()-_overall_t):.2e}s.','info')
    log(f'[symmetric_inner_extension] Currently, prob.status = {prob.status}','info')
    log(f'[symmetric_inner_extension] ========================= [END OF {k}-INNER-SYMMETRIC EXTENSION] ====================','info')

    return prob


def get_example_state(name:str, **kwargs) -> sparse.spmatrix:
    if name == 'choi':
        alpha = kwargs['alpha']
        dims = [3,3]
        psi_plus = maximally_entangled_psi_plus(3)
        psi_plus_mat = psi_plus @ psi_plus.getH()

        sigma_plus = 1.0/3.0 * basis([0,1], size=dims)@basis([0,1], size=dims,bra=True) +\
                     1.0/3.0 * basis([1,2],size=dims)@basis([1,2],size=dims,bra=True) +\
                    1.0/3.0 * basis([2,0],size=dims)@basis([2,0],size=dims,bra=True)

        V = sum( sum( basis([i,j],size=dims)@basis([j,i],size=dims,bra=True) for j in range(3) ) for i in range(3)  )
        
        rho = 2.0/7.0 * psi_plus_mat + alpha/7.0 * sigma_plus + (5-alpha)/7.0 * V @ sigma_plus @ V
    else:
        log(f'[get_example_state] Error: the name {name} has not been implemented :(. Returning None','error')
        return None

    return rho 