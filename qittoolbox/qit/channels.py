import scipy.sparse as sparse
from ..logging.logger import log
from math import prod, sqrt
from ..linalg.tensors import basis, tens, get_basis_idx_in_tensor_prod_space, get_basis_idx_in_full_space
import scipy.linalg as linalg 

def get_partial_transpose(mat: sparse.spmatrix, dims: 'list[int]', systems: 'list[int]') -> sparse.spmatrix:
    """
    Takes the partial transpose over a matrix mat. If mat = A1 otimes ... otimes A(n), each living in Hilbert spaces H(i),
    we take in a list dims with dims[i] = dim H(i), and a list of indices in `systems`, and we transpose each A(j) with a
    j in `systems`.

    INPUT:
        mat: sparse.spmatrix, matrix to be partially transposed
        dims: list of integers, dimensions of the Hilbert spaces H(i). If subsystems are not square, make it a list of list of integers.
        systems: list of integer, systems that we should transpose.

    OUTPUT:
        sparse.spmatrix that represents the partially transposed mat.
    """
    def is_iterable(obj) -> bool:
        try:
            _ = iter(obj)
        except TypeError as te:
            return False
        return True

    dims = [ [x,x] if not is_iterable(x) else x for x in dims ]
    # print('now dims are', dims)


    #Check if the dimensions check out
    if ( prod( x[0] for x in dims ), prod( x[1] for x in dims ) ) != mat.shape:
        log(f'[get_partial_transpose] Dimension mismatch. prod(dims) gives shape {( prod( x[0] for x in dims ), prod( x[1] for x in dims ) )} whilst mat.shape = {mat.shape}','fatal')
        return None
    
    #Check if the systems are valid
    if any(x<0 or x >= len(dims) for x in systems):
        log(f'[get_parital_tranpose] System mismatch, system integers passed {systems}, whilst they should all be between 0 and {len(dims)}.','fatal')
        return None

    mat_dok = mat.asformat('dok')

    partial_transpose_dims = [ [x[0],x[1]] if not i in systems else [x[1],x[0]] for i,x in enumerate(dims) ]
    
    out = sparse.dok_matrix((prod(x[0] for x in partial_transpose_dims), prod(x[1] for x in partial_transpose_dims)))

    for (i,j), value in mat_dok.items():
        i_lst = get_basis_idx_in_tensor_prod_space(i, [x[0] for x in dims])
        j_lst = get_basis_idx_in_tensor_prod_space(j, [x[1] for x in dims])

        for syst_i in systems:
            i_lst[syst_i], j_lst[syst_i] = j_lst[syst_i], i_lst[syst_i]
        
        new_i = get_basis_idx_in_full_space(i_lst, [x[0] for x in partial_transpose_dims])
        new_j = get_basis_idx_in_full_space(j_lst, [x[1] for x in partial_transpose_dims])
        out[new_i,new_j] = value
    return out


def get_choi_matrix_from_kraus(kraus_ops: 'list[sparse.spmatrix]', channel_left:bool=False, partial_transpose:bool=False) -> sparse.spmatrix:
    """
    Creates the Choi matrix from the Kraus operators of a quantum channel.

    INPUT:
        kraus_ops: list of sparse.spmatrix representing the Kraus operators of a quantum channel
        channel_left: bool=False, if False we  apply the channel in the right tensor leg, if True in the left tensor leg.
        partial_transpose: bool=False, if set to True, tranposes the first tensor leg.

    OUTPUT:
        sparse.spmatrix representing the (possibly partially-transposed) Choi matrix.
    """
    dim_out, dim_in = kraus_ops[0].shape

    choi_mat = sparse.coo_matrix((dim_in*dim_out,dim_in*dim_out))

    omega_mat = None
    if not partial_transpose:
        max_entangled = sum( basis([i,i], size=[dim_in,dim_in]) for i in range(dim_in) )
        max_ent_mat = max_entangled @ max_entangled.getH()
        omega_mat = max_ent_mat
    else:
        max_entangled_swapped_mat = sum(sum( basis([i,j], size=[dim_in,dim_in])@basis([j,i], size=[dim_in,dim_in],bra=True) for j in range(dim_in)) for i in range(dim_in))
        omega_mat = max_entangled_swapped_mat

    id_mat = sparse.eye(dim_in)
    for kraus in kraus_ops:
        kraus_extended = tens(kraus, id_mat) if channel_left else tens(id_mat, kraus)
        choi_mat += kraus_extended @ omega_mat @ kraus_extended.getH()
    return choi_mat


def check_kraus_representation(kraus_ops: 'list[sparse.spmatrix]', threshold:float=1e-10) -> 'tuple[bool,float,bool]':
    """
    Checks whether the Kraus operators E(i) satisfy the demand sum_i E(i)^* E(i) = id_n, where id_n is the identity matrix.
    INPUT:
        kraus_ops: list of sparse.spmatrices, representing the Kraus operators E(i) = kraus_ops[i]
        threshold: float=1e-10, amount the resulting matrix may differ from 0, component-wise.
    OUTPUT:
        tuple of three:
            0. boolean, True if component-wise max deviation from id_n is < threshold
            1. float, component-wise max deviation from id_n
            2. boolean, True if resulting sparse matrix has no non-zero entries. For debuggin purposes mainly.
    """
    out_dim = kraus_ops[0].shape[1]
    id_n = sparse.eye(out_dim,format='dok')

    sum_kraus_ops = sum(x.getH().dot(x) for x in kraus_ops).todok()
    diff = abs(sum_kraus_ops-id_n)

    max_diff = 0 if diff.getnnz() == 0 else max(diff.values())
    return max_diff < threshold, max_diff, diff.getnnz() == 0

def apply_kraus_channel(kraus_ops: 'list[sparse.spmatrix]', rho: sparse.spmatrix) -> sparse.spmatrix:
    """
    Applies the quantum channel described by Kraus operators E(i) = kraus_ops[i] to the density matrix rho.
    INPUT:
        kraus_ops: list of sparse.spmatrices, representing the Kraus operators E(i) = kraus_ops[i]
        rho: sparse.spmatrix, representing the input density matrix
    OUTPUT:
        rho_out: sparse.spmatrix, representing the output density matrix.
    """
    return sum( x @ rho @ x.getH() for x in kraus_ops)

def get_kraus_ops_from_stinespring(isometry: sparse.spmatrix, dim_trace: int, left_trace: bool) -> 'list[sparse.spmatrix]':
    """
    Gets the Kraus operator representation of a quantum channel from its Stinespring representation Phi(rho) = Tr_{A/B} ( V @ rho @ V* ),
    where Tr_{A/B} is the partial trace over the first or second tensor leg in H_out = H_A otimes H_B, given by left_trace, where the 
    dimension of the traced-out Hilbert space is dim_trace. Here, V is an isometry given by the isometry parameter.

    INPUT:
        isometry: sparse.spmatrix, Stinespring isometry that embed H_in in H_A otimes H_B.
        dim_trace: int, size of the subsystem H_A or H_B that is to be traced out.
        left_trace: bool, if True, we trace out H_A, else we trace out H_B in H_out = H_A otimes H_B.

    OUTPUT:
        list of sparse.spmatrices representing the Kraus operators belonging to this channel.
    """
    kraus_ops = []
    dim_out = float(isometry.shape[0]) / float(dim_trace)
    if not dim_out.is_integer():
        log(f'[get_kraus_ops_from_stinespring] : Cannot continue, because dim(H_A otimes H_B) = {isometry.shape[0]}, but dim_trace = {dim_trace}, dividing them yields non-integer {dim_out}','fatal')
        return []
    dim_out = int(dim_out)
    id_matrix = sparse.eye(dim_out)

    for basis_idx in range(dim_trace):
        basis_bra = basis(basis_idx, size=dim_trace, bra=True)
        pre_kraus = tens(basis_bra,id_matrix) if left_trace else tens(id_matrix,basis_bra)
        kraus = pre_kraus @ isometry
        if kraus.getnnz() > 0:
            kraus_ops.append(kraus)
    return kraus_ops

def minimize_num_kraus_ops(kraus_ops: 'list[sparse.spmatrix]', tolerance:float=1e-14) -> 'list[sparse.spmatrix]':
    """
    Minimizes the number of Kraus operators necessary to describe the quantum channel, by diagonalizing the Choi matrix of the quantum channel.
    This can be quite costly!

    INPUT:
        kraus_ops: list of sparse.spmatrices, Kraus operator representation of the quantum channel.
        tolerance: float=1e-14, if eigenvalues are closer to 0 than tolerance, we ignore them.
    
    OUTPUT:
        list of sparse.spmatrices that are also a Kraus operator representation of the quantum channel.
    """
    new_kraus_ops = []
    dim_out, dim_in = kraus_ops[0].shape

    log(f'[minimize_num_kraus_ops] Filling the Choi matrix (dim_in={dim_in},dim_out={dim_out}), this can take some time...','info')

    choi_mat = sparse.coo_matrix((dim_in*dim_out,dim_in*dim_out))
    for i in range(dim_in):
        for j in range(dim_in):
            basis_ij = basis(i,size=dim_in) @ basis(j,size=dim_in,bra=True)
            choi_mat += tens(basis_ij, sum(kraus @ basis_ij @ kraus.getH() for kraus in kraus_ops) )
    
    log(f'[minimize_num_kraus_ops] Filled the Choi matrix, now diagonalizing it with shape {choi_mat.shape} and finding Kraus operators...','info')
    w, v = linalg.eigh(choi_mat.toarray() )
    for idx in range(len(w)):
        eigval = w[idx]
        eigvect = v[:,idx]
        if abs(eigval) > tolerance:
            kraus = sqrt(eigval) * eigvect.reshape((dim_out,dim_in),order="F")
            new_kraus_ops.append( sparse.coo_matrix(kraus) )
    
    return new_kraus_ops