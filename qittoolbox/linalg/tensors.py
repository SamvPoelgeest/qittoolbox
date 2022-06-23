import scipy.sparse as sparse
from ..logging.logger import log
from math import prod, sqrt

_GLOBAL_N = 3

def tens(*arg) -> sparse.spmatrix:
    """
    tens(x1,x2,...,xn) : alias for the function scipy.sparse.kron(a,b) with variable number of elements,
    see scipy.sparse.kron for full documentation.
    
    INPUT:
        args: variable number of scipy.sparse arrays
    OUTPUT:
        out: scipy.sparse array
    """
    if len(arg)==0:
        log(f'[tens] : len(args) = 0. Input: {arg}', 'fatal')
        raise Exception('tens: len(args)=0. Do not comprehend input.')
    if len(arg)==1:
        log(f'[tens] : len(args) = 1. Simply returning this element', 'warning')
        return arg[0]

    out = sparse.kron(arg[0],arg[1])
    for i in range(2,len(arg)):
        out = sparse.kron(out,arg[i])
    return out

def get_basis_idx_in_tensor_prod_space(x: int, d_lst: 'list[int]' ) -> 'list[int]':
    """
    get_basis_idx_in_tensor_prod_space(x,d_lst) : maps a ket |x> in a large Hilbert space
        H to a ket |x1,...,xn> in the Hilbert space H1 otimes ... otimes Hn simeq H. 
    
    INPUT:
        x: int, value between 0 and dim(H)-1 (inclusive).
        d_lst: list of ints, dimensions of the Hilbert spaces H(i) such prod(d_lst) = dim(H).
    
    OUTPUT:
        out: list of ints of the form [x1,...,xn].
    """
    #For example, the tensor |x>|y>|z> is mapped to x*d0d1 + y*d1 + z, so find (x,y,z)
    #by divmod.
    out = []
    for curr_idx in range(len(d_lst)-1,-1,-1):
        x, r = divmod(x, d_lst[curr_idx] )
        out.append(r)
    
    return out[::-1]

def maximally_entangled_psi_plus(dim:int) -> sparse.spmatrix:
    """
    Creates the maximally entangled state psi^+ = (1/sqrt(dim)) sum_i |ii>

    INPUT:
        dim: integer, dimensionality

    OUTPUT
        sparse.spmatrix column vector representing psi^+.
    """
    return 1.0/sqrt(dim) * sum( basis([x,x], size=[dim,dim]) for x in range(dim) )

def get_basis_idx_in_full_space(x_lst: 'list[int]', d_lst: 'list[int]') -> int:
    """
    get_basis_idx_in_full_space(x_lst,d_lst) : maps a ket |x1>....|xn> in a tensor product Hilbert space
    H1 otimes ... otimes Hn with dimensions d(i) in d_lst = [d1,d2,...,dn] to a ket |x> in the full Hilbert
    space H simeq H1 otimes ... otimes Hn.
    
    INPUT:
        x_lst: list of ints, value(i) between 0 and d(i)-1 (inclusive)
        d_lst: list of ints, dimensions of the Hilbert spaces H(i)
    OUTPUT:
        x: int, value of the integer in the ket |x> in H.
    """
    return int(sum( xi * prod(d_lst[i+1:]) for i,xi in enumerate(x_lst) ) )

def swap(ket: sparse.spmatrix, idx1: int, idx2: int, d_lst: 'list[int]') -> sparse.spmatrix:
    """
    swap(ket,idx1,idx2,d_lst) swaps two qudits around in the vector |ket>,
    where |ket> is a vector in Hilbert space H, which decomposes as
    H simeq H1 otimes .... otimes H(n) with dim H(i) = d_lst[i].

    INPUT:
        ket: sparse.spmatrix of shape (x,1) for x = prod(d_lst)
        idx1: integer, index of the first qudit, 0 <= idx1 < len(d_lst)
        idx2: integer, index of the second qudit, 0 <= idx2 < len(d_lst)
        d_lst: list of integers, dimensions of the Hilbert spaces H(i).
    OUTPUT:
        out: sparse.spmatrix of shape (x,1)
    """
    
    out = sparse.dok_matrix(ket.shape)
    for key,val in (ket.todok()).items():
        idx = key[0]
        idx_lst = get_basis_idx_in_tensor_prod_space(idx,d_lst)
        idx_lst[idx1], idx_lst[idx2] = idx_lst[idx2], idx_lst[idx1]
        new_idx = get_basis_idx_in_full_space(idx_lst, d_lst)
        out[(new_idx,0)] = val
    return out.asformat(ket.getformat())

def permute(ket: sparse.spmatrix, permutation: 'list[int]',d_lst:'list[int]') -> sparse.spmatrix:
    """
    permute(ket,permutation,d_lst) permutes the qudits around in the vector |ket>,
    where |ket> is a vector in Hilbert space H, which decomposes as
    H simeq H1 otimes ... otimes H(n) with dim H(i) = d_lst[i].
    |ket> is represented as |x1>...|x(n)>, and then this is permuted to
    |x(s(i))> ... |x(s(n))> where s(i) = permutation[i].

    INPUT:
        ket: sparse.spmatrix of shape (x,1) for x = prod(d_lst)
        permutation: list of ints, swaps |x(i)> to |x(s(i))>
        d_lst: list of ints, dimensions of the Hilbert spaces H(i).
    OUTPUT:
        out: sparse.spmatrix of shape (x,1)
    """

    out = sparse.dok_matrix(ket.shape)
    for key,val in (ket.todok()).items():
        idx = key[0]
        idx_lst = get_basis_idx_in_tensor_prod_space(idx,d_lst)
        new_idx_lst = [ idx_lst[permutation[i]] for i in range(len(idx_lst)) ]
        new_idx = get_basis_idx_in_full_space(new_idx_lst,d_lst)
        out[(new_idx,0)] = val
    return out.asformat(ket.getformat())

def basis(k: 'int|list[int]', size: 'int|list[int]'=_GLOBAL_N, bra: bool=False) -> sparse.spmatrix:
    """
    basis(k,size,bra) returns a basis vector |k> in Hilbert space H, which is
    either characterized by dim H = size if size is an integer, or 
    H simeq H1 otimes .... otimes H(n) with dim H(i) = size[i] if size is a list.
    If bra==True, then the vector is transposed.

    INPUT:
        k: integer or list of integers, index of the vector
        size: integer or list of integers, dimension(s) of the Hilbert space(s)
        bra: bool=False, if set to true the output vector is transposed.
    OUTPUT:
        out: sparse.spmatrix respresenting |ket>.
    """
    #Check if both k and size are iterable, or both are int-like
    k_iterable = False
    size_iterable = False
    try:
        _ = iter(k)
        k_iterable = True
    except TypeError as te:
        pass

    try:
        __ = iter(size)
        size_iterable = True
    except TypeError as te:
        pass

    if k_iterable != size_iterable:
        log(f'[basis] You may only pass either k,size as ints, or both as iterables, instead found k={k},type(k)={type(k)},size={size},type(size)={type(size)}.','fatal')
        return sparse.dok_matrix((0,0))

    outsize = prod(size) if type(size) is list else size
    outshape = (outsize,1) if not bra else (1,outsize)

    out = sparse.dok_matrix(outshape)
    
    if type(k) is int:
        key = (k,0) if not bra else (0,k)
        out[key] = 1
    else:
        idx = get_basis_idx_in_full_space(k,size)
        key = (idx,0) if not bra else (0,idx)
        out[key] = 1
    return out

def to_braket_string(vect_or_mat: sparse.spmatrix, size_in: 'int|list[int]', size_out: 'int|list[int]', format_precision: int=3, tensor_style: bool=False) -> str:
    """
    to_braket_string(vect_or_mat,size) gives a nice representation of a vector or matrix in Dirac-braket format, i.e.
    |11><0| + |21><4|.

    INPUT:
        vect_or_mat: sparse.spmatrix, the vector or matrix you want to stringify
        size_in: int or list of ints, the dimension of the Hilbert space H_in of the *input*, i.e. the bra-part. If list of ints is
            given, we assume that H_in simeq H(1) otimes ... otimes H(n) with dim H(i) = size_in[i].
        size_out: int or list of ints, the dimension of the Hilbert space H_out of the *output*, i.e. the ket-part. If list of ints is
            given, we assume that H_out simeq H(1) otimes ... otimes H(n) with dim H(i) = size_out[i].
        format_precision: int=3, number of digits displayed in the scalars before each ket.
        tensor_style: bool=False, if True, display a state like |345>, if False, display it like |3>|4>|5>.

    OUTPUT:
        out: string, a string the represents the input vector or matrix in Dirac bra-ket notation.
    """
    if not sparse.issparse(vect_or_mat):
        # Attempt to cast the input to a sparse matrix
        try:
            # If it is a 1D np.array, try to first cast it to a 2D array
            if len(vect_or_mat.shape) == 1:
                log(f'[to_braket_string] :: Casting a 1D array to a 2D array by setting shape=(len,1).','debug')
                vect_or_mat.shape = (len(vect_or_mat),1)
            vect_or_mat = sparse.dok_matrix(vect_or_mat)
        except Exception as e:
            log(f'[to_braket_string] :: Could not cast vect_or_mat type {type(vect_or_mat)} to sparse.dok_matrix','fatal')
            return ''

    has_ket, has_bra = vect_or_mat.shape[0] != 1, vect_or_mat.shape[1] != 1
    if not has_ket and not has_bra:
        log(f'[to_braket_string] Encountered a [1x1] matrix {vect_or_mat}, cannot cast this to bra-ket string :(','error')
        return ''

    #In the case of tensor_style, we need to determine whether we need a separator if any of the dimensions > 9.
    sep_ket = sep_bra = ""
    if tensor_style:
        sep_ket = "" if all(x <= 9 for x in size_out) else " "
        sep_bra = "" if all(x <= 9 for x in size_in) else " "


    def scalar_and_idx_to_str(idx_key: 'list[int]',scalar: 'float|complex') -> str:
        nonlocal size_out, sep_ket, sep_bra, tensor_style, has_ket, has_bra
        
        scalar_str = str( round(scalar,format_precision) )
        
        #Determine the ket-part first.
        ket_str = ""
        if has_ket:
            if type(size_out) == list:
                idx_lst = get_basis_idx_in_tensor_prod_space(idx_key[0], size_out)
                if tensor_style:
                    ket_str = "|" + sep_ket.join(str(x) for x in idx_lst) + ">"
                else:
                    ket_str = "".join( "|"+str(x)+">" for x in idx_lst)
            else:
                ket_str = "|" + str(idx_key[0]) + ">"

        #Determine the bra-part next.
        bra_str = ""
        if has_bra:
            if type(size_in) == list:
                idx_lst = get_basis_idx_in_tensor_prod_space(idx_key[1], size_in)
                if tensor_style:
                    bra_str = "<" + sep_bra.join(str(x) for x in idx_lst) + "|"
                else:
                    bra_str = "".join( "<"+str(x)+"|" for x in idx_lst)
            else:
                bra_str = "<" + str(idx_key[1]) + "|"
        
        return scalar_str + ket_str + bra_str

    
    out_str = " + ".join(scalar_and_idx_to_str(idx_key,scalar) \
                     for idx_key,scalar in vect_or_mat.todok().items() if abs(scalar) > 10**(-format_precision))
    if len(out_str) == 0:
        log(f'[to_braket_string] empty out_str. Possibly, format_precision={format_precision} is not high enough?','warning')
    return out_str

