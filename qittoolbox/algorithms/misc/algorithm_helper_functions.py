import numpy as np
import math
import scipy.linalg as linalg

def get_complex_vector_from_coords(coords: np.ndarray) -> np.ndarray:
    """
    Changes a real vector `coords` with len 2n-1 to a complex vector y with len n, according to the rule:
    y[0] = coords[0]
    y[1] = coords[1] + j * coords[2], ... , y[n] = coords[2n-2] + j*coords[2n-1]

    INPUT:
        coords: np.array of shape (2n-1,) with real numbers as values

    OUTPUT:
        complexified vector of shape (n,).
    """
    #coords stored as x[0], x[1], ..., x[n]. This corresponds to the complex vector y with y[0] = x[0] and y[1] = x[1] + j*x[2], y[2] = x[3] + j*x[4], ..., y[n/2] = x[n-1] + j*x[n].
    #So y[0] = x[0] and y[k] = x[2k-1] + 1j*x[2k]
    return np.concatenate( ([coords[0]] , coords[1::2] + 1j*coords[2::2] ) )

def get_complex_vector_from_coords_multi(coords: np.ndarray) -> np.ndarray:
    """
    Changes a real vector `x := coords[:,i]` with len 2n-1 to a complex vector y with len n, according to the rule:
    y[0] = coords[0,i]
    y[1] = coords[1,i] + j * coords[2,i], ... , y[n,i] = coords[2n-2,i] + j*coords[2n-1,i]

    INPUT:
        coords: np.ndarray of shape (2n-1,n_particles) with dtype real floats, where each column is a real vector.

    OUTPUT:
        np.ndarray of shape (n,n_particles), where each column is complex vector.
    """
    #coords stored as x[0], x[1], ..., x[n]. This corresponds to the complex vector y with y[0] = x[0] and y[1] = x[1] + j*x[2], y[2] = x[3] + j*x[4], ..., y[n/2] = x[n-1] + j*x[n].
    #So y[0] = x[0] and y[k] = x[2k-1] + 1j*x[2k]
    return np.concatenate( ([coords[0,:]] , coords[1::2,:] + 1j*coords[2::2,:] ) )

def get_coords_from_complex_vector(vector: np.ndarray) -> np.ndarray:
    """
    Changes a complex vector `vector` with len n to a real vector y with len 2n-1, according to the rule:
    y[0] = vector[0]
    y[1] = real( vector[1] )
    y[2] = imag( vector[1] )
    ....
    y[2k-1] = real( vector[k] )
    y[2k] = iamg( vector[k] )

    We assume `vector[0]` is real!

    INPUT:
        vector: np.ndarray of complex dtype with shape (n,)
    
    OUTPUT:
        np.ndarray of real float dtype with shape (2*n-1,)
    """

    shape = ( vector.shape[0]*2-1 , ) if len(vector.shape) == 1 else ( vector.shape[0]*2-1 , 1)
    out = np.zeros( shape )
    out[0] = np.real(vector[0])
    out[1::3] = np.real(vector[1:])
    out[2::2] = np.imag(vector[1:])
    return out


def get_density_matrix_from_vector(vector: np.ndarray) -> np.ndarray:
    """
    Changes a complex vector |vector> into the rank-one density matrix |vector><vector|.

    INPUT:
        vector: np.array of shape (n,)

    OUTPUT:
        |vector><vector| np.narray of shape (n,n).
    """
    return np.outer(vector,np.conjugate(vector))

def get_density_matrix_from_vector_multi(vectors: np.ndarray, make_copy: bool=True) -> np.ndarray:
    """
    Changes a complex vector `|x> := vectors[:,i]` into the rank-one density matrices `|x><x|` and put them in a 3D np.ndarray.

    INPUT:
        vectors: np.ndarray of shape (n,n_particles), where each `vectors[:,i]` denotes a vector that we should turn into a density matrix.
        make_copy: bool=True, set to False to directly change the np.ndarray `vectors` in-place.
    OUTPUT:
        np.ndarray of shape (n_particles,n,n), where each `[i,:,:]` represents the matrix `|x><x|` for `|x> = vectors[:,i]`. 
    """
    x = np.copy(vectors) if make_copy else vectors 
    y= np.conj(vectors.T) #this automatically makes a copy of the data, presumably #TODO
    
    x = x.T #transpose, so that the shape is now (n_particles,n)
    x.shape = (x.shape[0],x.shape[1],1) #add a third dimension for the outer product: (n_particles,n,1)

    y.shape = (y.shape[0],1,y.shape[1]) #add a third dimension, we are now (n_particles,1,n)
    out_prod = x @ y #matmul automatically used the last 2 dimensions, so we get (n_particles,n,n) dimensional array

    return out_prod
    #return np.transpose(out_prod, axes=[1,2,0]) # The i'th axis of the result corresponds to axes[i] of out_prod, i.e. we get (n,n,n_particles) as promised.

def get_von_neumann_entropy(rho: np.ndarray, base: int=None, **kwargs) -> float:
    """
    Computes the vNE H(rho) = - Tr(rho * log(rho) ) = - \sum_i lambda_i log lambda_i where lambda_i \in spectrum(rho). 

    INPUT:
        rho : np.ndarray, representing a density matrix
        base : int, base of the logarithm. Standard: None, which corresponds to base e.
        **kwargs: <<UNIMPLEMENTED>>

    OUTPUT:
        float representing the vNE.
    """
    if base is None:
        base = math.e
    return -sum( x * math.log(x,base) if x > 0 else 0 for x in linalg.eigvalsh(rho))

def get_von_neumann_entropy_multi(rho: np.ndarray, base: int=None, **kwargs) -> np.ndarray:
    """
    Computes the vNE H(rho) = - Tr(rho * log(rho) ) = - \sum_i lambda_i log lambda_i where lambda_i \in spectrum(rho)
    for a collection of density matrices rho[i,:,:].

    INPUT:
        rho : np.ndarray of shape (n_matrices, n, n), where each rho[i,:,:] represents a density matrix
        base : int, base of the logarithm. Standard: None, which corresponds to base e.
        **kwargs: <<UNIMPLEMENTED>>

    OUTPUT:
        np.ndarray of shape (n_matrices,) with floats representing the vNEs.
    """

    if base is None:
        base = math.e
    return np.array( [-sum( x * math.log(x,base) if x > 0 else 0 for x in linalg.eigvalsh(rho[i,:,:])) for i in range(rho.shape[0]) ] )


