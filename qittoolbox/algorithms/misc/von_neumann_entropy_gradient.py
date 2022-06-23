from typing import Callable
import numpy as np
from .algorithm_helper_functions import get_complex_vector_from_coords, get_density_matrix_from_vector
from ...logging.logger import log
import scipy.linalg as linalg

_gradient_mat_memoized = {}

def _setup_gradient_mat(name: str, dim_in_real: int, dim_out_cplx: int, channel_function: 'Callable[[np.ndarray], np.ndarray]') -> None:
    """
    Setup function that loads data from specific channel outputs into the globally accessible `_gradient_mat_memoized` dictionary
    that helps speed up the gradient calculation of the von Neumann entropy of a quantum channel.

    INPUT:
        name: str, name of the channel function. Used as a key in the `_gradient_mat_memoized` dictionary.
        dim_in_real: int, input dimension, this should be the dimensionality of the *real* space over which we are optimizing.
        dim_out_cplx: int, output dimension of the quantum channel.
        channel_function: function that takes in a complex square matrix of size (n,n) where dim_in_real = 2n-1, and outputs
            a complex square matrix of size (dim_out_cplx, dim_out_cplx).

    OUTPUT:
        None. Data is instead stored in the globally accessible `_gradient_mat_memoized`
    """
    global _gradient_mat_memoized

    grad_mat = np.zeros((dim_in_real,dim_in_real,dim_out_cplx,dim_out_cplx), dtype=np.complex128)
    for i in range(dim_in_real):
        coords_i = np.zeros((dim_in_real,1))
        coords_i[i] = 1
        vect_i = get_complex_vector_from_coords(coords_i)
        for j in range(dim_in_real):
            coords_j = np.zeros((dim_in_real,1))
            coords_j[j] = 1
            vect_j = get_complex_vector_from_coords(coords_j)

            grad_mat[i,j,:,:] = channel_function( vect_i @ np.conjugate( vect_j.T ) )

    _gradient_mat_memoized[name] = grad_mat

    log(f'[_setup_gradient_mat] Initialized the gradient matrix for the channel function with name {name}.','debug')

def grad_vne_channel(name: str, real_vect: np.ndarray, channel_function: 'Callable[[np.ndarray], np.ndarray]', regularizer:float=1e-15, threshold_check: float=1e-10) -> np.ndarray:
    """
    Calculates the gradient of the output entropy of a quantum channel described by `channel_function` at the position `real_vect`.

    INPUT:
        name: str, name of the channel function. Used as a key in the `_gradient_mat_memoized` dictionary.
        real_vect: np.ndarray of size (2n-1,) where n = dim H_1, where `channel_function` represents a quantum channel mapping states from H_1 to H_2.
        channel_function: function that takes in a complex square matrix of size (n,n) where dim_in_real = 2n-1, and outputs
            a complex square matrix of size (dim_out_cplx, dim_out_cplx).
        regularizer: float = 1e-15, regularizer used in the derivative to ensure it always exists. Set to 0 to avoid regularization.
        threshold_check: float = 1e-10, if set to nonzero value, we check whether the gradient has an imaginary part larger than `threshold_check`, as the
            gradient should be real-valued.

    OUTPUT:
        np.ndarray of size (2n-1,) of a real floating point dtype representing the gradient.
    """
    
    # Calculate the output density matrix at this location as we will need it, and we need to do this before checking whether the current channel 
    # has been initialized in _gradient_mat_memoized because we need n_dim_out for that. 
    vector = get_complex_vector_from_coords(real_vect)
    density_mat = get_density_matrix_from_vector(vector)
    density_mat_out = channel_function(density_mat)
    n_dim_out = density_mat_out.shape[0]

    global _gradient_mat_memoized
    if not name in _gradient_mat_memoized:
        log(f'[grad_vne_single_channel] name `{name}` is not in _gradient_mat_memoized, so attempting to intialize it...', 'debug')
        dim_in_real = real_vect.shape[0]
        _setup_gradient_mat(name, dim_in_real, n_dim_out, channel_function)

        if not name in _gradient_mat_memoized:
            log(f'[grad_vne_single_channel] We did not succeed in initializing the channel with name `name`, we cannot proceed. Aborting...','fatal')
            return None
    
    grad_mat = _gradient_mat_memoized[name]

    id_mat_out = np.eye(n_dim_out)
    I_plus_log_out_regularized = id_mat_out  + linalg.logm(density_mat_out + regularizer*id_mat_out)

    inproduct_x_grad_mat = np.einsum('i,ijkl->jkl', real_vect.reshape((len(real_vect),)), grad_mat)
    inproduct_x_grad_mat_herm = inproduct_x_grad_mat + np.conj( np.transpose(inproduct_x_grad_mat, axes=[0,2,1]) )

    outp = -1 * np.einsum('ijk,kj->i', inproduct_x_grad_mat_herm, I_plus_log_out_regularized)
    
    #Output should be real, but check it if the user wants to.
    outp_real = np.real(outp)
    
    if threshold_check != 0:
        if np.max(np.abs(outp_real-outp)) > threshold_check:
            log(f'[grad_f_single_channel_faster_2] outp_real differs too much from outp, max = {np.max(np.abs(outp_real-outp))}, threshold = {threshold_check}. This is unexpected!','warning')

    #Make sure that outp_real is a column vector
    return outp_real.reshape((len(outp_real),1))