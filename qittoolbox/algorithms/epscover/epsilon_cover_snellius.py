from datetime import datetime
import multiprocessing as mp
import os
import sys
from typing import Callable
import scipy.linalg as linalg
import numpy as np
import time
import math
import traceback

__VERSION = '1.0.1'

def get_coords_from_angles_v2(angles: np.ndarray) -> np.ndarray:
    """
    Computes a unit vector psi belonging to the n-spherical coordinates `angles`.
    
    INPUT:
        angles : 1-dimensional array of floating point numbers [a1,...,a(n-1)], with 0 <= a(i) <= pi for 1 <= i <= n-2, and 0 <= a(n-1) < 2pi.
    
    OUTPUT:
        psi: 1-dimensional array of floating point numbers [x1,...,x(n)] such that norm(psi) = 1.
    """
    sin_angles_cum = np.concatenate( ( [1], np.cumprod( np.sin(angles) ) ) )
    cos_angles = np.concatenate( ( np.cos(angles), [1] ) )
    return sin_angles_cum * cos_angles

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

def get_density_matrix_from_vector(vector: np.ndarray) -> np.ndarray:
    """
    Changes a complex vector |vector> into the rank-one density matrix |vector><vector|.

    INPUT:
        vector: np.array of shape (n,)

    OUTPUT:
        |vector><vector| np.narray of shape (n,n).
    """
    return np.outer(vector,np.conjugate(vector))

def apply_kraus_operators(kraus_ops: 'list[np.ndarray]', rho: 'np.ndarray') -> 'np.ndarray':
    """
    Applies the channel Phi described by the Kraus operators E(i) to the density matrix rho through sum_i E(i) rho E(i)* . 
    
    INPUT:
        kraus_ops: list of np.ndarray's, each with the same dimensions.
        rho: np.ndarray, representing the density matrix

    OUTPUT:
        np.ndarray representing the quantum channel output.
    """

    return sum( x @ rho @ np.conj(x.T) for x in kraus_ops)

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

def get_sizes_v2(n: int, eps: float) -> 'tuple[int,int,int]':
    """
    Computes the size t1, the size t2, and the total size of the epsilon-cover A2 given the dimension n and the precision eps.

    INPUT:
        n: dimension of the problem
        eps: precision of the epsilon-covering.
    
    OUTPUT:
        sizes: tuple of integers, sizes[0] is t1, sizes[1] is t2, sizes[2] represents the size of the epsilon-cover A2.
    """
    
    delta = get_delta_v2(n,eps)
    t1 = int(1 + np.ceil(np.pi/(2*delta)))
    t2 = int(np.ceil(np.pi/delta))
    size = (t1-2)**(n-2) * t2 + sum( (t1-2)**(n-2-k) * 2 for k in range(1,n-2+1) ) # pow(t1,n-2) * t2
    return t1, t2, size

def get_delta_v2(n: int, eps: float) -> float:
    """
    Computes the maximal delta that can be picked such that the resulting point set is indeed an epsilon-covering.

    INPUT:
        n: dimension of the problem
        eps: precision of the epsilon-covering.
    
    OUTPUT:
        delta: floating point number that gives the maximal discretization possible given a precision <<eps>>.
    """
    
    return eps/math.sqrt(n-1)

def get_angle_indices_from_index_v2(n:int, idx: int, t1: int, t2: int) -> 'list[int]':
    """
    Computes a list of angle indices corresponding to the index given and the A2 epsilon-cover parameters n, t1, t2.

    INPUT:
        n: integer, dimensionality
        idx: integer, should be 0 <= idx < size(A2) = (t1-2)**(n-2) * t2 + sum_k (t1-2)**(n-2-k) * 2 for 1 <= k <= n-2.
        t1: integer, parameter of A2,
        t2: integer, parameter of A2.

    OUTPUT:
        list of ints of length n-1 with angle indices.
    """
    is_interior = idx <= (t1-2)**(n-2) * t2 - 1
    if is_interior:
        out = []
        if idx == 0:
            # NOTE: The interior indices for the interior indices START AT 1!
            return [1]*(n-1)

        #First divide out the last angle idx with t2
        # NOTE: The interior indices for the interior indices START AT 1!
        out.append( int( idx%t2 ) + 1  )
        idx //= t2

        #Then divide out the t1 indices as usual
        while idx > 0:
            # NOTE: The interior indices for the interior indices START AT 1!
            out.append( int(idx%(t1-2)) + 1 )
            idx //= (t1-2)
        
        # Pad the list with 1's if necessary, those are the first angles.
        # NOTE: The interior indices for the interior indices START AT 1!
        return [1]*(n-1-len(out)) + out[::-1]
    
    else:
        #Exterior index offset is defined as max(interior_idx+1) =  (t1-2)**(n-2) * t2
        ext_idx = idx - (t1-2)**(n-2) * t2

        #Loop until we find which exterior index point we are dealing with
        k = 1
        next_offset = (t1-2)**(n-2-k) * 2
        while ext_idx - next_offset >= 0:
            ext_idx -= next_offset

            k += 1
            next_offset = (t1-2)**(n-2-k) * 2

            if k > n-2:
                printer_superprocess(filename='error_superproc.log', msg='ERROR in get_angle_indices_from_index_v2: k > n-2. Dumping output...',\
                                        n=n, idx=idx, t1=t1, t2=t2, is_interior=is_interior, ext_idx=ext_idx,k=k,next_offset=next_offset)
                return None
        
        # We now have an ext_idx that has the correct indices encoded in base-( 2*(t1-2) )
        # Furthermore, the k we previously found tells us how many zeroes we need to pad.
        # We pad the last angle index to 1, so that the last angle calculated as 2pi*(j-1)/t2 equals 0 as well.
        right_pad = [0]*(k-1) + [1]
        out = []
        
        # First divide out the last index, this is either 0 or 1, corresponding to either being 0 or t1-1 (extremal points)
        ext_idx, last_digit = divmod(ext_idx,2)
        out.append( 0 if last_digit == 0 else t1-1)

        while ext_idx > 0:
            ext_idx, digit = divmod(ext_idx, t1-2)
             # NOTE: The interior indices for the interior indices START AT 1!
            out.append( digit+1 )
        
        # First right-pad the zeroes we should add at the end, and immediately reverse the out list
        out = out[::-1] + right_pad

        # Pad the list with 1's if necessary, those are the first angles, note we already reversed the out list here!
        # NOTE: The interior indices for the interior indices START AT 1!
        return [1]*(n-1-len(out)) + out


def printer(pid: int, msg: str=None, sep:str=',', **kwargs) -> int:
    """
    Prints data to a file, using either a message `msg` or keyword arguments

    INPUT:
        pid: int, process id that wishes to log data. This determines the filename!
        msg: str=None, message that is printed to the file, followed by a line separator `\n`
        sep: str=',' , separator
        kwargs: any keyword arguments printed to the file in the format `key=val`, separated by `sep`, followed by a line separator `\n`

    OUTPUT:
        integer: 0 if everything is fine, 1 if an error occurred but the error log worked, 2 if a fatal error occurred.
    """
    filename_error = f"error_pid{pid}.log"
    filename = f"log_pid{pid}.log"
    out_str = sep.join(f'{key}={val}' for key,val in kwargs.items())

    try:
        with open(filename,'a') as f:
            if not msg is None:
                f.write(str(msg)+'\n')

            if not len(kwargs) == 0:
                f.write(out_str+'\n')
            f.close()
        return 0
    except Exception as e:
        try:
            with open(filename_error,'a') as f:
                f.write(f'Could not print to filename {filename}, error: {str(e)}. Tracebacking...\n')
                f.write(traceback.format_exc())
                f.write('\n\nOriginal message:\n')
                f.write(str(msg)+'\n')
                f.write(out_str+'\n')
                f.close()
        except Exception as e2:
            print(f'Fatal error occurred in printer for pid={pid}! \nFirst error={str(e)}\nSecond error={str(e2)}\nTraceback=\n{traceback.format_exc()}\n')
            return 2
        
        return 1

def run_program(n: int, eps: float, delta: float, t1: int, t2: int, size: int, start_idx: int, end_idx: int, queue:'mp.Queue', **kwargs) -> int:
    """
    Calculates the vNE for all angles with start_idx <= angle_idx <= end_idx.

    INPUT:
        #TODO

    OUTPUT:
        #TODO
    """
    LOG_TOTAL_NUMBER = int(1e3)
    PID = os.getpid()

    max_recorded_errorlvl = -1

    errorlvl = printer(PID, msg='START', process_id=PID, n=n, eps=eps, delta=delta, t1=t1, t2=t2, total_size=size, start_idx=start_idx, end_idx=end_idx, my_size=end_idx-start_idx+1)
    if errorlvl == 2:
        return 2
    max_recorded_errorlvl = max(max_recorded_errorlvl, errorlvl)

    errorlvl = printer(PID, msg='start_idx,start_indices,end_idx,end_indices,best_idx,best_indices,best_vNE,time')
    if errorlvl == 2:
        return 2
    max_recorded_errorlvl = max(max_recorded_errorlvl, errorlvl)

    #Manually set up the channel function
    prefac = kwargs['prefac']
    p1 = kwargs['p1']
    A_mat = kwargs['A_mat']
    A_mat_H = kwargs['A_mat_H']
    isometry = kwargs['isometry']
    isometry_H = kwargs['isometry_H']
    def channel_function(rho_in: np.ndarray) -> np.ndarray:
        return prefac * p1 @ A_mat @ np.kron(p1, p1 @ isometry @ rho_in @ isometry_H @ p1) @ A_mat_H @ p1

    log_interval = (end_idx+1-start_idx)//LOG_TOTAL_NUMBER
    log_start_idx = start_idx
    log_next_idx = start_idx + log_interval
    log_start_angle_indices = get_angle_indices_from_index_v2(n,log_start_idx,t1,t2)

    log_time = time.time()

    best_vNE_overall = best_vNE_interval = 9999
    best_idx_overall = best_idx_interval = -1
    best_angle_indices_overall = best_angle_indices_interval = []

    for idx in range(start_idx,end_idx+1):
        # #Call on the function that changes start_angle_indices in place, but only if this is not the first run.
        # if idx > start_idx:
        #     get_next_angle_indices_v2(start_angle_indices,n_for_next,t1,t2)
        angle_indices = get_angle_indices_from_index_v2(n,idx,t1,t2)

        angles = np.array([np.pi * k/(t1-1) for k in angle_indices])
        #Correct the last angle
        angles[n-2] = 2 * np.pi * (angle_indices[n-2]-1) / t2
        
        coords = get_coords_from_angles_v2(angles)
        vector = get_complex_vector_from_coords(coords)
        density_mat = get_density_matrix_from_vector(vector)
        density_mat_out = channel_function(density_mat)
        vNE = get_von_neumann_entropy(density_mat_out)

        if vNE < best_vNE_interval:
            best_vNE_interval = vNE
            best_idx_interval = idx
            best_angle_indices_interval = angle_indices

            if vNE < best_vNE_overall:
                best_vNE_overall = vNE
                best_idx_overall = idx
                best_angle_indices_overall = angle_indices

        if idx >= log_next_idx:
            errorlvl = printer(PID,msg=','.join(str(x) for x in (log_start_idx, log_start_angle_indices, idx, angle_indices, best_idx_interval, best_angle_indices_interval, best_vNE_interval,round(time.time()-log_time,3) )))
            if errorlvl == 2:
                return 2
            max_recorded_errorlvl = max(max_recorded_errorlvl, errorlvl)

            best_vNE_interval = 9999
            best_idx_interval = -1
            log_next_idx += min(log_interval, end_idx)
            log_start_idx = idx+1
            log_start_angle_indices = get_angle_indices_from_index_v2(n,log_start_idx,t1,t2)
            log_time = time.time()
    
    errorlvl = printer(PID, msg='END', best_idx_overall=best_idx_overall, best_angle_indices_overall=best_angle_indices_overall, best_vNE_overall=best_vNE_overall)
    max_recorded_errorlvl = max(max_recorded_errorlvl, errorlvl)

    queue.put((PID,best_idx_overall,best_vNE_overall))

    return max_recorded_errorlvl

def printer_superprocess(filename:str=None ,msg: str=None, sep:str=',', **kwargs) -> int:
    """
    Prints data to a file, using either a message `msg` or keyword arguments

    INPUT:
        filename:str=None, if not set to None will use this as a filename instead
        msg: str=None, message that is printed to the file, followed by a line separator `\n`
        sep: str=',' , separator
        kwargs: any keyword arguments printed to the file in the format `key=val`, separated by `sep`, followed by a line separator `\n`

    OUTPUT:
        integer: 0 if everything is fine, 1 if an error occurred but the error log worked, 2 if a fatal error occurred.
    """
    filename_error = "error_superproc.log"
    filename = filename if not filename is None else "log_superproc.log"
    out_str = sep.join(f'{key}={val}' for key,val in kwargs.items())

    print(out_str)
    try:
        with open(filename,'a') as f:
            if not msg is None:
                f.write(str(msg)+'\n')

            if not len(kwargs) == 0:
                f.write(out_str+'\n')
            f.close()
        return 0
    except Exception as e:
        try:
            with open(filename_error,'a') as f:
                line_out = f'Could not print to filename {filename}, error: {str(e)}. Tracebacking...\n'
                f.write(line_out)
                print(line_out)

                line_out = traceback.format_exc()
                f.write(line_out)
                print(line_out)

                line_out = '\n\nOriginal message:\n'
                f.write(line_out)
                print(line_out)

                line_out = str(msg)+'\n'
                f.write(line_out)
                print(line_out)

                line_out = out_str+'\n'
                f.write(line_out)
                print(line_out)
                
                f.close()
        except Exception as e2:
            print(f'Fatal error occurred in printer_superprocess! \nFirst error={str(e)}\nSecond error={str(e2)}\nTraceback=\n{traceback.format_exc()}\n')
            return 2
        
        return 1

def get_best_vNE_from_all(N_workers: int, queue: 'mp.Queue') -> 'tuple[int,int,int,int]':
    best_pid_overall = -1
    best_vNE_overall = 9999
    best_idx_overall = -1

    try:
        for _ in range(N_workers):
            pid, best_idx, best_vNE = queue.get_nowait()
            if best_vNE < best_vNE_overall:
                best_pid_overall = pid
                best_vNE_overall = best_vNE
                best_idx_overall = best_idx
    except Exception as e:
        printer_superprocess(filename=filename_error,msg=f'ERROR in emptying the queue, e = {e}, tracebacking...', traceback=traceback.format_exc())
        return 2, -1, -1, -1
    
    return 0, best_pid_overall, best_vNE_overall, best_idx_overall

if __name__ == '__main__':
    filename_error = 'error_superproc.log'

    # Dimensionality needs to be set manually! 
    N = 4
    k = l = m = 1
    trace_out_first = True

    # N_workers and eps are set dynamically
    eps = None
    N_workers = None

    #Get arguments from the command line
    if len(sys.argv) != 3: #[<thisfilename>, str(eps), str(N_workers)]
        printer_superprocess(filename=filename_error, msg=f'[Setup error] You specified a wrong number of arguments. I expected 3,' +\
                                f'(0) <filename>, (1) eps, (2) N_workers. Instead I got: \n{sys.argv}')
        sys.exit(2)

    try:
        eps =        float(sys.argv[1])
        N_workers =  int(sys.argv[2])
    except Exception as e:
        printer_superprocess(filename=filename_error,msg=f'Error whilst parsing the command line options. sys.argv={sys.argv}, error = {str(e)}. Tracebacking...',\
                                        traceback=traceback.format_exc())
        sys.exit(2)
   
    # Output
    best_pid_overall = best_idx_overall = -1 
    best_vNE_overall = 9999

    # Get the dimensions of the epsilon cover...
    n_dim = 2*(N-1)-1
    delta = get_delta_v2(n_dim, eps)
    t1, t2, size = get_sizes_v2(n_dim, eps)

    #Get the channel function from file...
    filename = f'data_N{N}_k{k}_l{l}_m{m}_first{trace_out_first}.npz'
    try:
        data = np.load(filename)
        prefac = data['prefac']
        p1 = data['p1']
        A_mat = data['A_mat']
        A_mat_H = data['A_mat_H']
        isometry = data['isometry']
        isometry_H = data['isometry_H']
        kwargs_for_proc = {'prefac':prefac, 'p1':p1, 'A_mat':A_mat, 'A_mat_H':A_mat_H, 'isometry':isometry, 'isometry_H':isometry_H}
    except Exception as e:
        printer_superprocess(filename=filename_error, msg='[Setup error] Could not open numpy file {filename}, error: {str(e)}. Tracebacking...',\
                                traceback=traceback.format_exc())
        sys.exit(2)


    #Overflow will be encountered with `intervals = np.linspace(0,size, dtype=int, num=N_workers+1)`` , so switch to native implementation
    Delta = size//N_workers

    printer_superprocess(msg=f'START PROGRAM. VERSION = {__VERSION}. DATE = {str(datetime.now())}', N_workers=N_workers,eps=eps,N=N,n_dim=n_dim,delta=delta,t1=t1,t2=t2,size=size,Delta=Delta)

    queue = mp.Queue()
    
    try:
        processes = []
        for iidx in range(N_workers):
            start_idx = iidx*Delta
            end_idx = (iidx+1)*Delta-1

            #Let the last worker clear up the rounding error
            if iidx == N_workers-1:
                end_idx = size-1

            #start_angle_indices = get_angle_indices_from_index_v2(n_dim, start_idx, t1_prime, t2)
            
            args_for_proc = (n_dim,eps,delta,t1,t2,size,start_idx,end_idx, queue)
            p = mp.Process(target=run_program, args=args_for_proc, kwargs=kwargs_for_proc )
            processes.append(p)
            printer_superprocess(msg=f'Started process {p} for indices {start_idx} <= idx <= {end_idx}.')
    except Exception as e:
        printer_superprocess(filename=filename_error,msg=f'ERROR in setting up processes, e = {e}, tracebacking...', traceback=traceback.format_exc())
        sys.exit(2)

    printer_superprocess(msg='We have successfully set up all processes. Now starting them...')

    try:
        for idx, p in enumerate(processes):
            p.start()
            printer_superprocess(msg=f'Started process {p}.')
    except Exception as e:
        printer_superprocess(filename=filename_error,msg=f'ERROR in starting processes, e = {e}, tracebacking...', traceback=traceback.format_exc())
        sys.exit(2)
        
    printer_superprocess(msg='We have successfully started all processes. Now waiting for them to finish...')
        
    try:
        for idx, p in enumerate(processes):
            printer_superprocess(msg=f"Now waiting for the join on idx={idx} process={p}...")
            p.join()
    except Exception as e:
        printer_superprocess(filename=filename_error,msg=f'ERROR in joining processes, e = {e}, tracebacking...', traceback=traceback.format_exc())
        sys.exit(2)

    error_code, best_pid_overall, best_vNE_overall, best_idx_overall = get_best_vNE_from_all(N_workers, queue)
    if error_code == 2:
        printer_superprocess(filename=filename_error,msg=f'Seemingly an error occured in emtpying the queue. Aborting...')
        sys.exit(2)
    printer_superprocess(best_pid_overall=best_pid_overall,best_idx_overall=best_idx_overall,best_vNE_overall=best_vNE_overall)
    printer_superprocess(msg="END")