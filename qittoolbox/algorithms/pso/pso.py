from typing import Callable
import numpy as np
from ...logging.logger import log
from collections import deque
import time

def pso(cost_f: 'Callable[[np.ndarray],np.ndarray]', dim: int, x0: np.ndarray = None, v0: np.ndarray=None, **kwargs) -> 'dict':
    """
    Implements the Particle Swarm Optimization (PSO) algorithm.

    INPUT:
        cost_f: Callable, will receive an np.ndarray of shape (len(x0), n_particles), must compute the cost function for each column and return
                    an np.ndarray of shape (n_particles,).
        dim: int, dimensionality
        x0: np.ndarray=None, original position of the swarm. If not set, the start positions are randomized.
        v0: np.ndarray=None, original velocities of the swarm. If not set, the velocities will be initialized to 0.
        kwargs: keyword arguments, see below.

    KEYWORD ARGUMENTS:
        maxiter: integer, maximal number of iterations
        n_particles: integer, number of particles
        nostalgia: float, weight of nostalgia in the PSO algorithm,
        social: float, weight of social in the PSO algorithm,
        inertial_weight: float or Callable(n_iter) -> float, inertial weight of the previous velocity,
        stopping_criterion: Callable(ft,ftp1,ct,ctp1,iter_idx) -> bool, decision to stop the iteration,
        lower_bound_pos: np.ndarray of dtype float with shape (dim,), lower bounds on the starting positions of the particles,
        upper_bound_pos: np.ndarray of dtype float with shape (dim,), upper bounds on the starting positions of the particles,
        lower_bound_vel: np.ndarray of dtype float with shape (dim,), lower bounds on the velocities of the particles,
        upper_bound_vel: np.ndarray of dtype float with shape (dim,), upper bounds on the velocities of the particles,
        log_N: integer, number of times information is logged to logging.logger.log,
        history_depth: int, set to positive int `t` to store the last `t` velocities and positions in a np.ndarray of shape (dim,n_particles,t). Might decrease performance!
            If set to `0`, no historical data is stored. If set to `-1`, *all* data is stored.

    OUTPUT:
        dictionary with key,val:
            'curr_idx': the index of the last iteration, 
            'xt': the last positions of the particles, 2D np.ndarray of shape (dim,n_particles), 
            'vt': the last velocities of the particles, 2D np.ndarray of shape (dim,n_particles),
            'ft': the last cost vector of all particles, 1D np.ndarray of length n_particles, 
            'pt': the last best position of each particle, 2D np.ndarray of shape (dim,n_particles),
            'it': the last best index of the best position, integer,
            'ct': the last best cost for all particles, float, 
            'gt': the last best position of all particles, 1D np.ndarray of shape (dim,)
            'xt_hist': 3D np.ndarray of shape (dim,n_particles,history_depth) representing the positions from the last iterations,
            'vt_hist': 3D np.ndarray of shape (dim,n_particles,history_depth) representing the velocities from the last iterations,
    """

    defaults = {
        'maxiter': max(500,4*dim),
        'n_particles': min(40,10*dim),
        'nostalgia': 1.4,
        'social': 1.4,
        'inertial_weight' : None,
        'stopping_criterion': None,
        'lower_bound_pos': -1 * np.ones((dim,)),
        'upper_bound_pos': 1  * np.ones((dim,)),
        'lower_bound_vel': None,
        'upper_bound_vel': None,
        'log_N': 10,
        'history_depth': 0,
    }

    #Fill up kwargs with default values if not set
    for key,val in defaults.items():
        if not key in kwargs:
            kwargs[key] = val

    if kwargs['inertial_weight'] is None:
        #Set default function but using `maxiter` that might be passed by the user.
        kwargs['inertial_weight'] =  lambda n_iter: 0.4 + 0.5 * (1 - n_iter / kwargs['maxiter'])
    elif not callable(kwargs['inertial_weight']):
        # If `inertial_weight` was set but was not callable, then assume it is a value that should always be used
        _inertial_weight_val = kwargs['inertial_weight']
        kwargs['inertial_weight'] = lambda n_iter: _inertial_weight_val

    if kwargs['stopping_criterion'] is None:
        #Set default function but using `maxiter` that might be passed by the user.
        kwargs['stopping_criterion'] = lambda ft, ftp1, ct, ctp1, iter_idx : iter_idx >= kwargs['maxiter']

    if kwargs['lower_bound_vel'] is None:
        kwargs['lower_bound_vel'] = -(kwargs['upper_bound_pos']-kwargs['lower_bound_pos'])/10
    if kwargs['upper_bound_vel'] is None:
        kwargs['upper_bound_vel'] = -kwargs['lower_bound_vel']

    #Set kwargs to normal variables for more readability
    n_particles = kwargs['n_particles']
    lower_bound_pos = kwargs['lower_bound_pos']
    upper_bound_pos = kwargs['upper_bound_pos']
    lower_bound_vel = kwargs['lower_bound_vel']
    upper_bound_vel = kwargs['upper_bound_vel']
    stopping_criterion = kwargs['stopping_criterion']
    log_N = kwargs['log_N']
    maxiter = kwargs['maxiter']
    inertial_weight = kwargs['inertial_weight']
    nostalgia = kwargs['nostalgia']
    social = kwargs['social']

    history_depth = kwargs['history_depth']
    #In the case history_depth is -1, we store all information, which is at most maxiter+1 steps
    if history_depth == -1:
        history_depth = maxiter+1

    
    log(f'[pso] PSO running with the following parameters: \n{kwargs}','info')
    

    if not x0 is None and x0.shape != (dim,n_particles):
        log(f'[pso] x0 should have shape (dim,kwargs[n_particles]) = ({dim},{n_particles}), but has x0.shape={x0.shape}. Aborting...','fatal')
        return None

    
    lower_bound_pos.shape = (len(lower_bound_pos),1)
    upper_bound_pos.shape = (len(upper_bound_pos),1)

    if x0 is None:
        x0 = np.random.uniform( low= lower_bound_pos , high= upper_bound_pos , size = (dim,n_particles ) )
    
    if not v0 is None and v0.shape != (dim,n_particles):
         log(f'[pso] v0 should have shape (dim,kwargs[n_particles]) = ({dim},{n_particles}), but has v0.shape={v0.shape}. Aborting...','fatal')

    
    lower_bound_vel.shape = (len(lower_bound_vel),1)
    upper_bound_vel.shape = (len(upper_bound_vel),1)

    if v0 is None:
        v0 = np.random.uniform( low = lower_bound_vel , high = upper_bound_vel , size = (dim,n_particles) )

    ft = np.zeros((n_particles,)) #cost function values
    
    ft = cost_f(x0)
    it = np.argmin(ft) #best index
    ct = ft[it] #best cost
    gt = x0[:,it] #best position

    log(f'[pso] After init, found best cost ct = {ct} located at index it = {it}, pos = {gt}.','info')
    xt = x0 #particle position
    vt = v0 #particle speed
    pt = np.copy(xt) #best particle position

    xtp1 = np.zeros(xt.shape,dtype=xt.dtype)
    vtp1 = np.zeros(vt.shape,dtype=vt.dtype)
    ptp1 = np.zeros(pt.shape,dtype=pt.dtype)
    ftp1 = np.zeros(ft.shape,dtype=ft.dtype)
    itp1 = np.copy(it)
    ctp1 = np.copy(ct)
    gtp1 = np.zeros(gt.shape, dtype=gt.dtype)

    xt_history = deque(maxlen=history_depth)
    vt_history = deque(maxlen=history_depth)

    xt_history.append(xt)
    vt_history.append(vt)

    log(f'[pso] Initialized, now starting the loop...','info')
    #Do not print the best position if the dimension is too big
    header_str = f'[pso] {"% done" :>8}   {"idx" :>8}   {"weight":>10}   {"ct":>20}   {"it":>8}   {"time":>10}'
    if dim < 6:
         header_str += f'  gt'
    log(header_str, 'info')

    
    log_nextstop = maxiter//log_N
    
    curr_idx = 0
    stop_reached = False

    _t_since_start = time.time()
    while not stop_reached:
        curr_idx += 1
        if curr_idx >= log_nextstop:
            progress_str = f'[pso] {round(curr_idx/maxiter*100,2):>8}   {curr_idx:>8}   {curr_weight:>10.3e}   {ct:>20.12e}   {it:>8}   {round(time.time()-_t_since_start,1):>10}'
            if dim < 6:
                progress_str += f'  {gt.T}'
            log(progress_str,'info')
            log_nextstop += maxiter//log_N

        curr_weight = inertial_weight(curr_idx)

        random_nostalgia = np.random.uniform(low=0,high=nostalgia,size=(dim,n_particles))
        random_social = np.random.uniform(low=0,high=social,size=(dim,n_particles))

        vtp1 =  curr_weight * vt + np.multiply(random_nostalgia,pt) + np.multiply(random_social.T, gt ).T -\
                np.multiply(random_nostalgia+random_social, xt)

        
        #Bound vtp1
        vtp1 = np.where( vtp1 <= upper_bound_vel , vtp1, upper_bound_vel)
        vtp1 = np.where( vtp1 >= lower_bound_vel , vtp1, lower_bound_vel )

        xtp1 = xt + vt

        ftp1 = cost_f(xtp1)
        ptp1 = np.where(ftp1<ft, xtp1, pt)
        itp1 = np.argmin(ftp1)

        if ftp1[itp1] < ct:
            ctp1 = ftp1[itp1]
            gtp1 = xtp1[:,itp1]
        else:
            ctp1 = ct

        # Evaluate now, before overwriting the new positions and such, in case the stopping criterion wants to make a comparison (i.e. how much the best cost
        # has decreased between to iterations.
        stop_reached = stopping_criterion(ft,ftp1,ct,ctp1,curr_idx)

        xt = xtp1
        vt = vtp1
        ft = ftp1
        pt = ptp1
        it = itp1
        ct = ctp1
        gt = gtp1

        if history_depth > 0:
            xt_history.append(xt)
            vt_history.append(vt)

    #Always log a final time
    progress_str = f'[pso] {round(curr_idx/maxiter*100,2):>8}   {curr_idx:>8}   {curr_weight:>10.3e}   {ct:>20.12e}   {it:>8}   {round(time.time()-_t_since_start,1):>10}'
    if dim < 6:
        progress_str += f'  {gt.T}'
    log('-'*len(progress_str),'info')
    log(progress_str,'info')

    #Change the deque to a 3D numpy array of shape (dim, n_particles, history_depth)
    xt_3d = None
    vt_3d = None
    if history_depth > 0:
        #It might be possible that we terminated in less than `history_depth` steps, in this case just continue until the queue is empty
        range_end = min(history_depth, len(xt_history))

        xt_3d = np.zeros((dim,n_particles,range_end))
        vt_3d = np.zeros((dim,n_particles,range_end))

        for i in range( range_end ):
            xt_3d[:,:,i] = xt_history.popleft()
            vt_3d[:,:,i] = vt_history.popleft()
    
    return  { 'curr_idx':curr_idx, 'xt':xt, 'vt':vt, 'ft':ft, 'pt':pt, 'it':it, 'ct':ct, 'gt':gt, 'xt_hist':xt_3d, 'vt_hist':vt_3d }
