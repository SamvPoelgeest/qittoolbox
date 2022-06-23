from typing import Callable
import numpy as np
from ...logging.logger import log
from ..misc.line_search_backtrack import line_search_backtrack_sphere
from collections import deque
import time

# log = print

def pso_sphere_gd(cost_f: 'Callable[[np.ndarray],np.ndarray]', gradient_f: 'Callable[[np.ndarray],np.ndarray]', dim: int, x0: np.ndarray = None, v0: np.ndarray=None, **kwargs) -> 'dict':
    """
    Implements the Particle Swarm Optimization (PSO) algorithm supplement with gradient descent (GD) on the unit sphere S^{dim-1} as a subset of R^{dim}.

    INPUT:
        cost_f: Callable, will receive an np.ndarray of shape (len(x0), n_particles), must compute the cost function for each column and return
                    an np.ndarray of shape (n_particles,).
        gradient_f: Callable, will receive an np.ndarray of shape (dim,), must compute the gradient and return an np.ndarray of shape (dim,).
        dim: int, dimensionality
        x0: np.ndarray=None, original position of the swarm. If not set, the start positions are randomized.
        v0: np.ndarray=None, original velocities of the swarm. If not set, the velocities will be initialized to 0.
        kwargs: keyword arguments, see below.

    KEYWORD ARGUMENTS:
        maxiter: integer, maximal number of iterations,
        maxiter_gd: int, maximal number of iterations for the gradient descent step,
        n_particles: integer, number of particles,
        nostalgia: float, weight of nostalgia in the PSO algorithm,
        social: float, weight of social in the PSO algorithm,
        inertial_weight: float or Callable(n_iter) -> float, inertial weight of the previous velocity,
        stopping_criterion: Callable(ft,ftp1,ct,ctp1,iter_idx) -> bool, decision to stop the iteration,
        stopping_criterion_gd: Callable(ct_gd, ctp1_gd, iter_idx_gd) -> bool, decision to stop the gradient descent step,
        upper_bound_vel: float, upper bound on the norm of the velocities of the particles,
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
            'vt_proj_hist': 3D np.ndarray of shape (dim,n_particles,history_depth) representing the projected velocities from the last iterations.
    """

    defaults = {
        'maxiter': max(500,4*dim),
        'maxiter_gd': 20,
        'n_particles': min(40,10*dim),
        'nostalgia': 0.5,
        'social': 0.1,
        'inertial_weight' : 1,
        'stopping_criterion': None,
        'stopping_criterion_gd': None,
        'upper_bound_vel': 0.5,
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
    if kwargs['stopping_criterion_gd'] is None:
        #Set default function but using `maxiter_gd` that might be passed by the user.
        kwargs['stopping_criterion_gd'] = lambda ct_gd, ctp1_gd, iter_idx_gd: iter_idx_gd >= kwargs['maxiter_gd']

    #Set kwargs to normal variables for more readability
    maxiter = kwargs['maxiter']
    maxiter_gd = kwargs['maxiter_gd']
    n_particles = kwargs['n_particles']
    nostalgia = kwargs['nostalgia']
    social = kwargs['social']
    inertial_weight = kwargs['inertial_weight']
    stopping_criterion = kwargs['stopping_criterion']
    stopping_criterion_gd = kwargs['stopping_criterion_gd']
    upper_bound_vel = kwargs['upper_bound_vel']
    log_N = kwargs['log_N']   
    
    history_depth = kwargs['history_depth']
    #In the case history_depth is -1, we store all information, which is at most maxiter+1 steps
    if history_depth == -1:
        history_depth = maxiter+1

    
    log(f'[pso_sphere_gd] PSO running with the following parameters: \n{kwargs}','info')

    def sphere_dist(x: np.ndarray, y: np.ndarray, more_accurate=False) -> np.ndarray:
        """Computes the distance between vectors x[:,i] and y[:,i]"""
        euclid_dist = np.linalg.norm(x-y,axis=0)
        great_circle = np.real( 2 * np.arcsin(0.5 * euclid_dist) )
        if more_accurate:
            euclid_dist_inv = np.linalg.norm(x+y,axis=0)
            great_circle_inv = np.pi - np.real( 2 * np.arcsin(0.5 * euclid_dist_inv) )
            return np.where( euclid_dist > 1.9 , great_circle_inv , great_circle )
        return great_circle

    def sphere_proj(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Projects the velocity vectors v[:,i] on the position vectors x[:,i]"""
        outp = v - x * np.sum( np.multiply(x,v), axis=0)
        if np.any(np.isnan(outp)):
            log('[pso_sphere_gd] sphere_proj reports isnan values. This is not expected to happen.','error') #something is going wrong
        return outp

    def sphere_proj_isometric(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Projects the velocity vectors v[:,i] on the position vectors x[:,i] and then rescales them to their original lengths"""
        proj_v = sphere_proj(x,v)
        
        original_v_norm = np.linalg.norm(v, axis=0)
        proj_v_norm = np.linalg.norm(proj_v, axis=0)

        #Rescale only those who are not too close to zero -- otherwise this becomse very inaccurate
        proj_v_norm_nosmall = np.where( proj_v_norm > 1e-10, proj_v_norm, 1 )
        original_v_norm_nosmall = np.where( proj_v_norm > 1e-10, original_v_norm, 1)

        return proj_v / proj_v_norm_nosmall * original_v_norm_nosmall

    def sphere_log(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Takes the logarithm with respect to the vectors x[:,i] and y[:,i]."""
        proj_diff = sphere_proj(x,y-x)
        dist_diff = sphere_dist(x,y)
        proj_diff_norm = np.linalg.norm(proj_diff, axis=0)

        dist_diff_nosmall = np.where( proj_diff_norm > 1e-10, dist_diff, 1 )
        proj_diff_norm_nosmall = np.where( proj_diff_norm > 1e-10, proj_diff_norm, 1)

        outp = proj_diff * dist_diff_nosmall / proj_diff_norm_nosmall
        # outp = np.where( np.isposinf(outp) | np.isneginf(outp) | np.isnan(outp) , 0, outp)
        # np.nan_to_num(outp,copy=False)


        # outp = np.where( (dist_diff > 1e-6) & () , proj_diff * dist_diff / proj_diff_norm , proj_diff )

        if np.any(np.isnan(outp)):
            log('[pso_sphere_gd] sphere_log reports isnan values. This is not expected to happen.','error') #something is going wrong
        return outp
    

    if not x0 is None and x0.shape != (dim,n_particles):
        log(f'[pso_sphere_gd] x0 should have shape (dim,kwargs[n_particles]) = ({dim},{n_particles}), but has x0.shape={x0.shape}. Aborting...','fatal')
        return None

    if x0 is None:
        #Initialize all particles randomly on the sphere. One can do this uniformly by scaling a vector of independent standard normal random variables.
        x0 = np.random.randn(dim,n_particles)
        x0 = x0 / np.linalg.norm(x0, axis=0)
    
    if not v0 is None and v0.shape != (dim,n_particles):
         log(f'[pso_sphere_gd] v0 should have shape (dim,kwargs[n_particles]) = ({dim},{n_particles}), but has v0.shape={v0.shape}. Aborting...','fatal')


    if v0 is None:
        v0 = np.random.randn(dim,n_particles)
        v0 = sphere_proj(x0, v0)
        v0 = v0 / np.linalg.norm(v0,axis=0) * upper_bound_vel

    ft = np.zeros((n_particles,)) #cost function values
    
    ft = cost_f(x0)
    it = np.argmin(ft) #best index
    ct = ft[it] #best cost
    gt = x0[:,it] #best position
    #Make sure that gt is always stored as column vector to help np broadcasting
    gt.shape = (len(gt),1)

    log(f'[pso_sphere_gd] After init, found best cost ct = {ct} located at index it = {it}' + (f', pos = {gt.T}.' if dim < 10 else '') ,'info')
    xt = x0 #particle position
    vt = v0 #particle speed
    pt = np.copy(xt) #best particle position

    # xtp1 = np.zeros(xt.shape,dtype=xt.dtype)
    # vtp1 = np.zeros(vt.shape,dtype=vt.dtype)
    ptp1 = np.zeros(pt.shape,dtype=pt.dtype)
    ftp1 = np.zeros(ft.shape,dtype=ft.dtype)
    itp1 = np.copy(it)
    ctp1 = np.copy(ct)
    gtp1 = np.copy(gt)

    xt_history = deque(maxlen=history_depth)
    vt_history = deque(maxlen=history_depth)
    vt_proj_history = deque(maxlen=history_depth)

    xt_history.append(xt)
    vt_history.append(vt)
    vt_proj_history.append(sphere_proj(xt,vt))

    log(f'[pso_sphere_gd] Initialized, now starting the loop...','info')
    #Do not print the best position if the dimension is too big
    header_str = f'[pso_sphere_gd] {"% done" :>8}   {"idx" :>8}   {"weight":>10}   {"ct":>20}   {"it":>8}   {"time":>10}'
    if dim < 6:
         header_str += f'  gt'
    log(header_str, 'info')
    
    log_nextstop = maxiter//log_N
    
    curr_idx = 0
    stop_reached = False
    gt_changed_this_iter = True

    _t_since_start = time.time()
    while not stop_reached:
        curr_idx += 1
        curr_weight = inertial_weight(curr_idx)

        if curr_idx >= log_nextstop:
            progress_str = f'[pso_sphere_gd] {round(curr_idx/maxiter*100,2):>8}   {curr_idx:>8}   {curr_weight:>10.3e}   {ct:>20.12e}   {it:>8}   {round(time.time()-_t_since_start,1):>10}'
            if dim < 6:
                progress_str += f'  {gt.T}'
            log(progress_str,'info')
            log_nextstop += maxiter//log_N

        

        random_nostalgia = np.random.uniform(low=0,high=nostalgia,size=(dim,n_particles))
        random_social = np.random.uniform(low=0,high=social,size=(dim,n_particles))

        vt =  curr_weight * sphere_proj_isometric(xt,vt) +\
                np.multiply(random_nostalgia,sphere_log(xt,pt)) +\
                np.multiply(random_social, sphere_log(xt,gt) )
        
        #Bound vtp1
        vt_norms = np.linalg.norm(vt,axis=0)

        vt = np.where( vt_norms <= upper_bound_vel , vt, vt/vt_norms * upper_bound_vel)

        xt = xt + vt
        #Retraction of x(t) is simply to normalize the new vectors
        xt = xt / np.linalg.norm(xt, axis=0)

        ftp1 = cost_f(xt)
        ptp1 = np.where(ftp1<ft, xt, pt)
        itp1 = np.argmin(ftp1)

        if ftp1[itp1] < ct:
            ctp1 = ftp1[itp1]
            gtp1 = xt[:,itp1,None]
            gt_changed_this_iter = True
        else:
            ctp1 = ct

        # Evaluate now, before overwriting the new positions and such, in case the stopping criterion wants to make a comparison (i.e. how much the best cost
        # has decreased between to iterations.
        stop_reached = stopping_criterion(ft,ftp1,ct,ctp1,curr_idx)

        # xt = xtp1
        # vt = vtp1
        ft = ftp1
        pt = ptp1
        it = itp1
        ct = ctp1
        gt = gtp1

        if history_depth > 0:
            xt_history.append(xt)
            vt_history.append(vt)
            vt_proj_history.append(sphere_proj(xt,vt))

        #Run the gradient descent algorithm on the best position
        if gt_changed_this_iter:
            log(f'[pso_sphere_gd] Using gradient descent method as the global optimum has changed since last time, currently ct = {ct}...','debug')
            gt_gd = np.copy(gt)
            ct_gd = ct

            stop_reached_gd = False
            iter_idx_gd = 0
            while not stop_reached_gd:
                iter_idx_gd += 1

                grad = gradient_f(gt_gd)
                #Change the Euclidean gradient into a tangent vector to pass to the line searching algorithm
                tangent = sphere_proj(gt_gd, grad)

                ctp1_gd, gt_gd = line_search_backtrack_sphere(gt_gd, tangent, ct_gd, cost_f)
                
                stop_reached_gd = stopping_criterion_gd(ct_gd, ctp1_gd, iter_idx_gd)
                
                ct_gd = ctp1_gd
            
            #Check if we improved the estimate, if so, update the global optimum 
            if ct_gd < ct:
                log(f'[pso_sphere_gd] Updated the global optimum using gradient descent, from ct={ct} to ct_gd={ct_gd}.','debug')
                ct = ct_gd
                gt = gt_gd
        
        gt_changed_this_iter = False

    #Always log a final time
    progress_str = f'[pso_sphere_gd] {round(curr_idx/maxiter*100,2):>8}   {curr_idx:>8}   {curr_weight:>10.3e}   {ct:>20.12e}   {it:>8}   {round(time.time()-_t_since_start,1):>10}'
    if dim < 6:
        progress_str += f'  {gt.T}'
    log('-'*len(progress_str),'info')
    log(progress_str,'info')

    #Change the deque to a 3D numpy array of shape (dim, n_particles, history_depth)
    xt_3d = None
    vt_3d = None
    vt_proj_3d = None
    if history_depth > 0:
        #It might be possible that we terminated in less than `history_depth` steps, in this case just continue until the queue is empty
        range_end = min(history_depth, len(xt_history))

        xt_3d = np.zeros((dim,n_particles,range_end))
        vt_3d = np.zeros((dim,n_particles,range_end))
        vt_proj_3d = np.zeros((dim,n_particles,range_end))

        
        for i in range( range_end ):
            xt_3d[:,:,i] = xt_history.popleft()
            vt_3d[:,:,i] = vt_history.popleft()
            vt_proj_3d[:,:,i] = vt_proj_history.popleft()

    return { 'curr_idx':curr_idx, 'xt':xt, 'vt':vt, 'ft':ft, 'pt':pt, 'it':it, 'ct':ct, 'gt':gt, 'xt_hist':xt_3d, 'vt_hist':vt_3d, 'vt_proj_hist':vt_proj_3d }
