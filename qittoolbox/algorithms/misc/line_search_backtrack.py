from typing import Callable
import numpy as np
from ...logging.logger import log

def line_search_backtrack_sphere(x0: np.ndarray, grad: np.ndarray, cost_x0: float, cost_f: 'Callable[[np.ndarray],np.ndarray]', **kwargs) -> 'tuple[float,np.ndarray]':
    """
    Employs a simple backtracking line search algorithm for the gradient descent method on a sphere.

    INPUT:
        x0: np.ndarray of shape (dim,) , the current position
        grad: np.ndarray of shape (dim,), the current gradient at point `x0` of the cost function `cost_f`
        cost_x0: float, the cost of `cost_f` at `x0`, i.e. `cost_x0 = cost_f(x0)`
        cost_f: Callable(np.ndarray) -> np.ndarray, cost function

    OUTPUT:
        tuple with:
            [0] : float, best new cost
            [1]:  np.ndarray of shape (dim,), an approximation to the argmin of `cost_f` along this search direction.
    """
    defaults = {
        'contraction_fact' : 0.5,
        'optimism': 2.0,
        'sufficient_decrease': 1e-4,
        'maxiter_ls' : 25,
        'initial_stepsize': 1,
    }
    for key,val in defaults.items():
        if not key in kwargs:
            kwargs[key] = val
    
    contraction_fact = kwargs['contraction_fact']
    optimism = kwargs['optimism']
    sufficient_decrease = kwargs['sufficient_decrease']
    maxiter_ls = kwargs['maxiter_ls']
    initial_stepsize = kwargs['initial_stepsize']

    descent_dir = -grad #The steepest descent is in the negative direction of the gradient
    norm_grad = np.linalg.norm(grad)

    #Step size 
    alpha = initial_stepsize / norm_grad

    #Compute the cost at the new point
    xt = x0 + alpha * descent_dir
    xt = xt / np.linalg.norm(xt)
    cost_xt = cost_f(xt)[0]  #the output of cost_f is always a numpy array, so take the value out of the array.

    curr_idx = 1

    while cost_xt > cost_x0 - sufficient_decrease * alpha * norm_grad:
        #Reduce step size
        alpha *= contraction_fact

        # Compute the cost at the new point
        xt = x0 + alpha * descent_dir
        xt = xt / np.linalg.norm(xt)
        cost_xt = cost_f(xt)[0]  #the output of cost_f is always a numpy array, so take the value out of the array.
        curr_idx += 1

        if curr_idx >= maxiter_ls:
            log(f'[line_search_backtrack_sphere] The line search did not converge, last cost_xt = {cost_xt}, original cost_x0 = {cost_x0}.','debug')
            break
    
    if cost_xt > cost_x0:
        log(f'[line_search_backtrack_sphere] The line search did not improve the estimate, last cost_xt = {cost_xt}, original cost_x0 = {cost_x0}.','debug')
        return cost_x0, x0
    else:
        log(f'[line_search_backtrack_sphere] Improved the cost from cost_x0 = {cost_x0} to cost_xt = {cost_xt}.','debug')
        return cost_xt, xt

