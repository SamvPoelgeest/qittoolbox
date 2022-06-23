import scipy.sparse as sparse
from ..logging.logger import log
from math import prod, sqrt

_GLOBAL_N = 3

def q0_param(N: int=_GLOBAL_N) -> float:
    """
    Quantum parameter q0 = 1/N * 2 / (1 + sqrt(1-4/N^2) ) \in (0,1]
    INPUT:
        N: integer, dimensionality
    OUTPUT:
        q0(N), float in (0,1]
    """
    return 1.0/N * 2.0 / (1.0 + sqrt(1.0-4.0/(N*N)) )

def q0_bracket(x: int, N: int=None, q0: float=None, check_threshold: float=1e-10, round_ans:bool=True) -> 'int|float':
    """
    Quantum bracket [x]_{q0}, defined by [k+1]_{q0} := q0^{-k} * (1-q0^{2k+2})/(1-q0^2).
    You can either pass the dimensionality N or the quantum parameter q0. 

    INPUT:
        x: int, x = k+1
        N: int=None, dimensionality
        q0: float=None, quantum parameter.
        check_threshold: float=1e-10, threshold check whether the output is indeed an integer (also see round_ans)
        round_ans: bool=True, set this to False if you do not want to round the answer
    OUTPUT:
        [x]_{q0}, integer in case round_ans, otherwise float.
    """
    if N is None and q0 is None:
        log('[q0_bracket] both N and q0 are None. Cannot continue.','fatal')
        return 0
    if not N is None and not q0 is None:
        log(f'[q0_bracket] Both N={N} and q0={q0} are not-None values, will pick N to define q0, but be warned!','warning')
    
    if not N is None:
        q0 = q0_param(N)

    if abs(q0-1) < check_threshold:
        if N == 2 or q0 == 1:
            log(f'[q0_bracket] N was explicitly passed as the value 2 or q0 with value 1, in this case, [x] = x, so returning the passed value x = {x}','debug')
            return x
        else:
            log(f'[q0_bracket] q0 is extremely close to 1. This may lead to numerical errors. Are you sure you want this? q0 =  {q0}.','warning')
    
    #The original function is expressed as [k+1]_{q0}, so use k+1 = x
    k = x - 1
    val = pow(q0,-k) * (1 - pow(q0,2*k+2))/(1-pow(q0,2))

    if round_ans and abs(val-round(val)) > check_threshold:
        log(f'[q0_bracket] val={val} differs from round(val)={round(val)} by {abs(val-round(val))}, whilst only {check_threshold} is allowed. Still returning rounded val!','warning')

    if round_ans:    
        return int(round(val))
    else:
        return val
        
def q0_factorial(x: int, N: int=None, q0: float=None, check_threshold: float=1e-10, round_ans:bool=True) -> 'int|float':
    """
    Quantum factorial [x]_q ! := [x]_q [x-1]_q ... [2]_q [1]_q
    You can either pass the dimensionality N or the quantum parameter q0. 
    INPUT:
        x: int, x = k+1
        N: int=None, dimensionality
        q0: float=None, quantum parameter
        check_threshold: float=1e-10, threshold check whether the output of q0_bracket is an integer.
        round_ans: bool=True, set this to False if you do not want to round the answer from q0_bracket.

    OUTPUT:
        integer of float [x]_q!
    """
    return prod( q0_bracket(i,N=N,q0=q0,check_threshold=check_threshold,round_ans=round_ans) for i in range(1,x+1) )

def theta_q0(k: int, l: int, m: int, N: int=None, q0: float=None, check_threshold: float=1e-10, round_ans:bool=True) -> float:
    """
    Quantum theta net theta_q(k,l,m).
    You can either pass the dimensionality N or the quantum parameter q0. 
    INPUT:
        k: int
        l: int
        m: int
        N: int=None, dimensionality
        q0: float=None, quantum parameter
        check_threshold: float=1e-10, threshold check whether the output of q0_bracket is an integer (also see round_ans).
        round_ans: bool=True, set this to False if you do not want to round the answer from q0_bracket.

    OUTPUT:
        theta_q(k,l,m) float
    """
    #k = l + m -2r
    r = float(l+m-k)/2.0
    if not r.is_integer():
        log(f'[theta_q0] r=(l+m-k)/2 = ({l}+{m}-{k})/2 = {r} is not an integer. Aborting.','fatal')
        return 0.0
    
    r = int(r)
    
    numerator = q0_factorial(r,     N=N,q0=q0,check_threshold=check_threshold,round_ans=round_ans) * \
                q0_factorial(l-r,   N=N,q0=q0,check_threshold=check_threshold,round_ans=round_ans) * \
                q0_factorial(m-r,   N=N,q0=q0,check_threshold=check_threshold,round_ans=round_ans) * \
                q0_factorial(k+r+1, N=N,q0=q0,check_threshold=check_threshold,round_ans=round_ans)
    denominator =   q0_factorial(l,   N=N,q0=q0,check_threshold=check_threshold,round_ans=round_ans) * \
                    q0_factorial(m,   N=N,q0=q0,check_threshold=check_threshold,round_ans=round_ans) * \
                    q0_factorial(k,   N=N,q0=q0,check_threshold=check_threshold,round_ans=round_ans)
    return numerator/denominator

def BC16_C_q0(N: int=None, q0: float=None, threshold: float=1e-10, maxiter: int=100) -> float:
    """
    Computes the function C(q0) defined as C(q0) = (1-q0^2)^{-1/2} * [ prod_{s=1}^{infty} 1/(1-q0^{2s}) ]^{3/2}.
    You can either pass the dimensionality N or the quantum parameter q0. 
    INPUT:
        N: int=None, dimensionality
        q0: float=None, quantum parameter
        threshold: float=1e-10, cut-off value for the infinite sum, once the term 1/(1-q^{2s}) comes threshold-close to 1.
        maxiter: int, cut-off index for the infinite sum. Raises warning if reached. Heuristically, 100 has never been reached.
    OUTPUT:
        C(q0), float.
    """
    if N is None and q0 is None:
        log('[BC16_C_q0] both N and q0 are None. Cannot continue.','fatal')
        return 0.0
    if not N is None and not q0 is None:
        log(f'[BC16_C_q0] Both N={N} and q0={q0} are not-None values, will pick N to define q0, but be warned!','warning')
    
    if not N is None:
        q0 = q0_param(N)

    prefac = pow(1-pow(q0,2), -1.0/2.0)
    prodfac = 1.0
    idx = 1
    while idx < maxiter+1:
        newfac = 1.0/(1-pow(q0,2*idx))
        prodfac *= pow(newfac,3.0/2.0)
        if abs(newfac-1) < threshold:
            break
        idx += 1

    if idx == maxiter+1:
        log(f'[BC16_C_q0] The product did not converge fully, last newfac={newfac}, threshold={threshold}.','warning')
    return prefac*prodfac


