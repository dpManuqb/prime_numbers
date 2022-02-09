import multiprocessing, gmpy2, utils
from typing import Callable

############################################################################
################################ Main class ################################
############################################################################

class AKS:
    
    @staticmethod
    def full_prove(n:int, processes:int=1) -> bool:
        """AKS full prove"""
        if(processes == 1):
            return _full_prove(n)
        
        else:
            start, stop = 1, (n>>1 + 1)
            block_size = int(gmpy2.ceil((stop-start+1)/processes))
            start, stop, _stop = list(range(start, processes*block_size, block_size)), list(range(block_size, (processes+1)*block_size, block_size)), stop
            stop[-1] = _stop
        
            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_full_prove, zip(processes*[n], start, stop))

            return all(results)
    
    @staticmethod
    def prove(n:int, factorize:Callable, processes:int=1) -> bool:
        """AKS prove"""
        log_2_2, r = gmpy2.floor(gmpy2.log2(n)**2), 2
        while(r < n):
            g = gmpy2.gcd(r,n)
            if(g != 1 and g != n):
                return False
            if(utils.order(n,r) > log_2_2):
                break
            r += 1

        if(r == n):
            return True

        start, stop = 1, int(gmpy2.floor(gmpy2.sqrt(utils.phi(factorize(r)))*gmpy2.log2(n)))

        if(processes == 1):
            for a in range(start, stop+1):
                if(not _prove(n, r, a)):
                    return False

            return True

        else:
            block_size = int(gmpy2.ceil((stop-start+1)/processes))
            start, stop, _stop = list(range(start, processes*block_size, block_size)), list(range(start+block_size-1, (processes+1)*block_size, block_size)), stop
            stop[-1] = _stop

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_prove, zip(processes*[n], processes*[r], start, stop))
            
            return all(results)

############################################################################
############################## Single Process ##############################
############################################################################

def _full_prove(n):
    comb = 1
    for k in range(1, 1+(n>>1)):
        comb = ((n-k+1)*comb)//k
        if(comb%n != 0):
            return False
    return True

def _prove(n, r, a):
    return _pow_poly([a, 1], n, r, n) == [a] + [0]*(n%r - 1) + [1]

############################################################################
############################# Multiprocessing ##############################
############################################################################

def _multiprocess_full_prove(n, start, stop):
    comb = gmpy2.comb(n, start-1)
    for k in range(start, stop+1):
        if(utils.event.is_set()):
            break
        comb = ((n-k+1)*comb)//k
        if(comb%n != 0):
            utils.event.set()
            return False
    return True

def _multiprocess_prove(n, r, start, stop):
    for a in range(start, stop+1):
        if(utils.event.is_set()):
            return False
        if(_pow_poly([a, 1], n, r, n) != [a] + [0]*(n%r - 1) + [1]):
            utils.event.set()
            return False
    return True

############################################################################
########################### Supporting functions ###########################
############################################################################

def _norm_poly(P):
    i = len(P) - 1
    while(P[i] == 0 and i > 0):
        i -= 1
    return P[:i+1]

def _mult_poly(P, Q):
    result = [0]*(len(P)+len(Q)-1)
    for i,a in enumerate(P):
        for j,b in enumerate(Q):
            result[i+j] += a*b
    return _norm_poly(result)

def _mod_poly(P, r, m):
    result = [0]*(r+1)
    for i,a in enumerate(P):
        result[i%r] += a
    return _norm_poly([a%m for a in result])

def _pow_poly(P, n, r, m):
    base, result = P, [1]
    while(n > 0):
        if(n & 1):
            result = _mod_poly(_mult_poly(result, base), r, m)
        base = (base * base).mod(r, m)
        base = _mod_poly(_mult_poly(base, base), r, m)
        n >>= 1
    return result