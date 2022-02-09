import multiprocessing, utils, gmpy2
from typing import Callable, Union, List, Tuple


############################################################################
################################ Main class ################################
############################################################################

class Lucas:

    @staticmethod
    def test(n:int, P_Q:Union[Tuple[int,int],List[Tuple[int,int]],None]=None, processes:int=1) -> bool:
        """Lucas test with parameters P, Q"""
        if(P_Q is None):
            P_Q = _selfridge(n)

        if(type(P_Q) == tuple):
            P_Q = [P_Q]

        if(processes == 1):
            return _lucas_test(n, P_Q)

        else:
            block_size = int(gmpy2.ceil(len(P_Q)/processes))
            P_Q = [P_Q[i*block_size:(i+1)*block_size] for i in range(processes)]
            
            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_lucas_test, zip(processes*[n], P_Q))

            return all(results)

    @staticmethod
    def strong_test(n:int, P_Q:Union[Tuple[int,int],List[Tuple[int,int]],None]=None, processes:int=1) -> bool:
        """Lucas Strong test with parameters P, Q"""
        if(P_Q is None):
            P_Q = _selfridge(n)

        if(type(P_Q) == tuple):
            P_Q = [P_Q]

        if(processes == 1):
            return _lucas_strong_test(n, P_Q)

        else:
            block_size = int(gmpy2.ceil(len(P_Q)/processes))
            P_Q = [P_Q[i*block_size:(i+1)*block_size] for i in range(processes)]
            
            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_lucas_strong_test, zip(processes*[n], P_Q))

            return all(results)

    @staticmethod
    def p_plus_1_factorization(n:int, B:int, checks:int, generator:Callable, processes:int=1) -> Tuple[int,int]:
        """Williams p+1 factorization method"""
        if(processes == 1):
            for P in range(checks):
                l, r = _williams_factorization(n, P, B, generator)
                if(l not in [1,n]):
                    return l, r
        else:
            block_size = int(gmpy2.ceil(checks/processes))
            P = [list(range(checks))[i*block_size:(i+1)*block_size] for i in range(processes)]

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_williams_factorization, zip(processes*[n], P, processes*[B], processes*[generator]))

            results = list(filter(lambda x: x[0] not in [1,n], results))
            if(results != []):
                return results[0]

        return n, 1

############################################################################
########################## Single process versions #########################
############################################################################

def _lucas_test(n, P_Q):
    for P, Q in P_Q:
        if(_lucas_condition(n, P, Q) == False):
            return False
    return True
    
def _lucas_strong_test(n, P_Q):
    for P, Q in P_Q:
        if(_lucas_strong_condition(n, P, Q) == False):
            return False
    return True

def _williams_factorization(n, P, B, generator):
    v_M = P
    for v in generator(start=1, stop=B):
        v_M = _v_lucas_sequence(v_M, v, n)
        g = gmpy2.gcd((v_M-2)%n, n)
        if(g not in [1,n]):
            return g, n//g
    return n, 1

############################################################################
######################### Multiprocessing versions #########################
############################################################################

utils.event = multiprocessing.Event()

def _multiprocess_lucas_test(n, P_Q):
    for P, Q in P_Q:
        if(utils.event.is_set()):
            return False
        D = P**2-4*Q
        nth = n-gmpy2.jacobi(D,n)
        Q %= n
        u, v, q, D, inv_2 = 1, P, Q, (P**2-4*Q)%n, utils.inverse_mod(2, n)
        for bit in bin(nth)[3:]:
            if(utils.event.is_set()):
                return False
            if(bit == "1"):
                u, v, q = ((P*u+v)*v*inv_2-q)%n, ((D*u+P*v)*v*inv_2-P*q)%n, (Q*q*q)%n
            else:
                u, v, q = (u*v)%n, (v**2-2*q)%n, (q*q)%n
        if(u != 0):
            utils.event.set()
            return False
    return True

def _multiprocess_lucas_strong_test(n, P_Q):
    for P, Q in P_Q:
        if(utils.event.is_set()):
                return False
        Q %= n
        D = P**2-4*Q
        s, d = -1, n-gmpy2.jacobi(D,n)
        while not d & 1:
            d, s= d>>1, s+1
        u, v, q, inv_2 = 1, P, Q, utils.inverse_mod(2, n)
        for bit in bin(d)[3:]:
            if(utils.event.is_set()):
                return False
            if(bit == "1"):
                u, v, q = ((P*u+v)*v*inv_2-q)%n, ((D*u+P*v)*v*inv_2-P*q)%n, (Q*q*q)%n
            else:
                u, v, q = (u*v)%n, (v**2-2*q)%n, (q*q)%n
        if(u == 0):
            break
        for _ in range(s):
            if(utils.event.is_set()):
                return False
            if(v == 0):
                break
            u, v, q = (u*v)%n, (v**2 -2*q)%n, (q*q)%n
        if(v != 0):
            utils.event.set()
            return False
    return True

def _multiprocess_williams_factorization(n, P, B, generator):
    for i in P:
        v_M = i
        for v in generator(start=1, stop=B):
            if(utils.event.is_set()):
                return n, 1
            v_M = _v_lucas_sequence(v_M, v, n)
            g = gmpy2.gcd((v_M-2)%n, n)
            if(g not in [1,n]):
                utils.event.set()
                return g, n//g
    return n, 1

############################################################################
########################### Supporting functions ###########################
############################################################################

def _nth_lucas_number(n, P, Q, m):
    Q %= m
    u, v, q, D, inv_2 = 1, P, Q, (P**2-4*Q)%m, utils.inverse_mod(2, m)
    for bit in bin(n)[3:]:
        if(bit == "1"):
            u, v, q = ((P*u+v)*v*inv_2-q)%m, ((D*u+P*v)*v*inv_2-P*q)%m, (Q*q*q)%m
        else:
            u, v, q = (u*v)%m, (v**2-2*q)%m, (q*q)%m
    return u, v, q

def _v_lucas_sequence(v_0, m, n):
    x, y = v_0, (v_0**2-2)%n
    for bit in bin(m)[3:]:
        if(bit == "1"):
            x, y = (x*y - v_0)%n, (y*y - 2)%n
        else:
            y, x = (x*y - v_0)%n, (y*y - 2)%n
    return x

def _lucas_condition(n, P, Q):
    D = P**2-4*Q
    u, _, _ = _nth_lucas_number(n-gmpy2.jacobi(D,n), P, Q, n)
    return  u == 0

def _lucas_strong_condition(n, P, Q):
    D = P**2-4*Q
    s, d = -1, n-gmpy2.jacobi(D,n)
    while not d & 1:
        d, s= d>>1, s+1
    u, v, q = _nth_lucas_number(d, P, Q, n)
    if(u == 0):
        return True
    for _ in range(s):
        if(v == 0):
            return True
        u, v, q = (u*v)%n, (v**2 -2*q)%n, (q*q)%n
    return v == 0

def _selfridge(n):
    D, s = 5, 1
    while(gmpy2.jacobi(D*s,n) == -1):
        D, s = D+2, -1*s
    return 1, (1-D*s)//4
