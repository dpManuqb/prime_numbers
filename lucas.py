from typing import Union, List, Tuple
import multiprocessing
import utils

############################################################################
########################### Supporting functions ###########################
############################################################################

def steps(n:int) -> List[int]:

    result = []
    while(n > 1):
        if(n & 1):
            n -= 1
            result.insert(0,1)
        else:
            result.insert(0,0)
        n //= 2

    return result

def lucas_number(n:int, P:int, Q:int, m:int) -> Tuple[int,int,int]:

    Q %= m
    u, v, q, D, inv_2, n_steps = 1, P, Q, (P**2-4*Q)%m, utils.modinv(2, m), steps(n)

    for s in n_steps:
        if(s):
            u, v, q = ((P*u + v)*v*inv_2 - q)%m, ((D*u + P*v)*v*inv_2 - P*q)%m, (Q*q*q)%m
        else:
            u, v, q = (u*v)%m, (v**2 -2*q)%m, (q*q)%m

    return u, v, q

def selfridge(n:int) -> Tuple[int,int]:
    
    D, s = 5, 1
    while(utils.jacobi(D,n) == -1):
        D, s = D+2, -1*s

    return 1, (1-D*s)//4

############################################################################
################################ Main class ################################
############################################################################

class Lucas:

    @staticmethod
    def condition(n:int, P:int, Q:int) -> bool:

        D = (P**2-4*Q)%n
        u, _, _ = lucas_number(n-utils.jacobi(D,n), P, Q, n)

        return  u == 0

    @staticmethod
    def test(n:int, P_Q:Union[Tuple[int,int],List[Tuple[int,int]]]=None) -> bool:

        if(P_Q is None):
            P_Q = selfridge(n)

        if(type(P_Q) == tuple):
            P_Q = [P_Q]

        for P,Q in P_Q:
            if(Lucas.condition(n, P, Q) == False):
                return False

        return True

    
    @staticmethod
    def multiprocess_test(n:int, P_Q:Union[Tuple[int,int],List[Tuple[int,int]]]=None, processes:int=2):
        
        if(processes == 1):
            return Lucas.test(n, P_Q)

        if(P_Q is None):
            P_Q = selfridge(n)

        if(type(P_Q) == tuple):
            P_Q = [P_Q]

        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(condition_with_event, zip(len(P_Q)*[n], *zip(*P_Q)))

        return all(results)

    @staticmethod
    def strong_condition(n:int, P:int, Q:int) -> bool:

        D = (P**2-4*Q)%n

        s, d = -1, n-utils.jacobi(D,n)
        while not d & 1:
            d, s= d>>1, s+1

        u, v, q = lucas_number(d, P, Q, n)

        if(u == 0):
            return True

        for _ in range(s):
            if(v == 0):
                return True
            u, v, q = (u*v)%n, (v**2 -2*q)%n, (q*q)%n

        return v == 0

    @staticmethod
    def strong_test(n:int, P_Q:Union[Tuple[int,int],List[Tuple[int,int]]]=None) -> bool:

        if(P_Q is None):
            P_Q = selfridge(n)

        if(type(P_Q) == tuple):
            P_Q = [P_Q]

        for P,Q in P_Q:
            if(Lucas.strong_condition(n, P, Q) == False):
                return False

        return True

    @staticmethod
    def multiprocess_strong_test(n:int, P_Q:Union[Tuple[int,int],List[Tuple[int,int]]]=None, processes:int=2) -> bool:
        
        if(processes == 1):
            return Lucas.strong_condition(n, P_Q)

        if(P_Q is None):
            P_Q = selfridge(n)

        if(type(P_Q) == tuple):
            P_Q = [P_Q]

        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(strong_condition_with_event, zip(len(P_Q)*[n], *zip(*P_Q)))

        return all(results)

############################################################################
################ Multiprocessing versions need to be global ################
############################################################################

event = None

def setup(event_):
    global event
    event = event_

def lucas_number_with_event(n:int, P:int, Q:int, m:int) -> Tuple[int,int,int]:

    Q %= m
    u, v, q, D, inv_2, n_steps = 1, P, Q, (P**2-4*Q)%m, utils.modinv(2, m), steps(n)

    for s in n_steps:
        if(event.is_set()):
            return None, None, None
        if(s):
            u, v, q = ((P*u + v)*v*inv_2 - q)%m, ((D*u + P*v)*v*inv_2 - P*q)%m, (Q*q*q)%m
        else:
            u, v, q = (u*v)%m, (v**2 -2*q)%m, (q*q)%m

    return u, v, q

def condition_with_event(n:int, P:int, Q:int) -> bool:
    D = (P**2-4*Q)%n
    u, _, _ = lucas_number_with_event(n-utils.jacobi(D,n), P, Q, n)

    if(u is None):
        return True

    if(u != 0):
        event.set()

    return  u == 0

def strong_condition_with_event(n:int, d:int, s:int, P:int, Q:int) -> bool:

    u, v, q = lucas_number_with_event(d, P, Q, n)

    if(u == 0):
        return True

    for _ in range(s):
        if(event.is_set()):
            return True
        if(v == 0):
            return True
        u, v, q = (u*v)%n, (v**2 -2*q)%n, (q*q)%n

    if(v != 0):
        event.set()

    return v == 0
