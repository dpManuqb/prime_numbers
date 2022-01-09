from bpsw import BPSW
import utils
import math
import multiprocessing
from polinomial import Poly
from trial import Trial
from pollard import Pollard
from typing import Dict

############################################################################
########################### Supporting functions ###########################
############################################################################

def factorize_(n:int) -> Dict[int,int]:

    f,r = Trial.factorization(n)
    
    if(r != 1 and BPSW.test(r)):
        f[r] = 1
    elif(r != 1):
        f1,f2 = Pollard.p_1_factorization(r)
        if(f1 != 1 and f1 != r):
            if(f1 != f2):
                f[f1], f1 = 1, 1
            else:
                f[f1], f1, f2 = 2, 1, 1
        if(f2 != 1 and f2 != r and f1 != f2):
            f[f2], f2 = 1, 1
        r = f1*f2
        f[r] = 1
        
    return f

############################################################################
################################ Main class ################################
############################################################################

class AKS:
    
    @staticmethod
    def full_test(n:int) -> bool:

        comb = 1
        for k in range(1, 1+(n>>1)):
            comb = ((n-k+1)*comb)//k
            if(comb%n != 0):
                return False

        return True

    @staticmethod
    def multiprocess_full_test(n:int, processes:int=2) -> bool:

        length = (1+(n>>1))//processes
        
        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(full_test_with_event, zip(processes*[n], [i*length + 1 for i in range(processes)], [(i+1)*length + 1 for i in range(processes)]))

        return all(results)

    @staticmethod
    def test(n:int) -> bool:

        log_2_2 = math.floor(math.log2(n)**2)
        r = 2
        while(r < n):
            g = utils.gcd(r,n)
            if(g != 1 and g != n):
                return False
            if(utils.order(n,r) > log_2_2):
                break
            r += 1

        if(r == n):
            return True

        guess = [0]*(n%r + 1)
        guess[-1] = 1
        right = Poly(guess)

        for a in range(1, math.floor(math.sqrt(utils.phi(factorize_(r)))*math.log2(n))+1):
            left = Poly([a, 1]).pow(n, r, n)
            right.coefs[0] = a
            if(left != right):
                return False

        return True
        
    @staticmethod
    def multiprocess_test(n:int, processes:int=2) -> bool:
        
        log_2_2 = math.floor(math.log2(n)**2)
        r = 2
        while(r < n):
            g = utils.gcd(r,n)
            if(g != 1 and g != n):
                return False
            if(utils.order(n,r) > log_2_2):
                break
            r += 1

        if(r == n):
            return True

        length = math.floor(math.sqrt(utils.phi(factorize_(r)))*math.log2(n))

        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(test_with_event, zip(processes*[n], processes*[r], [i*length + 1 for i in range(processes)], [(i+1)*length + 1 for i in range(processes)]))

        return all(results)


############################################################################
################ Multiprocessing versions need to be global ################
############################################################################

event = None

def setup(event_):
    global event
    event = event_

def full_test_with_event(n:int, k_min:int, k_max:int):

    comb = utils.ncr(n, k_min-1)

    for k in range(k_min, k_max):
        if(event.is_set()):
            break
        comb = ((n-k+1)*comb)//k
        if(comb%n != 0):
            event.set()
            return False

    return True

def test_with_event(n:int, r:int, k_min:int, k_max:int):

    guess = [0]*(n%r + 1)
    guess[-1] = 1
    right = Poly(guess)

    for a in range(k_min, k_max):
        if(event.is_set()):
            return True
        left = Poly([a, 1]).pow(n, r, n)
        right.coefs[0] = a
        if(left != right):
            event.set()
            return False

    return True