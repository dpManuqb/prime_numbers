import utils, logging, multiprocessing, math, functools
from typing import List, Dict

logging.basicConfig(level=logging.DEBUG)

############################################################################
################################ Main class ################################
############################################################################

class Trial:
    
    @staticmethod
    def test(n:int, B:int=None) -> bool:

        if(B is None):
            B = utils.primeList[-1]
        B = min([utils.i2root(n), B])

        for p in utils.primeList:
            if(p > B):
                break
            if(n%p == 0):
                return True

        return False

    @staticmethod
    def multiprocess_test(n:int, B:int, processes:int=2) -> bool:

        if(B is None):
            B = utils.primeList[-1]
        B = min([utils.i2root(n), B])

        length = math.ceil((1 + utils.primeList.index(B))/processes)
        start, stop = [i*length for i in range(0, processes)], [i*length-1 for i in range(1, processes+1)]

        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(test_with_event, zip(processes*[n], start, stop))

        return all(results)

    @staticmethod
    def factorization(n:int, B:int=None) -> Dict[int,List[int]]:

        if(B is None):
            B = utils.primeList[-1]
        B = min([utils.i2root(n), B])

        factorization = {}
        for p in utils.primeList:
            if(p > B):
                break
            exp = 0
            while(n%p == 0):
                n //= p
                exp += 1
            if(exp != 0):
                factorization[p] = exp

        return factorization, n

    @staticmethod
    def multiprocess_factorization(n:int, B:int, processes:int=2) -> Dict[int,List[int]]:
        
        if(B is None):
            B = utils.primeList[-1]
        B = min([utils.i2root(n), B])

        length = math.ceil((1 + utils.primeList.index(B))/processes)
        start, stop = [i*length for i in range(0, processes)], [i*length-1 for i in range(1, processes+1)]

        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(factorization_with_event, zip(processes*[n], start, stop))

        f = functools.reduce(utils.merge_factors, results)
        return f, f//utils.n_from_factorization(f)

############################################################################
################ Multiprocessing versions need to be global ################
############################################################################

event = None

def setup(event_):
    global event
    event = event_

def test_with_event(n:int, x_min:int, x_max:int):
    for i in range(x_min, x_max+1):
        if(event.is_set()):
            return False
        if(n%utils.primeList[i] == 0):
            event.set()
            return False
    return True

def factorization_with_event(n:int, x_min:int, x_max:int):
    factorization = {}
    for i in range(x_min, x_max+1):
        p = utils.primeList[i]
        exp = 0
        while(n%p == 0):
            n //= p
            exp += 1
        if(exp != 0):
            factorization[p] = exp
    return factorization