import multiprocessing, utils, functools, gmpy2
from typing import Dict, List, Tuple, Union

############################################################################
################################ Main class ################################
############################################################################

class Trial:
    @staticmethod
    def trial_test(n:int, start:int=2, stop:Union[int,None]=None, processes:int=1) -> bool:
        """Trial division primality test"""
        root = gmpy2.isqrt(n)
        if(stop is None):
            stop = root

        stop = min(root, stop)

        if(start == 2):
            if(not n & 1):
                return False
            start = 3

        if(processes == 1):
            return _trial_test(n, start, stop)

        else:
            block_size = int(gmpy2.ceil((stop-start+1)/processes))
            start, stop, _stop = list(range(start, processes*block_size, block_size)), list(range(start+block_size-1, (processes+1)*block_size, block_size)), stop
            stop[-1] = _stop

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_trial_test, zip(processes*[n], start, stop))

        return all(results)

    @staticmethod
    def trial_factorization(n:int, start:int=2, stop:Union[int,None]=None) -> Tuple[Dict[int,int],int]:
        """Trial division factorization"""
        return _trial_factorization(n, start, stop)

    @staticmethod
    def eratosthenes_sieve(n:int) -> List[int]:
        """Eratosthenes prime list"""
        return _eratosthenes_sieve(n)

    @staticmethod
    def eratosthenes_test(n:int, stop:Union[int,None]=None) -> bool:
        """Eratosthenes primality test"""
        return _eratosthenes_test(n, stop)

    @staticmethod
    def eratosthenes_factorization(n:int, stop:Union[int,None]=None) -> Tuple[Dict[int,int],int]:
        """Eratosthenes factorization"""
        return _eratosthenes_factorization(n, stop)

    @staticmethod
    def prime_list_test(n:int, start:int=0, stop:Union[int,None]=None, processes:int=1) -> bool:
        """Test based on precomputed prime list"""
        root_prime_index, root = 0, gmpy2.isqrt(n)
        while(utils.prime_list[root_prime_index] < root):
            root_prime_index += 1
            if(root_prime_index == len(utils.prime_list)):
                root_prime_index = len(utils.prime_list)-1
                break
        if(stop is None):
            stop = root_prime_index

        stop = min(root_prime_index, stop)

        if(processes == 1):
            return _prime_list_test(n, start, stop)

        else:
            block_size = int(gmpy2.ceil((stop-start+1)/processes))
            start, stop, _stop = list(range(start, processes*block_size, block_size)), list(range(start+block_size-1, (processes+1)*block_size, block_size)), stop
            stop[-1] = _stop

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_prime_list_test, zip(processes*[n], start, stop))

            return all(results)

    @staticmethod
    def prime_list_factorization(n:int, start:int=0, stop:Union[int,None]=None, processes:int=1) -> Tuple[Dict[int,int],int]:
        """Factorization based on precomputed prime list"""
        full = True
        root_prime_index, root = 0, gmpy2.isqrt(n)
        while(utils.prime_list[root_prime_index] < root):
            root_prime_index += 1
            if(root_prime_index == len(utils.prime_list)):
                root_prime_index = len(utils.prime_list)-1
                full = False
                break
        if(stop is None):
            stop = root_prime_index
        stop = min(root_prime_index, stop)

        if(processes == 1):
            f,r = _prime_list_factorization(n, start, stop)

        else:
            block_size = int(gmpy2.ceil((stop-start+1)/processes))
            start, stop, _stop = list(range(start, processes*block_size, block_size)), list(range(start+block_size-1, (processes+1)*block_size, block_size)), stop
            stop[-1] = _stop

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_prime_list_factorization, zip(processes*[n], start, stop))

            f = functools.reduce(utils.merge_factors, [x for x,y in results])
            r = n//utils.n_from_factorization(f)
            
        if(full and n != 1):
            f[n], n, r = 1, 1, 1
        
        return f, r
        
############################################################################
########################## Single process versions #########################
############################################################################

def _trial_test(n, start, stop):
    for p in range(start, stop+1, 2):
        if(n%p == 0):
            return False
    return True

def _trial_factorization(n, start, stop):
    root = gmpy2.isqrt(n)
    full = False
    if(stop is None):
        stop = root
        full = True
    max_p = min(root, stop)
    f, p = {}, start
    if(start == 2):
        exp = 0
        while(not n & 1):
            exp += 1
            n >>= 1
        p += 1
    while(n > 1 and p <= max_p):
        exp = 0
        while(n%p == 0):
            exp += 1
            n //= p
        if(exp != 0):
            f[p] = exp
        p += 2
    if(full and n != 1):
        f[n], n = 1, 1
    return f, n

def _eratosthenes_sieve(n):
    sieve = list(range(2,n+1))
    for p in sieve:
        for q in sieve[sieve.index(p)+1:]:
            if(q%p == 0):
                sieve.remove(q)
    return sieve

def _eratosthenes_test(n, stop):
    root = gmpy2.isqrt(n)
    if(stop is None):
        stop = root
    max_p = min(root, stop)
    sieve = list(range(2,max_p+1))
    for p in sieve:
        if(n%p == 0):
            return False
        for q in sieve[sieve.index(p)+1:]:
            if(q%p == 0):
                sieve.remove(q)
    return True

def _eratosthenes_factorization(n, stop):
    root = gmpy2.isqrt(n)
    full = False
    if(stop is None):
        stop = root
        full = True
    max_p = min(root, stop)
    sieve = list(range(2,max_p+1))
    f = {}
    for p in sieve:
        exp = 0
        while(n%p == 0):
            exp += 1
            n //= p
        if(exp != 0):
            f[p] = exp
        if(n == 1):
            break
        for q in sieve[sieve.index(p)+1:]:
            if(q%p == 0):
                sieve.remove(q)
    if(full and n != 1):
        f[n], n = 1, 1
    return f, n

def _prime_list_test(n, start, stop):
    for i in range(start, stop+1):
        p = utils.prime_list[i]
        if(n%p == 0):
            return False
    return True

def _prime_list_factorization(n, start, stop):
    f = {}
    for i in range(start, stop+1):
        p = utils.prime_list[i]
        exp = 0
        while(n%p == 0):
            exp += 1
            n //= p
        if(exp != 0):
            f[p] = exp
        i += 1
        if(n == 1):
            break
    return f, n

############################################################################
######################### Multiprocessing versions #########################
############################################################################

def _multiprocess_trial_test(n, start, stop):
    if(not start & 1):
        start += 1
    for p in range(start, stop+1, 2):
        if(utils.event.is_set()):
            return False
        if(n%p == 0):
            utils.event.set()
            return False
    return True

def _multiprocess_prime_list_test(n, start, stop):
    for i in range(start, stop+1):
        if(utils.event.is_set()):
            return False
        p = utils.prime_list[i]
        if(n%p == 0):
            utils.event.set()
            return False
    return True