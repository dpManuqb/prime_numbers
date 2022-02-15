import utils, random, functools, itertools, multiprocessing, gmpy2
from typing import List,  Union

############################################################################
################################ Main class ################################
############################################################################

class Fermat:

    @staticmethod
    def test(n:int, a_list:Union[None,int,List[int]]=None, k:int=1, processes:int=1) -> bool:
        """Fermat pseudoprimality test"""
        if(a_list is None):
            if(k == 1):
                a_list = [2]
            else:
                a_list = [random.randrange(2, n-1) for _ in range(k)]
        elif(type(a_list) == int):
            a_list = [a_list]

        if(processes == 1):
            return _test(n, a_list)

        else:
            block_size = int(gmpy2.ceil(len(a_list)/processes))
            a_list = [a_list[i*block_size:(i+1)*block_size] for i in range(processes)]

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_test, zip(processes*[n], a_list))

            return all(results)

    @staticmethod
    def strong_test(n:int, a_list:Union[None,int,List[int]]=None, k:int=1, processes:int=1) -> bool:
        """MillerRabin pseudoprimality test"""
        s, d = -1, n-1      
        while(not d & 1):   
            s, d = s+1, d>>1

        if(a_list is None):
            if(k == 1):
                a_list = [2]
            else:
                a_list = [random.randrange(2, n-1) for _ in range(k)]
        elif(type(a_list) == int):
            a_list = [a_list]

        if(processes == 1):
            return _strong_test(n, s, d, a_list)

        else:
            block_size = int(gmpy2.ceil(len(a_list)/processes))
            a_list = [a_list[i*block_size:(i+1)*block_size] for i in range(processes)]
            
            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_strong_test, zip(processes*[n], processes*[s], processes*[d], a_list))

            return all(results)

    @staticmethod
    def proof(n:int, processes:int=1) -> bool:
        """If the ERH is True then this function is a deterministic primality test"""
        return Fermat.strong_test(n, a_list=list(range(2, int(1+gmpy2.floor(utils.log(n)**2)))), processes=processes)

############################################################################
########################## Single process versions #########################
############################################################################

def _test(n, a_list):
    for a in a_list:
        if(not _fermat_condition(n, a)):
            return False
    return True

def _strong_test(n, s, d, a_list):
    for a in a_list:
        if(not _miller_rabin_condition(n, s, d, a)):
            return False
    return True

############################################################################
######################### Multiprocessing versions #########################
############################################################################

def _multiprocess_test(n, a_list):
    for a in a_list:
        if(utils.event.is_set()):
            return False
        if(not _fermat_condition(n, a)):
            utils.event.set()
            return False
    return True

def _multiprocess_strong_test(n, s, d, a_list):
    for a in a_list:
        if(utils.event.is_set()):
            return False
        x = pow(a, d, n)
        if(x == 1 or x == n-1):
            continue
        prime = False
        for _ in range(s):
            if(utils.event.is_set()):
                return False 
            x = pow(x, 2, n)
            if(x == n - 1):
                prime = True
                break
        if(not prime):
            utils.event.set()
            return False
    return True

############################################################################
################################## Support #################################
############################################################################

def _fermat_condition(n, a):
    return pow(a, n-1, n) == 1

def _miller_rabin_condition(n, s, d, a):
    x = pow(a, d, n)
    if(x == 1 or x == n-1):
        return True
    for _ in range(s):     
        x = pow(x, 2, n)
        if(x == n - 1):    
            return True
    return False