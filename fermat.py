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

    @staticmethod
    def factorization(n:int, processes:int=1, sieve:bool=False, F:List[int]=[7,9,11,13,16]):
        """Fermat factorization assuming n is odd and not a perfect square"""
        a_min, a_max = utils.introot(n)[0]+1, (n+9)//6

        if(sieve):
            congruences, M = _build_congruences(n, F, a_min, a_max)
            congruences = list(sorted(congruences, key=lambda c: c[0]))
        if(processes == 1):
            if(not sieve):
                generator = range(a_min, a_max)
            else:
                generator = _congruences_generator(congruences, M)

            return _factorization(n, generator)

        else:
            if(not sieve):
                block_size = int(gmpy2.ceil((a_max-a_min+1)/processes))
                a_min, a_max, _a_max = list(range(a_min, processes*block_size, block_size)), list(range(a_min+block_size-1, (processes+1)*block_size, block_size)), a_max
                a_max[-1] = _a_max
                generators = [range(start,stop+1) for start,stop in zip(a_min,a_max)]
                M = None
            else:
                block_size = int(gmpy2.ceil(len(congruences)/processes))
                generators = [congruences[i*block_size:(i+1)*block_size] for i in range(processes)]

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_factorization, zip(processes*[n], generators, processes*[M]))

            results = list(filter(lambda x: x[0] not in [1,n], results))

            if(results == []):
                return n, 1
            else:
                return results[0]

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

def _factorization(n, generator):
    for a in generator:
        b2 = a*a - n
        b, check = utils.introot(b2)
        if(check):
            x, y = abs(a+b), abs(a-b)
            return x, y
    return n, 1

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

def _multiprocess_factorization(n, generator, M):
    if(type(generator) != range):
        generator = _congruences_generator(generator, M) # Cannot pickle generator so its created here
    for a in generator:
        if(utils.event.is_set()):
            return n, 1
        b2 = a*a - n
        b, check = utils.introot(b2)
        if(check):
            utils.event.set()
            x, y = abs(a+b), abs(a-b)
            return x, y
    return n, 1

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

def _merge_combine_congruences(congruences, a_min, a_max):   
    congruences = [list(zip(r,len(r)*[m])) for m,r in congruences.items()]
    M = functools.reduce(lambda x,y:x*y,[m[0][1] for m in congruences])
    result = []
    for cong in itertools.product(*congruences):
        result.append(utils.chinese_remainder(cong, a_min, a_max, M))
    return result, M

def _congruences_generator(congruences, M):
    if(len(congruences) > 0):
        max_k = max([sp-st+1 for _, st, sp in congruences])
        for k in range(max_k):
            for r,s,_ in congruences:
                yield r + (k+s)*M

def _build_congruences(n, F, a_min, a_max):
    if(not utils.are_coprime(F)):
        raise Exception("F list items must be coprime")
    congruences = {}
    for m in F:
        congruences[m] = []
        residues = utils.quadratic_residues(m)
        [congruences[m].extend(residues.get(r, [])) for r in [(r+n)%m for r in residues.keys()]]
    return _merge_combine_congruences(congruences, a_min, a_max)