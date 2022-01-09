import utils, random, math, functools, itertools, multiprocessing, logging
from trial import Trial
from typing import List, Dict, Tuple, Union

logging.basicConfig(level=logging.DEBUG)

############################################################################
################################ Main class ################################
############################################################################

class Fermat:
    
    @staticmethod
    def condition(p:int, a:int) -> bool:
        """Fermat condition: if p is prime and a and p are coprime then a^p = a mod p"""
        return pow(a, p-1, p) == 1

    @staticmethod
    def test(n:int, a:Union[int,List[int]]=2, k:int=1) -> bool:
        """The test consists in checking the Fermat condition for several a's"""
        if(type(a) == list):
            k = len(a)
        elif(k > 1):
            a = random.sample(range(2,n), k)
        else:
            a = [a]

        logging.debug(f"Checking Fermat condition for a = {a}")

        for w in a:
            if(Fermat.condition(n, w) == False):
                logging.debug(f"Fermat condition failed for a = {w}")
                return False
        
        return True

    @staticmethod
    def multiprocess_test(n:int, a:Union[int,List[int]]=2, k:int=1, processes:int=2) -> bool:
        """Multiprocess version of the Fermat test"""
        if(processes == 1):
            return Fermat.test(n, a, k)

        if(type(a) == list):
            k = len(a)
        elif(k > 1):
            a = random.sample(range(2,n), k)
        else:
            a = [a]

        logging.debug(f"Checking Fermat condition for a = {a} and {processes} processes")

        length = math.ceil(len(a)/processes)

        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(test_with_event, zip(processes*[n], [a[i:i + length] for i in range(0, len(a), length)]))

        return all(results)

    @staticmethod
    def factorization(n:int, rand:bool=False, start:int=0, stop:int=None) -> Tuple[int,int]:
        """Fermat factorization method: find a^2-b^2=n then n = (a+b)(a-b)"""

        ########################
        a_min = utils.i2root(n) # The a_min is the ceil square root of n. We compute the integer square root of n,
        if(a_min*a_min != n):   # which is the floor square root of n. If a_min^2 is n then n can be factorized as
            a_min += 1          # a_min * a_min, else the ceil square root of n is the floor square root + 1.
        else:                   # We make this to ensure that b is greater than 0
            logging.debug(f"Factorization found: {a_min} * {a_min}")
            return a_min, a_min #
        ########################
        a_max = (n+9)//6 # For odd n so that (a-b) = 3

        if(stop is None):
            stop = a_max
        a_min, a_max = max(a_min, start), min(a_max, stop)

        if(not rand):
            generator = range(a_min, a_max+1)
        else:
            generator = utils.random_range(a_min, a_max+1)
            

        logging.debug(f"Checking Fermat factorization squares for a in [{a_min}, {a_max}] --> {1+a_max-a_min} total checks")
        for a in generator:
            b2 = a*a - n
            b = utils.i2root(b2)
            if(b*b == b2):
                x,y = abs(a+b), abs(a-b)
                logging.debug(f"Factorization found: {x} * {y}")
                return x, y

        logging.debug("No factorization found")
        return n, 1

    @staticmethod
    def multiprocess_factorization(n:int,  rand:bool=False, processes:int=2, start:int=0, stop:int=None):
        """Multiprocess version of Fermat factorization"""
        if(processes == 1):
            return Fermat.factorization(n)

        ########################
        a_min = utils.i2root(n) # The a_min is the ceil square root of n. We compute the integer square root of n,
        if(a_min*a_min != n):   # which is the floor square root of n. If a_min^2 is n then n can be factorized as
            a_min += 1          # a_min * a_min, else the ceil square root of n is the floor square root + 1.
        else:                   # We make this to ensure that b is greater than 0
            logging.debug(f"Factorization found: {a_min} * {a_min}")
            return a_min, a_min #
        ########################
        a_max = (n+9)//6 # For odd n so that (a-b) = 3

        if(stop is None):
            stop = a_max
        a_min, a_max = max(a_min, start), min(a_max, stop)

        length = math.ceil((a_max - a_min + 1)/processes)
        start, stop = [a_min+i*length for i in range(0, processes)], [a_min+i*length-1 for i in range(1, processes+1)]
        stop[-1] = a_max

        logging.debug(f"Checking Fermat factorization squares for a in [{a_min}, {a_max}] --> {1+a_max-a_min} total checks in {processes} processes\n{length} checks per process")
        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(factorization_with_event, zip(processes*[n], start, stop, processes*[rand]))

        results = list(filter(lambda x: x is not None, results))

        if(results == []):
            logging.debug("No factorization found")
            return n, 1
        else:
            return results[0]

    @staticmethod
    def sieve_factorization(n:int, modulus=[9], start:int=0, stop:int=None):

        if(not utils.areCoprime(modulus)):
            raise Exception("Modulus list items must be coprime")

        logging.debug(f"Building square congruences for m in {modulus}")
        congruences = {}
        for m in modulus:
            congruences[m] = []
            residues = utils.sqr_mod_res(m)
            [congruences[m].extend(residues.get(r, [])) for r in [(r+n)%m for r in residues.keys()]]
        
        logging.debug(f"Building generators from congruences = {congruences}")
        generators, M = build_generators(n, congruences, start, stop)

        for a in mod_generator(generators, M):
            b2 = a*a - n
            b = utils.i2root(b2)
            if(b*b == b2):
                x, y = abs(a+b), abs(a-b)
                logging.debug(f"Factorization found: {x} * {y}")
                return x, y

        logging.debug("No factorization found")
        return n, 1

    @staticmethod
    def multiprocess_sieve_factorization(n:int, modulus=[9,11], processes:int=2, start:int=0, stop:int=None):

        if(processes == 1):
            return Fermat.sieve_factorization(n, modulus)

        if(not utils.areCoprime(modulus)):
            raise Exception("Modulus list items must be coprime")

        logging.debug(f"Building square congruences for m in {modulus}")
        congruences = {}
        for m in modulus:
            congruences[m] = []
            residues = utils.sqr_mod_res(m)
            [congruences[m].extend(residues.get(r, [])) for r in [(r+n)%m for r in residues.keys()]]

        logging.debug(f"Building generators from congruences = {congruences}")
        generators, M = multiprocess_build_generators(n, congruences, processes, start, stop)

        length = math.ceil(len(generators)/processes)

        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(sieve_factorization_with_event, zip(processes*[n], [generators[i:i+length] for i in range(0, len(generators), length)], processes*[M]))

        results = list(filter(lambda x: x is not None, results))

        if(results == []):
            logging.debug("No factorization found")
            return n, 1
        else:
            return results[0]

############################################################################
################ Multiprocessing versions need to be global ################
############################################################################

event = None

def setup(event_):
    global event
    event = event_

def test_with_event(p:int , a:List[int]) -> bool:
    """Interruptable version of Fermat test"""
    logging.debug(f"{multiprocessing.current_process().name} checking a in {a}")
    for guess in a:
        if(event.is_set()):
            return False
        
        if(Fermat.condition(p, guess) == False):
            event.set()
            logging.debug(f"Fermat condition failed for a = {guess} in {multiprocessing.current_process().name}")
            return False

    return True

def factorization_with_event(n:int, a_min:int, a_max:int, rand:bool) -> Union[Tuple[int,int], None]:
    """Interruptable version of Fermat factorization"""
    logging.debug(f"{multiprocessing.current_process().name} checking a in [{a_min}, {a_max}]")
    if(not rand):
        generator = range(a_min, a_max+1)
    else:
        generator = utils.random_range(a_min, a_max+1)
    for a in generator:
        if(event.is_set()):
            break
        b2 = a*a - n
        b = utils.i2root(b2)
        x, y = abs(a+b), abs(a-b)
        if(b*b == b2 and x != n and x != 1):
            event.set()
            logging.debug(f"{multiprocessing.current_process().name} Factorization found: {x} * {y}")
            return x, y

    return None

def sieve_factorization_with_event(n:int, generators:List[Tuple[int,int,int]], M:int) -> Union[Tuple[int,int], None]:
    """Interruptable version of Fermat sieved factorization """
    for a in mod_generator(generators, M):
        if(event.is_set()):
            break
        b2 = a*a - n
        b = utils.i2root(b2)
        x, y = abs(a+b), abs(a-b)
        if(b*b == b2 and x != n and x != 1):
            event.set()
            logging.debug(f"{multiprocessing.current_process().name} Factorization found: {x} * {y}")
            return x, y

    return None

def multiprocess_build_generators(n:int, congruences:Dict[int,List[int]], processes:int=2, start:int=0, stop:int=None) -> Tuple[List[Tuple[int,int,int]], int]:
    """Multiprocess version for building the generators"""
    congruences = [list(zip(r,len(r)*[m])) for m,r in congruences.items()]

    M = functools.reduce(lambda x,y:x*y,[m[0][1] for m in congruences])

    a_min = utils.i2root(n)
    if(a_min*a_min != n):
        a_min += 1
    a_max = (n+9)//6

    if(stop is None):
            stop = a_max
    a_min, a_max = max(a_min, start), min(a_max, stop)
    
    logging.debug(f"Total posible a's to check = {1+a_max-a_min}")

    congruences = list(itertools.product(*congruences))
    length = len(congruences)

    with multiprocessing.Pool(processes) as pool:
        generators = pool.starmap(chinese_remainder_solution, zip(congruences, length*[a_min], length*[a_max], length*[M]))

    return generators, M

############################################################################
########################### Supporting functions ###########################
############################################################################

def build_generators(n:int, congruences:Dict[int,List[int]], start:int=0, stop:int=10**10) -> Tuple[List[Tuple[int,int,int]], int]:   
    congruences = [list(zip(r,len(r)*[m])) for m,r in congruences.items()]
    M = functools.reduce(lambda x,y:x*y,[m[0][1] for m in congruences])
    a_min = utils.i2root(n)
    if(a_min*a_min != n):
        a_min += 1
    a_max = (n+9)//6
    if(stop is None):
        stop = a_max
    a_min, a_max = max(a_min, start), min(a_max, stop)
    logging.debug(f"Total posible a's to check = {1+a_max-a_min}")
    generators = []
    for cong in itertools.product(*congruences):
        generators.append(chinese_remainder_solution(cong, a_min, a_max, M))
    return generators, M

def mod_generator(generators:List[Tuple[int,int,int]], M:int):
    max_k = max([sp-st for _, st, sp in generators])
    logging.debug(f"{multiprocessing.current_process().name} Checking total values = {max_k*len(generators)}")
    for k in range(max_k):
        for r,s,_ in generators:
            yield r + (k+s)*M

def chinese_remainder_solution(congr:List[Tuple[int,int]], a_min:int, a_max:int, M:int) -> Tuple[int,int,int]:
    residue = 0
    for r, m in congr:
        c = M//m
        d = utils.modinv(c,m)
        residue += r*c*d
    residue = residue%M
    start, stop = (a_min - residue)//M, 1+(a_max - residue)//M
    return residue, start, stop