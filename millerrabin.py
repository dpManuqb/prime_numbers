import utils, random, math, multiprocessing, logging
from typing import Union, List, Tuple

logging.basicConfig(level=logging.DEBUG)

############################################################################
################################ Main class ################################
############################################################################

class MillerRabin:

    @staticmethod
    def condition(n:int, a:int) -> bool:
        """Miller-Rabin condition for n = d * 2^s + 1 and a coprime to n: 
        if n is prime then a^d = +-1 mod n or a^(d*2^r) = -1 mod n for r
        in [0, s-1]"""
        #####################
        s, d = -1, n-1      # n as d * 2^s + 1
        while(not d & 1):   #
            s, d = s+1, d>>1#
        #####################

        x = pow(a, d, n) # a^d mod n

        if(x == 1 or x == n-1):
            return True

        ########################
        for _ in range(s):     # 
            x = pow(x, 2, n)   # a^(d*2^r) = -1 mod n for r in [0, s-1]
            if(x == n - 1):    #
                return True    #
            # elif(x == 1):    # This only happens when n is a Carmichael number,
            #     return False # which are rare
        ########################

        return False

    @staticmethod
    def test(n:int, a:Union[int,List[int]]=2, k:int=1) -> bool:
        """The Miller Rabin test consists of checking the condition for several a's"""
        if(type(a) == list):
            k = len(a)
        elif(k > 1):
            a = random.sample(range(2,n), k)
        else:
            a = [a]

        logging.debug(f"Checking Miller-Rabin condition for a = {a}")

        for w in a:
            if(MillerRabin.condition(n, w) == False):
                logging.debug(f"Miller-Rabin condition failed for a = {w}")
                return False

        return True

    @staticmethod
    def multiprocess_test(n:int, a:Union[int,List[int]]=2, k:int=1, processes:int=2) -> bool:
        """Multiprocess version of the Miller Rabin test"""
        if(processes == 1):
            return MillerRabin.test(n, a, k)

        if(type(a) == list):
            k = len(a)
        elif(k > 1):
            a = random.sample(range(2,n), k)
        else:
            a = [a]

        length = math.ceil(len(a)/processes)
        logging.debug(f"Checking Miller-Rabin condition for a = {a} and {processes} processes")

        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(test_with_event, zip(processes*[n], [a[i:i + length] for i in range(0, len(a), length)]))

        return all(results)

    @staticmethod
    def proof(n:int) -> bool:
        """If the ERH is True then this function is a deterministic primality test"""
        return MillerRabin.test(n, a=list(range(2, 1+math.floor(math.log(n)**2))))

    @staticmethod
    def multiprocess_proof(n:int, processes:int=2) -> bool:
        """Multiprocess version of the Miller-Rabin proof"""
        return MillerRabin.multiprocess_test(n, a=list(range(2, 1+math.floor(math.log(n)**2))), processes=processes)

    @staticmethod
    def factorization(n:int, a:Union[int,List[int]]=2, k:int=1) -> Tuple[int,int]:
        """"Miller-Rabin factorization only for Carmichael numbers"""
        def _factorization(b:int):
            x = pow(b, d, n)

            if(x == 1 or x == n-1):
                return n, 1

            for _ in range(s):
                x_prev, x = x, pow(x, 2, n)
                if(x == n - 1):
                    return n, 1
                elif(x == 1):
                    factor = utils.gcd(x_prev-1, n)
                    return factor, n//factor

            return n, 1

        s, d = -1, n-1
        while(not d & 1):
            s, d = s+1, d>>1

        if(type(a) == list):
            k = len(a)
        elif(k > 1):
            a = random.sample(range(2,n), k)
        else:
            a = [a]
        
        logging.debug(f"Checking Miller-Rabin condition for a = {a}")

        for base in a:
            f,r = _factorization(base)
            if(f != 1 and r != 1):
                logging.debug(f"Factorization faound: {f} * {r}")
                return f, r
        
        logging.debug("Factorization not found")
        return n, 1

    @staticmethod
    def multiprocess_factorization(n:int, a:Union[int,List[int]]=2, processes:int=2) -> Tuple[int,int]:

        s, d = -1, n-1
        while(not d & 1):
            s, d = s+1, d>>1

        if(type(a) == int):
            a = [a]

        event = multiprocessing.Event()
        with multiprocessing.Pool(processes, setup, (event,)) as pool:
            results = pool.starmap(factorization_with_event, zip(len(a)*[n], len(a)*[d], len(a)*[s], a))
        
        for f,r in results:
            if(f != 1 and r != 1):
                return f, r
                
        return n, 1
            
############################################################################
################ Multiprocessing versions need to be global ################
############################################################################

event = None

def setup(event_):
    global event
    event = event_

def condition_with_event(n:int, a:int):

    if(event.is_set()):
        return False

    s, d = -1, n-1
    while(not d & 1):
        s, d = s+1, d>>1

    x = pow(a, d, n)

    if(x == 1 or x == n-1):
        return True

    for _ in range(s):
        if(event.is_set()):
            return False
        x = pow(x, 2, n)
        if(x == n - 1):
            return True

    return False

def test_with_event(n:int, a:List[int]):
    logging.debug(f"{multiprocessing.current_process().name} checking a in {a}")
    for w in a:
        guess = condition_with_event(n, w)
        if(guess == False):
            event.set()
            logging.debug(f"Fermat condition failed for a = {guess} in {multiprocessing.current_process().name}")
            return False

    return guess

def factorization_with_event(n:int, d:int, s:int, a:int):

    if(event.is_set()):
        return None

    x = pow(a, d, n)

    if(x == 1 or x == n-1):
        return n, 1

    for _ in range(s):
        if(event.is_set()):
            return None
        x_prev, x = x, pow(x, 2, n)
        if(x == n - 1):
            return n, 1
        elif(x == 1):
            factor = utils.gcd(x_prev-1, n)
            if(factor != n and factor != 1):
                event.set()
            return factor, n//factor

    return None