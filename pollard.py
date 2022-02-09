import gmpy2, random, primes, multiprocessing, utils
from typing import Callable


############################################################################
################################## Support #################################
############################################################################

def _pseudorandom(x, a, m):
    return (x*x + a)%m

############################################################################
################################ Main class ################################
############################################################################

class Pollard:
    
    @staticmethod
    def rho_floyd_factorization(n:int, group:int=100, func:Callable=_pseudorandom, tries:int=-1, processes:int=1):
        """Pollard Rho factorization with floyd algorithm for cicle detection"""
        if(processes == 1):
            return _rho_floyd_factorization(n, group, func, tries)
        
        else:
            if(tries == -1):
                tries = processes*[-1]
            else:
                tries = processes*[int(gmpy2.ceil(tries/processes))]

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_rho_floyd_factorization, zip(processes*[n], processes*[group], processes*[func], tries))
            
            results = list(filter(lambda x: x[0] not in [1,n], results))

            if(results == []):
                return n, 1
            else:
                return results[0]

    @staticmethod
    def rho_brent_factorization(n:int, func:Callable=_pseudorandom, tries:int=-1, processes:int=1):
        """Pollard Rho factorization with floyd algorithm for cicle detection"""
        if(processes == 1):
            return _rho_brent_factorization(n, func, tries)
        
        else:
            if(tries == -1):
                tries = processes*[-1]
            else:
                tries = processes*[int(gmpy2.ceil(tries/processes))]

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_rho_brent_factorization, zip(processes*[n], processes*[func], tries))
            
            results = list(filter(lambda x: x[0] not in [1,n], results))

            if(results == []):
                return n, 1
            else:
                return results[0]

    @staticmethod
    def p_minus_1_factorization(n:int, B1:int, B2:int, generator:Callable):
        """Pollard p-1 method"""
        return _p_minus_1_factorization(n, B1, B2, generator)

############################################################################
########################## Single process versions #########################
############################################################################

def _rho_floyd_factorization(n, group, func, tries):
    t = 0
    while(t > tries):
        c, s = random.randrange(1, n-2), random.randrange(0, n)
        x, y, g, total, product = s, s, 1, 0, 1
        while(g == 1):
            xs, ys = x, y
            for _ in range(group):
                x = func(x, c, n)
                y = func(func(y, c, n), c, n)
                product, total = (product*abs(x-y))%n, total + 1
            g, product, total = gmpy2.gcd(product, n), 1, 0
        if(g != n):
            return g, n//g
        elif(group > 1):
            g = 1
            while(g == 1):
                    xs = func(xs, c, n)
                    ys = func(func(ys, c, n), c, n)
                    g = gmpy2.gcd(abs(xs-ys), n)
            if(g != n):
                return g, n//g
        t += 1
    return n, 1

def _rho_brent_factorization(n, func, tries):
    t = 0
    while(t > tries):
        y, c, m = random.randrange(1, n), random.randrange(1, n), random.randrange(1, n)
        g, r, q = 1, 1, 1
        while(g == 1):
            x = y
            for _ in range(r):
                y = func(y, c, n)
            k = 0
            while(k < r and g == 1):
                ys = y
                for _ in range(min(m, r - k)):
                    y = func(y, c, n)
                    q = q*abs(x - y)%n
                g = gmpy2.gcd(q, n)
                k = k + m
            r *= 2
        if(g == n):
            while(True):
                ys = func(ys, c, n)
                g = gmpy2.gcd(abs(x - ys), n)
                if(g > 1):
                    break
        if(g != n):
            return g, n//g
    return n, 1

def _p_minus_1_factorization(n:int, B1:int, B2:int, generator:Callable):
    a_m = gmpy2.mpz(random.randrange(2, n))
    coprime = gmpy2.gcd(a_m,n)
    if(coprime != 1):
        return coprime, n//coprime
    for m in generator(start=1, stop=B1):
        a_m = pow(a_m, m, n)
        factor = gmpy2.gcd((a_m-1)%n, n)
        if(factor != 1 and factor != n):
            return factor, n//factor
    for m in primes.prime_generator(start=B1+1, stop=B2):
        Q = pow(a_m, m, n) - 1
        factor = gmpy2.gcd(Q%n, n)
        if(factor != 1 and factor != n):
            return factor, n//factor
    return n, 1

############################################################################
######################### Multiprocessing versions #########################
############################################################################

def _multiprocess_rho_floyd_factorization(n, group, func, tries):
    t = 0
    while(t > tries):
        c, s = random.randrange(1, n-2), random.randrange(0, n)
        x, y, g, total, product = s, s, 1, 0, 1
        while(g == 1):
            xs, ys = x, y
            for _ in range(group):
                if(utils.event.is_set()):
                    return n, 1
                x = func(x, c, n)
                y = func(func(y, c, n), c, n)
                product, total = (product*abs(x-y))%n, total + 1
            g, product, total = gmpy2.gcd(product, n), 1, 0
        if(g != n):
            utils.event.set()
            return g, n//g
        elif(group > 1):
            g = 1
            while(g == 1):
                if(utils.event.is_set()):
                    return n, 1
                xs = func(xs, c, n)
                ys = func(func(ys, c, n), c, n)
                g = gmpy2.gcd(abs(xs-ys), n)
            if(g != n):
                utils.event.set()
                return g, n//g
        t += 1
    return n, 1

def _multiprocess_rho_brent_factorization(n, func, tries):
    t = 0
    while(t > tries):
        y, c, m = random.randrange(1, n), random.randrange(1, n), random.randrange(1, n)
        g, r, q = 1, 1, 1
        while(g == 1):
            x = y
            for _ in range(r):
                if(utils.event.is_set()):
                    return n, 1
                y = func(y, c, n)
            k = 0
            while(k < r and g == 1):
                ys = y
                for _ in range(min(m, r - k)):
                    if(utils.event.is_set()):
                        return n, 1
                    y = func(y, c, n)
                    q = q*abs(x - y)%n
                g = gmpy2.gcd(q, n)
                k = k + m
            r *= 2
        if(g == n):
            while(True):
                if(utils.event.is_set()):
                    return n, 1
                ys = func(ys, c, n)
                g = gmpy2.gcd(abs(x - ys), n)
                if(g > 1):
                    break
        if(g != n):
            utils.event.set()
            return g, n//g
    return n, 1

def _multiprocess_p_minus_1_factorization(n, B1, B2, generator):
    a_m = gmpy2.mpz(random.randrange(2, n))
    coprime = gmpy2.gcd(a_m,n)
    if(coprime != 1):
        return coprime, n//coprime
    for m in generator(start=1, stop=B1):
        if(utils.event.is_set()):
            return n, 1
        a_m = pow(a_m, m, n)
        factor = gmpy2.gcd((a_m-1)%n, n)
        if(factor != 1 and factor != n):
            utils.event.set()
            return factor, n//factor
    for m in primes.prime_generator(start=B1+1, stop=B2):
        if(utils.event.is_set()):
            return n, 1
        Q = pow(a_m, m, n) - 1
        factor = gmpy2.gcd(Q%n, n)
        if(factor != 1 and factor != n):
            utils.event.set()
            return factor, n//factor
    return n, 1