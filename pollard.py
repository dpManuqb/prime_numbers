from typing import Callable, Tuple
import utils
import random
import math

class Pollard:

    @staticmethod
    def rho_factorization(n:int, random_function:Callable=lambda x,c,m: (x*x+c)%m, check:int=100) -> Tuple[int,int]:
        while(True):
            a, s = random.randrange(1, n-2), random.randrange(0, n)
            u, v, g, total, product = s, s, 1, 0, 1
            while(g == 1):
                u = random_function(u, a, n)
                v = random_function(random_function(v, a, n), a, n)
                product, total = (product*abs(u-v))%n, total + 1
                if(total == check):
                    g, product, total = utils.gcd(product, n), 1, 0
            if(g != n):
                return g, n//g
            elif(check > 1):
                g = 1
                while(g == 1):
                    u = random_function(u, a, n)
                    v = random_function(random_function(v, a, n), a, n)
                    g = utils.gcd(abs(u-v), n)
                if(g != n):
                    return g, n//g

    @staticmethod
    def p_1_factorization(n:int, B:int=1000000, fails:int=10) -> Tuple[int,int]:
    
        f = 0
        M = [p**(math.floor(math.log(B,p))) for p in utils.primeList if p <= B]
        while(f < fails):
            g = random.randrange(2,fails*10)
            factor = utils.gcd(g,n)
            if(factor != 1 and factor != n):
                return factor, n//factor
            for m in M:
                g = pow(g, m, n)
            factor = utils.gcd(g-1, n)
            if(factor != 1 and factor != n):
                return factor, n//factor
            f += 1
        return 1, n