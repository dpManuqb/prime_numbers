import primes, utils, gmpy2
import benchmark
from squares import Squares

@benchmark.timeit
def cfrac(*args):
    return Squares.cfrac_factorization(*args)

if(__name__ == "__main__"):
    n = primes.generate_semiprime(128)
    print(n)
    print(cfrac(n, utils.prime_list[:2000], 2000))