import primes, utils
from squares import Squares
from benchmark import timeit

@timeit
def factorize(*args):
    return Squares.dixon_factorization(*args)

if(__name__ == "__main__"):
    n = primes.generate_semiprime(100)
    print(n)
    print(factorize(n, utils.prime_list[:2000]))