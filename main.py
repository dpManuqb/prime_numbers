import primes, utils, gmpy2
from squares import Squares

if(__name__ == "__main__"):
    n = primes.generate_semiprime(128)
    print(n)
    print(Squares.cfrac_factorization(n, utils.prime_list[:1200], 5000))