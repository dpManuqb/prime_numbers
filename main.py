import random
from primes import Primes

if(__name__ == "__main__"):
    b = 80
    n = random.randrange(2**(b-2), 2**(b-1))*2+1
    print(n)
    print(Primes.Trial.multiprocess_factorization(n, processes=4))