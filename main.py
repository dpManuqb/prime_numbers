import random
from fermat import mod_generator
from primes import Primes

if(__name__ == "__main__"):
    b = 80
    n = random.randrange(2**(b-2), 2**(b-1))*2+1
    print(n)
    print(Primes.fermat_with_trial_factorization(n, sieve=[9,13,16,25]))