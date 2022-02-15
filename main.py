import primes, utils, gmpy2
from squares import sqrt_convergent_generator

if(__name__ == "__main__"):
    total = 40
    n = 13290059
    print(n)
    check = [10, 23, 26, 31, 40]
    for i,values in enumerate(sqrt_convergent_generator(n)):
        if(i in check):
            print(values[0], values[-1])
        if(i >= total):
            break