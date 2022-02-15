import utils, gmpy2, random
from typing import Callable
from bpsw import BPSW

def generate_composite(b:int) -> int:
    return random.randrange(2**((b>>1)-1), 2**(b>>1))*random.randrange(2**((b>>1)-1), 2**(b>>1))

def generate_prime(b:int, test:Callable=BPSW.strong_test, **kwargs) -> int:
    while(True):
        n = random.randrange(2**(b-2), 2**(b-1))*2+1
        if(test(n, **kwargs)):
            return n

def generate_semiprime(b:int, diff:int=0, test:Callable=BPSW.strong_test, **kwargs) -> int:
    return generate_prime((b+diff)//2, test, **kwargs)*generate_prime((b-diff)//2, test, **kwargs)

def generate_strong_prime(b:int, test:Callable=BPSW.strong_test, **kwargs) -> int:
    s, t = generate_prime((b>>1), test, **kwargs), generate_prime((b>>1), test, **kwargs)
    i = random.randrange(0, 100)
    r = 2*i*t+1
    while(not test(r, **kwargs)):
        i += 1
        r = 2*i*t+1
    p_0 = 2*pow(s, r-2, r)*s - 1
    j = random.randrange(0, 100)
    p = p_0 + 2*j*r*s
    while(not test(p, **kwargs)):
        j += 1
        p = p_0 + 2*j*r*s
    return p

def generate_strong_semiprime(b:int, test:Callable=BPSW.strong_test, **kwargs) -> int:
    return generate_strong_prime(b>>1, test, **kwargs)*generate_strong_prime(b>>1, test, **kwargs)

def generate_coprime(n:int, b:int) -> int:
    while(True):
        guess = random.randrange(2**(b-1), 2**b)
        if(gmpy2.gcd(n, guess) == 1):
            return guess

def next_prime(n:int, test:Callable=BPSW.strong_test, **kwargs):
  if(not n & 1):
    n -= 1
  n += 2
  while(not test(n, **kwargs)):
    n += 2
  return n

def range_generator(start:int, stop:int):
    for i in range(start, stop+1):
        yield i

def prime_generator(start:int, stop:int, test:Callable=BPSW.strong_test, **kwargs):
    if(start <= utils.prime_list[-1]):
        i, p = 0, 2
        while(p < start):
            i += 1
            p = utils.prime_list[i]
        while(p < stop and i < len(utils.prime_list)):
            p = utils.prime_list[i]
            yield p
            i += 1
    if(utils.prime_list[-1] < stop):
        p = next_prime(max(start, utils.prime_list[-1]), test, **kwargs)
        while(p <= stop):
            yield p
            p = next_prime(p, test, **kwargs)

def prime_power_generator(start:int, stop:int, test:Callable=BPSW.strong_test, **kwargs):
    log_B = utils.log(stop)
    if(start <= utils.prime_list[-1]):
        i, p = 0, 2
        while(p < start):
            i += 1
            p = utils.prime_list[i]
        while(p < stop and i < len(utils.prime_list)):
            p = utils.prime_list[i]
            yield pow(p, int(gmpy2.floor(log_B/utils.log(p))))
            i += 1
    if(utils.prime_list[-1] < stop):
        p = next_prime(max(start, utils.prime_list[-1]), test, **kwargs)
        while(p <= stop):
            yield pow(p, int(gmpy2.floor(log_B/utils.log(p))))
            p = next_prime(p, test, **kwargs)