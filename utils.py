import functools, random, operator
from typing import Dict, List, Tuple

file = open("./primes.txt", "r")
primeList = file.read()
file.close()

primeList = [int(n) for n in primeList.split("\n")]
max_p = 229
primorial = functools.reduce(lambda x,y: x*y, primeList[:primeList.index(max_p)+1])

def iroot(n:int, k:int) -> int:
    x, y = (n+1)//k, n
    while x < y:
        x, y = ((k-1)*x + n//pow(x,k-1))//k, x
    return y

def i2root(n:int) -> int:
    x, y = (n+1)>>1, n
    while x < y:
        x, y = (x + n//x)>>1, x
    return y

def isPower(n:int, B:int=primeList[-1]) -> Tuple[int,int]:
    for k in primeList:
        if(k > B):
            break
        root = iroot(n, k)
        if(pow(root, k) == n):
            return root, k
    return n, 1

def gcd(a:int, b:int) -> int:
    while(b != 0):
        a,b = b, a%b
    return a

def egcd(a:int, b:int) -> Tuple[int,int,int]:
  x,y, u,v = 0,1, 1,0
  while a != 0:
      q, r = b//a, b%a
      m, n = x-u*q, y-v*q
      b,a, x,y, u,v = a,r, u,v, m,n
  return b, x, y

@functools.lru_cache(maxsize=None)
def modinv(a:int, m:int) -> int:
    g, x, _ = egcd(a, m)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return x % m

def jacobi(a:int, n:int) -> int:
    if n <= 0:
        return jacobi(-1*a, n)
    if n % 2 == 0:
        raise Exception("'n' must be odd.")
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            n_mod_8 = n % 8
            if n_mod_8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    if n == 1:
        return result
    else:
        return 0

def isSmooth(f:Dict[int,int], B:int) -> bool:
    return all([k <= B for k in f.keys()])

def areCoprime(n:List[int]) -> bool:
    for i,p in enumerate(n):
        for q in n[(i+1):]:
            if(gcd(p,q) != 1):
                return False
    return True

def pi(n:int) -> int:
    if(n > primeList[-1]):
        raise Exception("Argument too long for the actual prime list")
    return sum([1 for p in primeList if p <= n])

def phi(f:Dict[int,int]) -> int:
    return functools.reduce(lambda x,y: x*y, [(p-1)*pow(p,k-1) for p,k in f.items()])

def sqr_mod_res(m:int) -> Dict[int,List[int]]:

    squares = {}
    for i in range(0,m):
        squares[i] = (i*i)%m

    residues = set([v for v in squares.values()])

    result = {}
    for r in residues:
        result[r] = []
        for k,v in squares.items():
            if(v == r):
                result[r].append(k)

    return result

def order(n, r):
    k = 1
    while(True):
        if(pow(n,k,r) == 1):
            return k
        k += 1
        
def initialTest(n:int):
    if(n == 2):
        return True
    elif(not n & 1):
        return False
    elif(n > max_p and gcd(n, primorial) != 1):
        return False
    else:
        return None

def ncr(n:int, r:int) -> int:
    r = min(r, n-r)
    numer = functools.reduce(operator.mul, range(n, n-r, -1), 1)
    denom = functools.reduce(operator.mul, range(1, r+1), 1)
    return numer // denom

def gen_prime(b, test, **kwargs):
    print("<", end="")
    while(True):
        print("*", end="")
        guess = random.randrange(2**(b-2), 2**(b-1))*2+1
        if(test(guess, **kwargs)):
            print(">")
            return guess

def gen_strong_prime(b, test, **kwargs):
    s, t = gen_prime(b//2, test, **kwargs), gen_prime(b//2, test, **kwargs)
    i = random.randrange(0, 100)
    r = 2*i*t+1
    print("<", end="")
    while(not test(r, **kwargs)):
        i += 1
        r = 2*i*t+1
        print("*", end="")
    p_0 = 2*pow(s, r-2, r)*s - 1
    j = random.randrange(0, 100)
    p = p_0 + 2*j*r*s
    while(not test(r, **kwargs)):
        j += 1
        p = p_0 + 2*j*r*s
        print("*", end="")
    print(">")
    return p

def gen_coprime(n, B):
    while(True):
        guess = random.randrange(2,B)
        if(gcd(n, guess) == 1):
            return guess

def n_from_factorization(factorization):
    return functools.reduce(lambda x,y: x*y, [pow(a,b) for a,b in factorization.items()])

def merge_factors(a,b):
    for k,v in b.items():
        value = a.get(k, 0) + v
        a[k] = value
    return a

def random_range(x_min:int, x_max:int):
    while(True):
        yield random.randrange(x_min, x_max)