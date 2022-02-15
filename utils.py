import functools, operator, multiprocessing, os, gmpy2
from typing import Dict, List, Tuple, Callable

file = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"primes.txt"), "r")
prime_list = file.read()
file.close()
prime_list = [int(n) for n in prime_list.split("\n")]

def introot(n:int, k:int=2) -> Tuple[int,bool]:
    if n < 0: return None if k%2 == 0 else -introot(-n, k)
    return gmpy2.iroot(n, k)

@functools.lru_cache(maxsize=128)
def inverse_mod(x:int ,m:int) -> int:
    return gmpy2.invert(x, m)

def log(x:int, b:int=None) -> float:
    if(b is None):
        return gmpy2.log(x)
    return gmpy2.log(x)/gmpy2.log(b)

def is_power(n:int, k:int=None) -> Tuple[int,int]:
    if(k is None):
        k = int(gmpy2.floor(gmpy2.log2(n)))
    p = min(int(gmpy2.floor(gmpy2.log2(n))), k)
    while(p > 1):
        root, check = introot(n, p)
        if(check):
            return root, p
        p -= 1
    return n, 1

def is_smooth(n:int, G:int):
    g = gmpy2.gcd(n, G)
    while(g > 1):
        while(n%g == 0):
            n //= g
        if(n == 1):
            return True
        g = gmpy2.gcd(n, G)
    return False

def are_coprime(n:List[int]) -> bool:
    for i,p in enumerate(n):
        for q in n[(i+1):]:
            if(gmpy2.gcd(p,q) != 1):
                return False
    return True

def n_from_factorization(factorization:Dict[int,int]) -> int:
    if(len(factorization) == 0):
        return 1
    return functools.reduce(lambda x,y: x*y, [pow(a,b) for a,b in factorization.items()])

def merge_factors(a:Dict[int,int], b:Dict[int,int]) -> Dict[int,int]:
    for k,v in b.items():
        value = a.get(k, 0) + v
        a[k] = value
    return a

def quadratic_residues(m:int) -> Dict[int,List[int]]:
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

def chinese_remainder(congr:List[Tuple[int,int]], a_min:int, a_max:int, M:int) -> Tuple[int,int,int]:
    residue = 0
    for r, m in congr:
        c = M//m
        d = inverse_mod(c,m)
        residue += r*c*d
    residue = residue%M
    start, stop = int(gmpy2.ceil((a_min-residue)/M)), int(gmpy2.floor((a_max - residue)/M))
    return residue, start, stop
        
def order(n:int, r:int) -> int:
    k = 1
    while(True):
        if(pow(n,k,r) == 1):
            return k
        k += 1

def pi(n:int) -> int:
    if(n > prime_list[-1]):
        return int(gmpy2.floor(n/log(n)))
    return sum([1 for p in prime_list if p <= n])

def phi(f:Dict[int,int]) -> int:
    return functools.reduce(operator.mul, [(p-1)*pow(p,k-1) for p,k in f.items()])

def product(X):
  if len(X) == 0: return 1
  while len(X) > 1:
    X = [X[i*2]*X[2*i+1] if(2*i+1 < len(X)) else X[i*2] for i in range((len(X)+1)//2)]
  return X[0]

def productTree(X):
  result = [X]
  while len(X) > 1:
    X = [X[i*2]*X[2*i+1] if(2*i+1 < len(X)) else X[i*2] for i in range((len(X)+1)//2)]
    result.append(X)
  return result

def remaindersTree(T,n):
  result = [n]
  for t in reversed(T):
    result = [result[i//2]%t[i] for i in range(len(t))]
  return result

def remainders(X, n):
  return remaindersTree(productTree(X), n)

def are_smooth(X, n):
  rems = remainders(X, n)
  e = pow(2, int(gmpy2.ceil(gmpy2.log2(gmpy2.log2(max(X))))))
  return [x//gmpy2.gcd(pow(r, e, x), x) for x,r in zip(X,rems)]

def init(l):
    global event
    event = l

def process(subprocess, args, kwargs):
    return subprocess(*args, **kwargs)

def parallelize(subprocesses:Tuple[Callable,Tuple,Dict], n_processes, initializer, initargs):
    with multiprocessing.Pool(n_processes, initializer=initializer, initargs=initargs) as pool:
        results = pool.starmap(process, subprocesses)
    return results
