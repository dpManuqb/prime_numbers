import tqdm

import utils, logging
from trial import Trial
from fermat import Fermat
from millerrabin import MillerRabin
from lucas import Lucas
from bpsw import BPSW
from aks import AKS
from pollard import Pollard

from typing import Dict, Tuple, Union, List

logging.basicConfig(level=logging.DEBUG)

class Primes:
    Trial = Trial
    Fermat = Fermat
    MillerRabin = MillerRabin
    Lucas = Lucas
    BPSW = BPSW
    AKS = AKS
    Pollard = Pollard

    @staticmethod
    def fermat_with_trial_factorization(n:int, B:int=None, sieve:Union[None,List[int]]=None) -> Tuple[Dict[int,int],Tuple[int,int]]:
        if(B is None):
            B = utils.primeList[-1]
        
        if(BPSW.test(n)):
            return {n:1}, (1,1)

        f, r = Trial.factorization(n, B)

        if(r != 1 and BPSW.test(r)):
            f[r], r = 1, (1, 1)
        elif(r != 1):
            if(sieve is None):
                f1, f2 = Fermat.factorization(n, start=(n+B**2)//(2*B))
            else:
                f1, f2 = Fermat.sieve_factorization(n, modulus=sieve, start=(n+B**2)//(2*B))
            if(f1 != 1 and BPSW.test(f1)):
                if(f1 == f2):
                    f[f1], f1, f2 = 2, 1, 1
                else:
                    f[f1], f1 = 1, 1
            elif(f2 != 1 and f1 != f2 and BPSW.test(f2)):
                f[f2], f2 = 1, 1
            r = (f1, f2)
        else:
            r = (1, 1)

        return f, r



def Square_factorization(n, limit=10**7):
    root = utils.i2root(n)
    residues = [r*r for r in range(min([root+1, limit]))]
    with tqdm.tqdm(total=n//2-root) as pbar:
        for a in range(root+1,min([limit, n//2])):
            r = (a*a)%n
            if(r in residues):
                b = residues.index(r)
                return utils.gcd(abs(a+b), n), utils.gcd(abs(a-b), n)
            else:
                residues.append(r)
            pbar.update(1)
        for a in range(limit+1,n//2):
            r = (a*a)%n
            if(r in residues):
                b = residues.index(r)
                return utils.gcd(abs(a+b), n), utils.gcd(abs(a-b), n)
            pbar.update(1)

""" def Dixon_factorization(n, B=50):
    residues = []
    root = i2root(n)
    total = 0
    needed = pi(B)
    for a in range(root+1, n//2):
        f,r = Trial_Division_factorization((a*a)%n, B, verb=False)
        if(r != 1 and BPSWTest(r)):
            f[r], r = 1, 1
        if(r == 1 and isBSmooth(f,B)):
            if(all([k%2 == 0 for k in f.values()])):
                b = {k:v//2 for k,v in f.items()}
                return gcd(a+b,n), gcd(abs(a-b),n)
            residues.append((a,f))
            total += 1
        if(total == needed):
            break
    return residues """