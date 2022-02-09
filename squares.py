import utils, multiprocessing, gmpy2, numba, functools, operator, logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

############################################################################
################################ Main class ################################
############################################################################

class Squares:

    @staticmethod
    def kraitchik_factorization(n:int, processes:int=1):
        """Range searching of square congruence"""
        start, stop = utils.introot(n)[0]+1, n-2
        if(processes == 1):
            return _kraitchik(n, start, stop)

        else:
            block_size = int(gmpy2.ceil((stop-start+1)/processes))
            start, stop, _stop = list(range(start, processes*block_size, block_size)), list(range(start+block_size-1, (processes+1)*block_size, block_size)), stop
            stop[-1] = _stop

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_kraitchik, zip(processes*[n], start, stop))

            results = list(filter(lambda x: x[0] not in [1,n], results))

            if(results == []):
                return n, 1
            else:
                return results[0]

    @staticmethod
    def dixon_factorization(n:int, F:List[int]):
        """Dixon factorization method with list of primes F + -1"""

        logger.info("Searching smooth squares by brute force...")
        x, start, stop = gmpy2.isqrt(n), 1, 2*(n-2)
        G = functools.reduce(operator.mul, F)
        congruences, total, j = [], int((len(F)+1)*1.001), 0
        with tqdm(total=total) as pbar:
            for i in range(start, stop+1):
                step = pow(-1,i+1)*i
                x = x + step
                y = pow(x, 2, n)
                if(step < 0):
                    y -= n
                if(utils.is_smooth(abs(y), G)):
                    j += 1
                    congruences.append({"x": x, "y": y,})
                    pbar.update()
                if(j == total):
                    break

        logger.info("Solving matrix...")
        for i, c in enumerate(congruences):
            congruences[i]["factors"] = _list_factorization(c["y"], F)
            congruences[i]["vector"] = _factorization_to_sparse_vector(congruences[i]["factors"], [-1]+F)
        M = _create_matrix(congruences, [-1]+F)
        M = _gauss_jordan_gf2(M)
        M = M[~np.all(M == 0, axis=1)]
        M = _nullspace_basis(M)

        logger.info("Searching valid factorization...")
        for solution in _solution_generator(M):
            g = _factor_from_solution(solution, congruences, n)
            if(g not in [1,n]):
                return g, n//g

        return n, 1

############################################################################
########################## Single process versions #########################
############################################################################

def _kraitchik(n, start, stop):
    for x in range(start,stop+1):
        y_2 = pow(x,2,n)
        y, check = utils.introot(y_2, 2)
        if(check):
            g = gmpy2.gcd(abs(x-y), n)
            if(1 < g < n):
                return g, n//g
    return n,1

############################################################################
######################### Multiprocessing versions #########################
############################################################################

def _multiprocess_kraitchik(n, start, stop):
    for x in range(start,stop+1):
        if(utils.event.is_set()):
            return n,1
        y_2 = pow(x,2,n)
        y, check = utils.introot(y_2, 2)
        if(check):
            g = gmpy2.gcd(abs(x-y), n)
            if(1 < g < n):
                utils.event.set()
                return g, n//g
    return n,1

############################################################################
################################## Support #################################
############################################################################

def _list_factorization(n:int, F:List[int]):
    f = {}
    if(n < 0):
        f[-1] = 1
        n *= -1
    for p in F:
        exp = 0
        while(n%p == 0):
            n //= p
            exp += 1
        if(exp > 0):
            f[p] = exp
        if(n == 1):
            break
    return f

def _factorization_to_sparse_vector(f:Dict[int,int], F:List[int]):
    vector = []
    for p, e in f.items():
        if(e%2 == 1):
            vector.append(F.index(p))
    return vector

def _create_matrix(congruences, F:List[int]):
    M = []
    for c in congruences:
        vector = [np.uint8(0)]*len(F)
        for i in c["vector"]:
            vector[i] = np.uint8(1)
        M.append(vector)
    return np.array(M).T

@numba.jit(nopython=True)
def _gauss_jordan_gf2(M):
    rows, cols = M.shape
    numpivots = 0
    for j in range(cols):
        if(numpivots >= rows):
            break
        pivotrow = numpivots
        while(pivotrow < rows and M[pivotrow, j] == 0):
            pivotrow += 1
        if(pivotrow == rows):
            continue
        row_aux = np.copy(M[numpivots])
        M[numpivots] = M[pivotrow]
        M[pivotrow] = row_aux
        pivotrow = numpivots
        numpivots += 1
        for i in range(pivotrow+1, rows):
            M[i] = (M[i] + M[i,j]*M[pivotrow])%2
    for i in range(numpivots-1, -1, -1):
        pivotcol = 0
        while(pivotcol < cols and M[i, pivotcol] == 0):
            pivotcol += 1
        if(pivotcol == cols):
            continue
        for j in range(i):
            M[j] = (M[j] + M[i]*M[j,pivotcol])%2
    return M
_gauss_jordan_gf2(np.array([[0]])) #Compile numba

def _nullspace_basis(M):
    rows, cols = M.shape
    ones, basis = [], []
    for i in range(rows):
        ones.append(list(np.argwhere(M[i] == 1).reshape(1,-1)[0]))
    ones = np.array(ones, dtype="object")
    for i,row in enumerate(ones):
        free_vars = row[1:]
        if(len(free_vars) > 0):
            for v in free_vars:
                vector = [0]*cols
                vector[v] = 1
                for j in range(i, rows):
                    if(v in ones[j]):
                        vector[ones[j][0]] = 1
                        ones[j].remove(v)
                basis.append(vector)
    all_zeros = np.argwhere(np.all(M == 0, axis=0)).reshape(1, -1)[0]
    for i in all_zeros:
        vector = [0]*cols
        vector[i] = 1
        basis.append(vector)
    return np.array(basis)

def _solution_generator(nullspace):
    length = len(nullspace)
    for i in range(2**length):
        solution = np.array([0]*len(nullspace[0]))
        for i,bit in enumerate(("{:0"+str(length)+"b}").format(i)):
            if(bit == "1"):
                solution += nullspace[i]
        yield solution%2

def _factor_from_solution(solution, congruences, n):
    x, y_2 = 1, 1
    for i,bit in enumerate(solution):
        if(bit == 1):
            x *= congruences[i]["x"]
            y_2 *= congruences[i]["y"]
    y = gmpy2.isqrt(y_2)
    return gmpy2.gcd(abs(x-y), n)