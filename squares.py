import utils, multiprocessing, gmpy2, numba, functools, operator, itertools, logging
import numpy as np
import tqdm
from typing import List

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

############################################################################
################################ Main class ################################
############################################################################

class Squares:

    @staticmethod
    def fermat_factorization(n:int, processes:int=1, sieve:bool=False, F:List[int]=[7,9,11,13,16]):
        """Fermat factorization assuming n is odd and not a perfect square"""
        a_min, a_max = utils.introot(n)[0]+1, (n+9)//6

        if(sieve):
            congruences, M = _fermat_build_congruences(n, F, a_min, a_max)
            congruences = list(sorted(congruences, key=lambda c: c[0]))
        if(processes == 1):
            if(not sieve):
                generator = range(a_min, a_max)
            else:
                generator = _fermat_congruences_generator(congruences, M)

            return _fermat_factorization(n, generator)

        else:
            if(not sieve):
                block_size = int(gmpy2.ceil((a_max-a_min+1)/processes))
                a_min, a_max, _a_max = list(range(a_min, processes*block_size, block_size)), list(range(a_min+block_size-1, (processes+1)*block_size, block_size)), a_max
                a_max[-1] = _a_max
                generators = [range(start,stop+1) for start,stop in zip(a_min,a_max)]
                M = None
            else:
                block_size = int(gmpy2.ceil(len(congruences)/processes))
                generators = [congruences[i*block_size:(i+1)*block_size] for i in range(processes)]

            with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
                results = pool.starmap(_multiprocess_fermat_factorization, zip(processes*[n], generators, processes*[M]))

            results = list(filter(lambda x: x[0] not in [1,n], results))

            if(results == []):
                return n, 1
            else:
                return results[0]

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
    def dixon_factorization(n:int, F:List[int], block_size:int=10000, partials:bool=False):
        """Dixon factorization method with list of primes F + -1"""

        logger.info("Searching smooth squares by brute force...")
        if(block_size == 1):
            congruences = _dixon_search_smooths(n, F)
        else:
            congruences = _dixon_search_smooth_blocks(n, F, block_size, partials)

        logger.info("Solving matrix...")
        for i, c in enumerate(congruences):
            congruences[i]["factors"] = _dixon_list_factorization(c["y"]//c["r"], F)
            congruences[i]["vector"] = _dixon_factorization_to_sparse_vector(congruences[i]["factors"], [-1]+F)
        M = _dixon_create_matrix(congruences, [-1]+F)
        M = _dixon_gauss_jordan_gf2(M)
        M = M[~np.all(M == 0, axis=1)]
        M = _dixon_nullspace_basis(M)

        logger.info("Searching valid factorization...")
        t = 1
        for solution in _dixon_solution_generator(M):
            g = _dixon_factor_from_solution(solution, congruences, n)
            if(g not in [1,n]):
                logger.info(f"Tested solutions: {t}")
                return g, n//g
            t += 1
        return n, 1

    @staticmethod
    def cfrac_factorization(n:int, F:List[int], block_size:int=10000, partials:bool=False):
        """CFRAC method with list of primes F + -1"""

        logger.info("Searching smooth squares in sqrt convergents...")
        if(block_size == 1):
            congruences = _cfrac_search_smooths(n, F)
        else:
            congruences = _cfrac_search_smooth_blocks(n, F, block_size, partials)

        logger.info("Solving matrix...")
        for i, c in enumerate(congruences):
            congruences[i]["factors"] = _dixon_list_factorization(c["y"]//c["r"], F)
            congruences[i]["vector"] = _dixon_factorization_to_sparse_vector(congruences[i]["factors"], [-1]+F)
        M = _dixon_create_matrix(congruences, [-1]+F)
        M = _dixon_gauss_jordan_gf2(M)
        M = M[~np.all(M == 0, axis=1)]
        M = _dixon_nullspace_basis(M)

        logger.info("Searching valid factorization...")
        t = 1
        for solution in _dixon_solution_generator(M):
            g = _dixon_factor_from_solution(solution, congruences, n)
            if(g not in [1,n]):
                logger.info(f"Tested solutions: {t}")
                return g, n//g
            t += 1
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

def _multiprocess_fermat_factorization(n, generator, M):
    if(type(generator) != range):
        generator = _fermat_congruences_generator(generator, M) # Cannot pickle generator so its created here
    for a in generator:
        if(utils.event.is_set()):
            return n, 1
        b2 = a*a - n
        b, check = utils.introot(b2)
        if(check):
            utils.event.set()
            x, y = abs(a+b), abs(a-b)
            return x, y
    return n, 1

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

def _fermat_factorization(n, generator):
    for a in generator:
        b2 = a*a - n
        b, check = utils.introot(b2)
        if(check):
            x, y = abs(a+b), abs(a-b)
            return x, y
    return n, 1

def _fermat_merge_combine_congruences(congruences, a_min, a_max):   
    congruences = [list(zip(r,len(r)*[m])) for m,r in congruences.items()]
    M = functools.reduce(lambda x,y:x*y,[m[0][1] for m in congruences])
    result = []
    for cong in itertools.product(*congruences):
        result.append(utils.chinese_remainder(cong, a_min, a_max, M))
    return result, M

def _fermat_congruences_generator(congruences, M):
    if(len(congruences) > 0):
        max_k = max([sp-st+1 for _, st, sp in congruences])
        for k in range(max_k):
            for r,s,_ in congruences:
                yield r + (k+s)*M

def _fermat_build_congruences(n, F, a_min, a_max):
    if(not utils.are_coprime(F)):
        raise Exception("F list items must be coprime")
    congruences = {}
    for m in F:
        congruences[m] = []
        residues = utils.quadratic_residues(m)
        [congruences[m].extend(residues.get(r, [])) for r in [(r+n)%m for r in residues.keys()]]
    return _fermat_merge_combine_congruences(congruences, a_min, a_max)

def _dixon_search_smooths(n, F):
    G = utils.product(F)
    x, start, stop = gmpy2.isqrt(n), 1, 2*(n-2)
    congruences, total, j = [], int((len(F)+1)*1.001), 0
    with tqdm.tqdm(total=total) as pbar:
        for i in range(start, stop+1):
            step = pow(-1,i+1)*i
            x = x + step
            y = pow(x, 2, n)
            if(step < 0):
                y -= n
            if(utils.is_smooth(abs(y), G)):
                j, _, _= j+1, congruences.append({"x": x, "y": y, "r": 1}), pbar.update()
            if(j == total):
                break
    return congruences

def _dixon_search_smooth_blocks(n,  F, length, partials):
    def _build_partial_squares(new_partial_rels, old_partial_rels):
        squares = []
        for x_, y_, r_ in new_partial_rels:
            old_rel = old_partial_rels.get(r_)
            if(old_rel):
                squares.append((x_*old_rel["x"], y_*old_rel["y"], r_))
                old_partial_rels.pop(r_)
            else:
                old_partial_rels[r_] = {"x":x_, "y":y_}
        return old_partial_rels, squares

    G = utils.product(F)
    middle, start, stop = gmpy2.isqrt(n), 1, n-2
    congruences, partial_relations, total, j, length = [], {}, int((len(F)+1)*1.001), 0, length//2
    with tqdm.tqdm(total=total) as pbar:
        for i in range(start, stop, length):
            # Forward
            x_candidates = list(range(middle+i,middle+i+length))
            y_candidates = [pow(x_i, 2, n) for x_i in x_candidates]
            relations = utils.are_smooth(y_candidates, G)
            full_relations = list(filter(lambda x: (x[2] == 1), list(zip(x_candidates, y_candidates, relations))))
            rel_found = len(full_relations)
            j,_,_ = j+rel_found, pbar.update(rel_found), congruences.extend(full_relations)
            if(j >= total):
                break

            if(partials):
                partial_relations, partial_squares = _build_partial_squares(list(filter(lambda x: (x[2] != 1) and x[2] < max(F)**2, list(zip(x_candidates, y_candidates, relations)))), partial_relations)
                partial_squares.extend([(x_,y_,gmpy2.isqrt(r_2)) for x_, y_, r_2 in list(filter(lambda x: (x[2] != 1) and gmpy2.is_square(x[2]), list(zip(x_candidates, y_candidates, relations))))])
                rel_found = len(full_relations)
                j,_,_ = j+rel_found, pbar.update(rel_found), congruences.extend(partial_squares)
                if(j >= total):
                    break

            # Backward
            x_candidates = list(range(middle-i-length+1,middle-i+1))
            y_candidates = [abs(pow(x_i, 2, n)-n) for x_i in x_candidates]
            relations = utils.are_smooth(y_candidates, G)
            full_relations = list(filter(lambda x: (x[2] == 1), list(zip(x_candidates, [-y for y in y_candidates], relations))))
            rel_found = len(full_relations)
            j,_,_ = j+rel_found, pbar.update(rel_found), congruences.extend(full_relations)
            if(j >= total):
                break

            if(partials):
                partial_relations, partial_squares = _build_partial_squares(list(filter(lambda x: (x[2] != 1) and x[2] < max(F)**2, list(zip(x_candidates, [-y for y in y_candidates], relations)))),partial_relations)
                partial_squares.extend([(x_,y_,gmpy2.isqrt(r_2)) for x_, y_, r_2 in list(filter(lambda x: (x[2] != 1) and gmpy2.is_square(x[2]), list(zip(x_candidates, [-y for y in y_candidates], relations))))])
                rel_found = len(full_relations)
                j,_,_ = j+rel_found, pbar.update(rel_found), congruences.extend(partial_squares)
                if(j >= total):
                    break

    return [{"x": x, "y": y, "r": r} for x,y,r in congruences]

def _dixon_list_factorization(n, F):
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

def _dixon_factorization_to_sparse_vector(f, F):
    vector = []
    for p, e in f.items():
        if(e%2 == 1):
            vector.append(F.index(p))
    return vector

def _dixon_create_matrix(congruences, F):
    M = []
    for c in congruences:
        vector = [np.uint8(0)]*len(F)
        for i in c["vector"]:
            vector[i] = np.uint8(1)
        M.append(vector)
    return np.array(M).T

@numba.jit(nopython=True)
def _dixon_gauss_jordan_gf2(M):
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
_dixon_gauss_jordan_gf2(np.array([[0]])) #Compile numba

def _dixon_nullspace_basis(M):
    rows, cols = M.shape
    ones, basis = [], []
    for i in range(rows):
        ones.append(list(np.argwhere(M[i] == 1).reshape(1,-1)[0]))
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

def _dixon_solution_generator(nullspace):
    length = len(nullspace)
    for i in range(1, length+1):
        for solution in itertools.combinations(nullspace, i):
            yield sum(solution)%2

def _dixon_factor_from_solution(solution, congruences, n):
    x, y, r = 1, 1, 1
    for i,bit in enumerate(solution):
        if(bit == 1):
            x, y, r = x*congruences[i]["x"], y*congruences[i]["y"], r*congruences[i]["r"]
    y = r * gmpy2.isqrt(y)
    return gmpy2.gcd(abs(x-y), n)

def _sqrt_convergent_generator(n:int, k:int=1):
    iroot = gmpy2.isqrt(n*k)

    # i = 0
    A_im1, A_i, B_im1, B_i, q_i = 1, iroot, 0, 1, iroot
    P_i, Q_i = 0, 1
    yield A_im1%n, A_i, B_i, q_i, Q_i, 1

    # i = 1
    i = 1
    P_i, Q_i, Q_im1 = iroot, n*k-iroot**2, 1
    q_i = int(gmpy2.floor((iroot+P_i)/Q_i))
    A_i, A_im1, B_i, B_im1 = (q_i*A_i+A_im1)%n, A_i, (q_i*B_i+B_im1)%n, B_i
    yield A_im1, A_i, B_i, q_i, Q_i, -1

    while(True):
        # i > 1
        i += 1
        P_i_ = q_i*Q_i-P_i
        P_i, Q_i, Q_im1 = P_i_, Q_im1+q_i*(P_i-P_i_), Q_i
        q_i = int(gmpy2.floor((iroot+P_i)/Q_i))
        A_i, A_im1, B_i, B_im1 = (q_i*A_i+A_im1)%n, A_i, (q_i*B_i+B_im1)%n, B_i
        yield A_im1, A_i, B_i, q_i, Q_i, pow(-1, i)
        if(q_i == 2*iroot):
            break

def _sqrt_convergent_block_generator(n, k, block_size):
    generator = _sqrt_convergent_generator(n, k)
    while(True):
        block = []
        try:
            while(len(block) < block_size):
                block.append(next(generator))
            yield block
        except StopIteration:
            yield block
            break

def _cfrac_search_smooths(n, F):
    congruences, total, i, j = [], int((len(F)+1)*1.001), 1, 0
    with tqdm.tqdm(total=total) as pbar:
        while(j < total):
            F_ = [2]+list(filter(lambda p: gmpy2.legendre(n*i,p) == 1, F[1:]))
            G = utils.product(F_)
            for value in _sqrt_convergent_generator(n, i):
                if(utils.is_smooth(value[-2], G)):
                    j, _, _= j+1, congruences.append({"x": value[0], "y": value[-2]*value[-1], "r": 1}), pbar.update()
                if(j >= total):
                    break
            i += 1
    return congruences

def _cfrac_search_smooth_blocks(n,  F, length, partials):
    def _build_partial_squares(new_partial_rels, old_partial_rels):
        squares = []
        for x_, y_, r_ in new_partial_rels:
            old_rel = old_partial_rels.get(r_)
            if(old_rel):
                squares.append((x_*old_rel["x"], y_*old_rel["y"], r_))
                old_partial_rels.pop(r_)
            else:
                old_partial_rels[r_] = {"x":x_, "y":y_}
        return old_partial_rels, squares


    congruences, partial_relations, total, i, j = [], {}, int((len(F)+1)*1.001), 1, 0
    with tqdm.tqdm(total=total) as pbar:
        while(j < total):
            F_ = [2]+list(filter(lambda p: gmpy2.legendre(n*i,p) == 1, F[1:]))
            G = utils.product(F_)
            for block in _sqrt_convergent_block_generator(n, i, length):
                relations = utils.are_smooth([y for _,_,_,_,y,_ in block], G)
                full_relations = [(rel[0][0], rel[0][-2]*rel[0][-1], 1) for rel in list(filter(lambda x: (x[1] == 1), list(zip(block, relations))))]
                rel_found = len(full_relations)
                j,_,_ = j+rel_found, pbar.update(rel_found), congruences.extend(full_relations)
                if(j >= total):
                    break

                if(partials):
                    partial_relations, partial_squares = _build_partial_squares([(values[0], values[-2]*values[-1], r) for values, r in list(filter(lambda x: (x[1] != 1) and x[1] < max(F)**2, list(zip(block, relations))))], partial_relations)
                    partial_squares.extend([(values[0],values[-1]*values[-2],gmpy2.isqrt(r)) for values,r in list(filter(lambda x: (x[1] != 1) and gmpy2.is_square(x[1]), list(zip(block, relations))))])
                    rel_found = len(full_relations)
                    j,_,_ = j+rel_found, pbar.update(rel_found), congruences.extend(partial_squares)
                    if(j >= total):
                        break
            i += 1

    return [{"x": x, "y": y, "r": r} for x,y,r in congruences]