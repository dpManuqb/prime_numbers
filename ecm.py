import gmpy2, utils, random, multiprocessing, primes
from typing import Callable

############################################################################
################################ Main class ################################
############################################################################

class ECM:
  
	@staticmethod
	def factorization(n:int, B1:int, B2:int, curves:int, generator:Callable, processes:int=1):
		"""Ellyptic Curve Method for factorization"""
		if(processes == 1):
			return _ecm_factorization(n, B1, B2, curves, generator)

		else:
			curves = int(gmpy2.ceil(curves/processes))

			with multiprocessing.Pool(processes, initializer=utils.init, initargs=(multiprocessing.Event(),)) as pool:
				result = pool.starmap(_multiprocess_ecm_factorization, zip(processes*[n], processes*[B1], processes*[B2], processes*[curves], processes*[generator]))

			result = list(filter(lambda f: f[0] not in [1,n], result))
			if(result == []):
				return n, 1
			else:
				return result[0]

############################################################################
########################## Single process versions #########################
############################################################################

def _ecm_factorization(n, B1, B2, curves, generator):
	for _ in range(curves):
		a, P = random.randrange(1,n), (gmpy2.mpz(random.randrange(1,n)), gmpy2.mpz(random.randrange(1,n)))
		try:
			for k in generator(start=1, stop=B1):
				P = ec_mult(k, P, a, n)
		except ZeroDivisionError as e:
			g = gmpy2.gcd(n, int(str(e)))
			if(1 < g < n):
				return g, n//g
			else:
				continue
		try:
			for k in primes.prime_generator(start=B1+1, stop=B2):
				ec_mult(k, P, a, n)
		except ZeroDivisionError as e:
			g = gmpy2.gcd(n, int(str(e)))
			if(1 < g < n):
				return g, n//g
			else:
				continue
	return n, 1

############################################################################
######################### Multiprocessing versions #########################
############################################################################

def _multiprocess_ecm_factorization(n, B1, B2, curves, generator):
	for _ in range(curves):
		a, P = random.randrange(1,n), (gmpy2.mpz(random.randrange(1,n)), gmpy2.mpz(random.randrange(1,n)))
		try:
			for k in generator(start=1, stop=B1):
				if(utils.event.is_set()):
					return n, 1
				P = ec_mult(k, P, a, n)
		except ZeroDivisionError as e:
			g = gmpy2.gcd(n, int(str(e)))
			if(1 < g < n):
				utils.event.set()
				return g, n//g
			else:
				continue
		try:
			for k in primes.prime_generator(start=B1+1, stop=B2):
				if(utils.event.is_set()):
					return n, 1
				ec_mult(k, P, a, n)
		except ZeroDivisionError as e:
			g = gmpy2.gcd(n, int(str(e)))
			if(1 < g < n):
				utils.event.set()
				return g, n//g
			else:
				continue
	return n, 1

############################################################################
################################## Support #################################
############################################################################

def _ec_sum(p1, p2, a, n):
	if(p1 == p2):
		try:
			m = (3*p2[0]**2 + a)*gmpy2.invert(2*p2[1], n)
		except ZeroDivisionError:
			raise ZeroDivisionError(2*p2[1])
	else:
		try:
			m = (p2[1]-p1[1])*gmpy2.invert(p2[0]-p1[0], n)
		except ZeroDivisionError:
			raise ZeroDivisionError(p2[0]-p1[0])
	x, y = (m**2-p2[0]-p1[0])%n, (m*(2*p2[0]+p1[0]-m**2)-p2[1])%n
	return x, y

def ec_mult(k, P0, a, n):
	P = P0
	for bit in bin(k)[3:]:
		if(bit == "1"):
			P = _ec_sum(P, P0, a, n)
		P = _ec_sum(P, P, a, n)
	return P