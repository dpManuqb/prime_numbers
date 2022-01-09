from typing import List

class Poly:
    def __init__(self, coefs:List[int]):
        self.coefs = Poly.normalize(coefs)

    @staticmethod
    def normalize(p):
        i = len(p) - 1
        while(p[i] == 0 and i > 0):
            i -= 1
        return p[:i+1]

    def __mul__(self, other):
        result = [0]*(len(self.coefs)+len(other.coefs)-1)
        for i,a in enumerate(self.coefs):
            for j,b in enumerate(other.coefs):
                result[i+j] += a*b
        return Poly(result)

    def mod(self, r, m):
        result = [0]*(r+1)
        for i,a in enumerate(self.coefs):
            result[i%r] += a
        return Poly([a%m for a in result])

    def pow(self, n, r, m):
        base, result = self, Poly([1])
        while(n > 0):
            if(n & 1):
                result = (result * base).mod(r, m)
            base = (base * base).mod(r, m)
            n >>= 1
        return result

    def __str__(self):
        return "+".join([f"{a}x^{i}" for i,a in enumerate(self.coefs) if(a != 0)])

    def __eq__(self, other):
        return self.coefs == other.coefs