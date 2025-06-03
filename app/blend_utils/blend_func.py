# implementation of Paper: 
# Smooth Piecewise Polynomial Blending Operations for Implicit Shapes

import numpy as np
import scipy
from math import comb, factorial

class UpAbsFunc():
    """docstring for UpAbsFunc."""
    def __init__(self, n, delta):
        # n: order of polynomial; delta: blending range
        self.n = n
        self.delta = delta
        self.n_delta = n / delta

        # precompute weights
        weights = np.asarray([comb(n-1, k) for k in range(n)], dtype=float)
        weights[1::2] *= -1.
        factor = factorial(n+1)* 2**n
        weights /= float(factor)

        self.weights = weights

        self.xs_val = np.asarray([n-2*k-1 for k in range(n)])

    def __call__(self, xs):
        return (1./self.n_delta)* self.Abs_n(self.n_delta* xs)

    def G(self, xs):
        xs_plus1 = xs + 1
        xs_minus1 = xs - 1

        plus = np.sign(xs_plus1)* np.power(xs_plus1, self.n+1)
        minus = np.sign(xs_minus1)* np.power(xs_minus1, self.n+1)
        return plus - minus
    
    def Abs_n(self, xs):
        vals = np.add.outer(xs, self.xs_val)
        vals = self.G(vals)
        return vals @ self.weights


class SmoothMaxMin():
    """docstring for SmoothMaxMin."""
    def __init__(self, n, delta):
        self.n = n
        self.delta = delta

        self.smooth_abs = UpAbsFunc(n, delta)
    
    def max(self, xs, ys):
        abs_diff = self.smooth_abs(xs - ys)
        return (xs + ys + abs_diff) /2.

    def min(self, xs, ys):
        abs_diff = self.smooth_abs(xs - ys)
        return (xs + ys - abs_diff) /2.
    
    def abs(self, xs):
        return self.smooth_abs(xs)
    

if __name__ == "__main__":
    # func = UpAbsFunc(1, 0.2)
    # a = np.asarray([0.])
    smooth = SmoothMaxMin(1, 0.2)
    x, y, z = np.random.rand(3)* 0.04
    val1 = smooth.max(x, y)
    val1 = smooth.max(val1, z)

    val2 = smooth.max(y,z)
    val2 = smooth.max(x, val2)

    print(f'val1: {val1}, val2:{val2}')
