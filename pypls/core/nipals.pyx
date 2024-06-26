cimport cython

from libc.stdlib cimport malloc, calloc, free
from libc.math cimport sqrt

cimport numpy as np
import numpy as np

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double nipals_c(double[:, ::1] x, double[::1] y, double tol, int max_iter,
                     double[::1] w, double[::1] t, double[::1] u):

    cdef:
        Py_ssize_t i, j
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        int niter = 0
        double * tmp_w = <double *> calloc(p, sizeof(double))
        double d = tol * 10.
        double c = 0.
        double unorm, wnorm, tnorm, tk, ck, uk, d1, d2

    for i in range(n):
        u[i] = y[i]

    while d > tol and niter <= max_iter:
        # w = X'u/u'u
        unorm = 0.
        for i in range(n):
            uk = u[i]
            unorm += uk * uk
            for j in range(p):
                tmp_w[j] += uk * x[i, j]

        wnorm = 0.
        for j in range(p):
            tmp_w[j] /= unorm
            wnorm += tmp_w[j] * tmp_w[j]

        # normalize w: w = w / sqrt(w'w)
        wnorm = sqrt(wnorm)
        for j in range(p):
            w[j] = tmp_w[j] / wnorm
            tmp_w[j] = 0.

        # calculate t, norm t, and c
        tnorm = 0.
        ck = 0.
        for i in range(n):
            tk = 0.
            for j in range(p):
                tk += x[i, j] * w[j]
            t[i] = tk
            tnorm += tk * tk
            ck += tk * y[i]
        c = ck / tnorm

        # update u and calculate the tolerance
        d1 = 0.
        d2 = 0.
        ck = c / (c * c)
        for i in range(n):
            uk = y[i] * ck
            d1 += (uk - u[i]) ** 2
            d2 += uk * uk
            u[i] = uk

        d = sqrt(d1 / d2)
        niter += 1

    free(tmp_w)

    return c


def nipals(double[:, ::1] x, double[::1] y, double tol=1e-10, int max_iter=1000):
    """
    Non-linear Iterative Partial Least Squares

    Parameters
    ----------
    x: np.ndarray
        Variable matrix with size n by p, where n number of samples,
        p number of variables.
    y: np.ndarray
        Dependent variable with size n by 1.
    tol: float
        Tolerance for the convergence.
    max_iter: int
        Maximal number of iterations.

    Returns
    -------
    w: np.ndarray
        Weights with size p by 1.
    c: float
        Y-weight
    t: np.ndarray
        Scores with size n by 1

    References
    ----------
    [1] Wold S, et al. PLS-regression: a basic tool of chemometrics.
        Chemometr Intell Lab Sys 2001, 58, 109–130.
    [2] Bylesjo M, et al. Model Based Preprocessing and Background
        Elimination: OSC, OPLS, and O2PLS. in Comprehensive Chemometrics.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        double[::1] w = np.zeros(p, dtype=np.float64)
        double[::1] t = np.zeros(n, dtype=np.float64)
        double[::1] u = np.zeros(n, dtype=np.float64)
        double c

    c = nipals_c(x, y, tol, max_iter, w, t, u)

    return np.asarray(w), np.asarray(t), c
