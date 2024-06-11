cimport cython

from libc.stdlib cimport calloc, malloc, free
from libc.math cimport sqrt

cimport numpy as np
import numpy as np

from .nipals cimport nipals_c

np.import_array()

DTYPE_F = np.float64
DTYPE = np.int32


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void pls_(double[:, ::1] x, double[::1] y, int num_comp, double tol,
               int max_iter, double[:, ::1] scores, double[:, ::1] loadings,
               double[:, ::1] weights, double[::1] y_weights):
    cdef:
        Py_ssize_t i, j, nc
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        double * px = <double *> calloc(p, sizeof(double))
        double[::1] w = np.zeros(p, dtype=DTYPE_F)
        double[::1] t = np.zeros(n, dtype=DTYPE_F)
        double c, tk

    # iterate through components
    for nc in range(num_comp):
        c = nipals_c(x, y, tol, max_iter, w, t)
        # loadings
        tk = 0.
        for i in range(n):
            tk += t[i] * t[i]
            for j in range(p):
                px[j] += t[i] * x[i, j]

        # loadings
        for j in range(p):
            px[j] /= tk

        # update to next component
        for i in range(n):
            y[i] -= t[i] * c
            for j in range(p):
                x[i, j] -= t[i] * px[j]

        # save to matrix
        for i in range(n):
            scores[nc, i] = t[i]

        for j in range(p):
            loadings[nc, j] = px[j]
            weights[nc, j] = w[j]
            px[j] = 0.

        y_weights[nc] = c

    free(px)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void vip_(double[:, ::1] w, double[:, ::1] t, double[::1] c, double[:, ::1] vip):

    cdef:
        Py_ssize_t i, j, nc
        Py_ssize_t p = w.shape[1]
        Py_ssize_t ncomp = w.shape[0]
        Py_ssize_t n = t.shape[1]
        double * w_weights = <double *> calloc(p, sizeof(double))
        double * yrec = <double *> malloc(n * sizeof(double))
        double pd = <double> p
        double ssy, ssy_nc, wk

    ssy = 0.
    for nc in range(ncomp):
        ssy_nc = 0.
        for i in range(n):
            yrec[i] = t[nc, i] * c[nc]
            ssy_nc += yrec[i] * yrec[i]

        ssy += ssy_nc
        for j in range(p):
            w_weights[j] += ssy_nc * w[nc, j] * w[nc, j]
            vip[nc, j] = sqrt(pd * w_weights[j] / ssy)

    free(w_weights)
    free(yrec)


@cython.boundscheck(False)
@cython.wraparound(False)
def pls_c(double[:, ::1] x, double[::1] y, int num_comp, double tol = 1e-10,
          int max_iter=1000):
    """
    Fit PLS model

    Parameters
    ----------
    x: np.ndarray
        Variable matrix with size n by p, where n number of
        samples/instances, p number of variables
    y: np.ndarray
        Dependent variable with size n by 1
    num_comp: int
        Number of components specified.
    tol: double
        Tolerance for stopping in NIPALS. Defaults to 1e-10.
    max_iter: int
        Maximum number of iteration in NIPALS. Defaults to 1000.

    Returns
    -------
    t: np.ndarray
        Scores
    p: np.ndarray
        Loadings
    w: np.ndarray
        Weights
    c: np.ndarray
        Y-weights
    coefs: np.ndarray
        Coefficients

    """
    cdef:
        Py_ssize_t nc, i, j, k, nb
        Py_ssize_t kb = -1
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        double * tmp_pc = <double *> malloc(num_comp * sizeof(double))
        double[:, ::1] t = np.zeros((num_comp, n), dtype=DTYPE_F)
        double[:, ::1] w = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] ld = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] coefs = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] vips = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] inv_pw
        double[::1] yw = np.zeros(num_comp, dtype=DTYPE_F)
        double v

    pls_(x, y, num_comp, tol, max_iter, t, ld, w, yw)

    for nc in range(num_comp):
        nb = nc + 1
        inv_pw = np.linalg.inv(np.dot(ld[:nb], w[:nb].T))
        for i in range(nb):
            v = 0.
            for j in range(nb):
                v += inv_pw[i, j] * yw[j]
            tmp_pc[i] = v

        for j in range(p):
            v = 0.
            for i in range(nb):
                v += w[i, j] * tmp_pc[i]
            coefs[nc, j] = v

    free(tmp_pc)

    return np.asarray(t), np.asarray(w), np.asarray(ld), np.asarray(yw), np.asarray(coefs)


def pls_vip(double[:, ::1] w, double[:, ::1] t, double[::1] c):
    """
    Calculates VIPs of PLS.

    Parameters
    ----------
    w: np.ndarray
        x-weights.
    t: np.ndarray
        x-scores.
    c: np.ndarray
        y-weights

    Returns
    -------
    np.ndarray
        VIPs

    """
    cdef:
        Py_ssize_t ncomp = w.shape[0]
        Py_ssize_t p = w.shape[1]
        double[:, ::1] vips = np.zeros((ncomp, p), dtype=DTYPE_F)

    vip_(w, t, c, vips)

    return np.asarray(vips)
