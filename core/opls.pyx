cimport cython

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

from .nipals cimport nipals_c

np.import_array()

DTYPE_F = np.float64


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void correct_fit_(double[:, ::1] x, double[::1] y, int num_comp,
                       double tol, int max_iter, double[:, ::1] tortho,
                       double[:, ::1] portho, double[:, ::1] wortho,
                       double[:, ::1] scores, double[::1] y_weight,
                       double[:, ::1] loadings, double[:, ::1] weights,
                       double[::1] score_weights):

    cdef:
        Py_ssize_t i, j, nc
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        double * tp = <double *> malloc(n * sizeof(double))
        double * w_o = <double *> malloc(p * sizeof(double))
        double * t_o = <double *> malloc(n * sizeof(double))
        double * p_o = <double *> malloc(p * sizeof(double))
        double * p_p = <double *> malloc(p * sizeof(double))
        double[::1] w = np.zeros(p, dtype=DTYPE_F)
        double[::1] t = np.zeros(n, dtype=DTYPE_F)
        double norm_y, tv, c, wnorm, tnorm, wv, pv, tc, tk

    # X-y variations
    norm_y = 0.
    for i in range(n):
        norm_y += y[i] * y[i]
        for j in range(p):
            score_weights[j] += y[i] * x[i, j]

    # normalize the weights
    tv = 0.
    for j in range(p):
        tv += score_weights[j] * score_weights[j]
    tv = sqrt(tv)

    for j in range(p):
        score_weights[j] /= tv

    # scores
    for i in range(n):
        tv = 0.
        for j in range(p):
            tv += x[i, j] * score_weights[j]
        tp[i] = tv

    # components
    c = nipals_c(x, y, tol, max_iter, w, t)
    tv = 0.
    for i in range(n):
        tv += t[i] * t[i]
    for j in range(p):
        pv = 0.
        for i in range(n):
            pv += t[i] * x[i, j]
        p_p[j] = pv / tv

    for nc in range(num_comp):
        # orthoganol weights
        tv = 0.
        for j in range(p):
            tv += score_weights[j] * p_p[j]

        wnorm = 0.
        for j in range(p):
            wv = p_p[j] - tv * score_weights[j]
            w_o[j] = wv
            wnorm += wv * wv

        wnorm = sqrt(wnorm)
        for j in range(p):
            w_o[j] /= wnorm
            # save to matrix
            wortho[nc, j] = w_o[j]

        # orthogonal scores
        tnorm = 0.
        for i in range(n):
            tv = 0.
            for j in range(p):
                tv += x[i, j] * w_o[j]
            t_o[i] = tv
            tnorm += tv * tv
            # save to matrix
            tortho[nc, i] = tv

        # orthogonal loadings
        for j in range(p):
            tv = 0.
            for i in range(n):
                tv += t_o[i] * x[i, j]
            p_o[j] = tv / tnorm
            # save to matrix
            portho[nc, j] = p_o[j]

        # update X to the residue matrix
        for i in range(n):
            for j in range(p):
                x[i, j] -= t_o[i] * p_o[j]

        # predictive scores
        tv = 0.
        for j in range(p):
            tv += p_o[j] * score_weights[j]

        tnorm = 0.
        tc = 0.
        for i in range(n):
            tk = tp[i] - t_o[i] * tv
            tp[i] = tk
            scores[nc, i] = tk
            tnorm += tk * tk
            tc += y[i] * tk

        y_weight[nc] = tc / tnorm

        # next component
        c = nipals_c(x, y, tol, max_iter, w, t)
        tv = 0.
        for i in range(n):
            tv += t[i] * t[i]
        for j in range(p):
            pv = 0.
            for i in range(n):
                pv += t[i] * x[i, j]
            p_p[j] = pv / tv
            # save to matrix
            loadings[nc, j] = p_p[j]
            weights[nc, j] = w[j]

    free(tp)
    free(w_o)
    free(t_o)
    free(p_o)
    free(p_p)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void correct_1d_(double[::1] x, double[:, ::1] wortho,
                      double[:, ::1] portho, int num_comp, double[::1] scores):
    cdef:
        Py_ssize_t nc, j
        Py_ssize_t p = x.shape[0]
        double tv

    for nc in range(num_comp):
        tv = 0.
        for j in range(p):
            tv += x[j] * wortho[nc, j]

        for j in range(p):
            x[j] -= tv * portho[nc, j]
        scores[nc] = tv


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void correct_2d_(double[:, ::1] x, double[:, ::1] wortho,
                      double[:, ::1] portho, double[:, ::1] scores):
    cdef:
        Py_ssize_t nc, i, j
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t n_comps = portho.shape[0]
        double * tmp_t = <double *> malloc(n * sizeof(double))
        double tv

    # scores
    for nc in range(n_comps):
        for i in range(n):
            tv = 0.
            for j in range(p):
                tv += x[i, j] * wortho[nc, j]
            tmp_t[i] = tv
            scores[nc, i] = tv

        # update data matrix
        for i in range(n):
            for j in range(p):
                x[i, j] -= tmp_t[i] * portho[nc, j]

    free(tmp_t)


@cython.wraparound(False)
@cython.boundscheck(False)
def correct_fit(double[:, ::1] x, double[::1] y, int num_comp, double tol, int max_iter):
    """
    Corrects and fits x to y.

    Parameters
    ----------
    x: np.ndarray
        Independent variable matrix.
    y: np.ndarray
        Dependent variable array.
    num_comp: int
        Number of components.
    tol: double
        Tolerance for NIPALS.
    max_iter: int
        Maximum iteration for NIPALS.

    Returns
    -------

    """
    cdef:
        Py_ssize_t i, j
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        double[:, ::1] tortho = np.zeros((num_comp, n), dtype=DTYPE_F)
        double[:, ::1] portho = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] wortho = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] tpred = np.zeros((num_comp, n), dtype=DTYPE_F)
        double[:, ::1] wpred = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] ppred = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] coefs = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[::1] cx = np.zeros(num_comp, dtype=DTYPE_F)
        double[::1] covars = np.zeros(p, dtype=DTYPE_F)

    correct_fit_(x, y, num_comp, tol, max_iter, tortho, portho, wortho, tpred,
                 cx, ppred, wpred, covars)

    for i in range(num_comp):
        for j in range(p):
            coefs[i, j] = cx[i] * covars[j]

    return (np.asarray(tortho), np.asarray(portho), np.asarray(wortho),
            np.asarray(tpred), np.asarray(wpred), np.asarray(ppred),
            np.asarray(coefs), np.asarray(cx), np.asarray(covars))


@cython.boundscheck(False)
@cython.wraparound(False)
def correct_x_1d(double[::1] x, double[:, ::1] wortho, double[:, ::1] portho):
    """
    Corrects 1D array x.

    Parameters
    ----------
    x: np.ndarray
        Independent variable array.
    wortho: np.ndarray
        Orthogonal weights.
    portho: np.ndarray
        Orthogonal loadings.

    Returns
    -------
    np.ndarray
        Corrected array
    np.ndarray
        Orthogonal scores

    """
    cdef:
        int num_comp = <int> wortho.shape[0]
        double[::1] scores = np.zeros(num_comp, dtype=DTYPE_F)

    correct_1d_(x, wortho, portho, num_comp, scores)

    return np.asarray(x), np.asarray(scores)


@cython.wraparound(False)
@cython.boundscheck(False)
def correct_x_2d(double[:, ::1] x, double[:, ::1] wortho, double[:, ::1] portho):
    """
    Corrects 2D matrix.

    Parameters
    ----------
    x: np.ndarray
        Independent data matrix.
    wortho: np.ndarray
        Orthogonal weights.
    portho: np.ndarray
        Orthogonal loadings.

    Returns
    -------

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        int num_comp = <int> wortho.shape[0]
        double[:, ::1] scores = np.zeros((num_comp, n), dtype=DTYPE_F)

    correct_2d_(x, wortho, portho, scores)

    return np.asarray(x), np.asarray(scores)
