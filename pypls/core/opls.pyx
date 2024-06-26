cimport cython

from libc.math cimport sqrt
from libc.stdlib cimport malloc, calloc, free

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
        double[::1] u = np.zeros(n, dtype=DTYPE_F)
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
    c = nipals_c(x, y, tol, max_iter, w, t, u)
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
        c = nipals_c(x, y, tol, max_iter, w, t, u)
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void vip_(double[:, ::1] tortho, double[:, ::1] portho, double[::1] tp,
               double[::1] pp, double[::1] vip_o, double[::1] vip_p,
               double[::1] vip_tot):

    cdef:
        Py_ssize_t i, j, a, b
        Py_ssize_t n = tp.shape[0]
        Py_ssize_t p = pp.shape[0]
        Py_ssize_t npc = tortho.shape[0]
        double * x_rec = <double *> calloc(n * p, sizeof(double))
        double * tmp_vip_o = <double *> calloc(p, sizeof(double))
        double k = <double> p / 2.
        double ssx_c = 0.
        double ssx_o = 0.
        double ssx_p = 0.
        double ssx_k, v, ko, kp, q, r_ssx_pc, rj

    # orthogonal SSX
    for a in range(npc):
        ssx_k = 0.
        b = 0
        for i in range(n):
            for j in range(p):
                v = tortho[a, i] * portho[a, j]
                ssx_k += v * v
                x_rec[b] += v
                b += 1
        for j in range(p):
            tmp_vip_o[j] += portho[a, j] * portho[a, j] * ssx_k

    # predictive, orthogonal and cumulative SSX
    b = 0
    for i in range(n):
        for j in range(p):
            v = tp[i] * pp[j]
            x_rec[b] += v
            # predictive
            ssx_p += v * v
            # orthogonal
            q = x_rec[b]
            ssx_o += q * q
            # cumulative, total
            ssx_c += (q + v) * (q + v)
            b += 1

    # orthogonal, predictive and total VIP
    r_ssx_pc = ssx_p / ssx_c
    ko = k / (ssx_o / ssx_c)
    kp = k / r_ssx_pc
    for j in range(p):
        v = pp[j] * pp[j]
        rj = tmp_vip_o[j] / ssx_c
        vip_o[j] = sqrt(ko * rj)
        vip_p[j] = sqrt(kp * (((v * ssx_p) / ssx_c) + v))
        vip_tot[j] = sqrt(k * (v * r_ssx_pc + rj + v))

    free(x_rec)
    free(tmp_vip_o)


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
        Py_ssize_t pc_k = <ssize_t> num_comp - 1
        double[:, ::1] tortho = np.zeros((num_comp, n), dtype=DTYPE_F)
        double[:, ::1] portho = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] wortho = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] tpred = np.zeros((num_comp, n), dtype=DTYPE_F)
        double[:, ::1] wpred = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] ppred = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[:, ::1] coefs = np.zeros((num_comp, p), dtype=DTYPE_F)
        double[::1] pred_y = np.zeros(n, dtype=DTYPE_F)
        double[::1] cx = np.zeros(num_comp, dtype=DTYPE_F)
        double[::1] covars = np.zeros(p, dtype=DTYPE_F)

    correct_fit_(x, y, num_comp, tol, max_iter, tortho, portho, wortho, tpred,
                 cx, ppred, wpred, covars)

    for i in range(num_comp):
        for j in range(p):
            coefs[i, j] = cx[i] * covars[j]
    
    # perform prediction
    for i in range(n):
        tv = 0.
        for j in range(p):
            tv += coefs[pc_k, j] * x[i, j]
        pred_y[i] = tv

    return (np.asarray(pred_y), np.asarray(tortho), np.asarray(portho),
            np.asarray(wortho), np.asarray(tpred), np.asarray(wpred),
            np.asarray(ppred), np.asarray(coefs), np.asarray(cx),
            np.asarray(covars))


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


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def summary_opls(double[:, ::1] x, double[::1] y, double[:, ::1] pred_scores,
                 double[:, ::1] ortho_scores, double[:, ::1] pred_loadings,
                 double[:, ::1] ortho_loadings, double[::1] y_weights,
                 int num_pc):
    """
    Summarizes OPLS fittings

    Parameters
    ----------
    x: np.ndarray
        x, n samples by p variables
    y: np.ndarray
        y, n samples by 1
    pred_scores: np.ndarray
        Predictive scores, a LVs by n samples
    ortho_scores: np.ndarray
        Orthogonal scores, a LVs by p variables
    pred_loadings: np.ndarray
        Predictive loadings, a LVs by p variables
    ortho_loadings: np.ndarray
        Orthogonal loadings, a LVs by p variables
    y_weights: np.ndarray
        y weights
    num_pc: int
        Number of latent variables.

    Returns
    -------

    """
    cdef:
        Py_ssize_t i, j, a, ji
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t na = <ssize_t> num_pc - 1
        double * w = <double *> calloc(p, sizeof(double))
        double * rec_x_c = <double *> calloc(p * n, sizeof(double))
        double * rec_y_c = <double *> malloc(n * sizeof(double))
        double[::1] r2x = np.zeros(num_pc, dtype=DTYPE_F)
        double[::1] r2y = np.zeros(num_pc, dtype=DTYPE_F)
        double[::1] cov = np.zeros(p, dtype=DTYPE_F)
        double[::1] corr = np.zeros(p, dtype=DTYPE_F)
        double ss_tp = 0.
        double ssx = 0.
        double ssy = 0.
        double tv, s, rss, rss_a, d

    # SSX and SSY
    for i in range(n):
        tv = 0.
        for j in range(p):
            tv += x[i, j] * x[i, j]
        ssx += tv
        ssy += y[i] * y[i]

    # predictive scores and weights
    for i in range(n):
        s = pred_scores[na, i]
        ss_tp += s * s
        for j in range(p):
            w[j] += s * x[i, j]

    # covariance and correlation for assessing variable importance
    for j in range(p):
        cov[j] = w[j] / ss_tp
        tv = 0.
        for i in range(n):
            tv += x[i, j] * x[i, j]
        corr[j] = w[j] / sqrt(tv * ss_tp)

    for a in range(num_pc):
        # reconstruct the matrix using scores and loadings
        ji = 0
        rss = 0.
        for i in range(n):
            for j in range(p):
                # current Tortho * Portho^T
                rec_x_c[ji] += ortho_scores[a, i] * ortho_loadings[a, j]
                tv = pred_scores[a, i] * pred_loadings[a, j]
                d = x[i, j] - rec_x_c[ji] - tv
                rss += d * d
                ji += 1
        r2x[a] = 1. - rss / ssx

        # reconstruct dependent vector y
        rss = 0.
        for i in range(n):
            rec_y_c[i] = pred_scores[a, i] * y_weights[a]
            d = y[i] - rec_y_c[i]
            rss += d * d
        r2y[a] = 1. - rss / ssy

    free(w)
    free(rec_x_c)
    free(rec_y_c)

    return np.asarray(cov), np.asarray(corr), np.asarray(r2x), np.asarray(r2y)


@cython.boundscheck(False)
@cython.wraparound(False)
def opls_vip(double[:, ::1] tortho, double[:, ::1] portho, double[::1] tp, double[::1] pp):
    """
    Calculates VIPs of OPLS.

    Parameters
    ----------
    tortho: np.ndarray
        Orthogonal scores T_O
    portho: np.ndarray
        Orthogonal loadings T_O
    tp: np.ndarray
        Predictive scores
    pp: np.ndarray
        Predictive loadings

    Returns
    -------
        Orthogonal, predictive and total VIPs

    References
    ----------
    [1] Galindo-Prieto B, et al. Variable influence on projection (VIP)
        for orthogonal projections to latent structures (OPLS).
        J Chemometrics. 2014, 28(8), 623-632.

    Notes
    -----
        Here, only VIPs from orthonomal loadings P were implemented,
        i.e., VIP4 of ref, which is recommended.

    """
    cdef:
        Py_ssize_t p = pp.shape[0]
        double[::1] vip_o = np.zeros(p, dtype=DTYPE_F)
        double[::1] vip_p = np.zeros(p, dtype=DTYPE_F)
        double[::1] vip_t = np.zeros(p, dtype=DTYPE_F)

    vip_(tortho, portho, tp, pp, vip_o, vip_p, vip_t)

    return np.asarray(vip_o), np.asarray(vip_p), np.asarray(vip_t)
