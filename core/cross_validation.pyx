cimport cython

from libc.stdlib cimport malloc, calloc, free
from libc.math cimport fmod, fabs, sqrt

import numpy as np
cimport numpy as np

from .opls cimport correct_fit_
from .pls cimport _pls

np.import_array()

DTYPE_F = np.float64
DTYPE = np.int32


@cython.wraparound(False)
@cython.boundscheck(False)
cdef int get_opt_pcs(double[:, ::1] pred_y, double[::1] y, int npc,
                     int[::1] no_mis_class):

    cdef:
        Py_ssize_t i, a
        Py_ssize_t n = y.shape[0]
        int n_opt = npc
        int nt = <int> n
        int nk, nj
        int r = nt
        double d

    for a in range(npc):
        nj = 0
        for i in range(n):
            # y is 0 or 1
            if pred_y[a, i] > 0.:
                d = y[i] - 1.
            else:
                d = y[i]
            if d == 0.:
                nj += 1
        # mis-classification rate
        nk = nt - nj
        no_mis_class[a] = nk
        if nk < r:
            n_opt = <int> a
            r = nk

    return n_opt


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int get_train_tests(double[:, ::1] x, double[::1] y, double[::1] tag_index,
                         double tag, double[:, ::1] tt_x, double[::1] tt_y,
                         int[::1] test_index):
    cdef:
        Py_ssize_t i, j, a
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        int * test_ix = <int *> malloc(n * sizeof(int))
        int kr = 0
        int ke = 0

    for i in range(n):
        if tag_index[i] == tag:
            test_ix[ke] = <int> i
            ke += 1
        else:
            for j in range(p):
                tt_x[kr, j] = x[i, j]
            tt_y[kr] = y[i]
            kr += 1

    for i in range(ke):
        test_index[i] = test_ix[i]
        a = <ssize_t> test_ix[i]
        for j in range(p):
            tt_x[kr, j] = x[a, j]
        tt_y[kr] = y[a]
        kr += 1

    free(test_ix)

    return kr - ke


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void correct_predict(double[:, ::1] x, double[:, ::1] coefs,
                          double[:, ::1] wortho, double[:, ::1] portho,
                          double[::1] cov_xy, double[::1] ssx_corr,
                          double[::1] ssx_ortho, double[:, ::1] corr_scores,
                          double[:, ::1] pred_y, double[:, ::1] pred_scores):

    cdef:
        Py_ssize_t nc, i, j, jk
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t n_comps = portho.shape[0]
        double * tmp_t = <double *> malloc(n * sizeof(double))
        double * tmp_tp = <double *> calloc(n * p, sizeof(double))
        double tv, ty, u, ssx

    for nc in range(n_comps):
        # correction scores
        for i in range(n):
            tv = 0.
            for j in range(p):
                tv += x[i, j] * wortho[nc, j]
            tmp_t[i] = tv
            corr_scores[nc, i] = tv

        # update data matrix
        ssx = 0.
        for i in range(n):
            tv = 0.
            ty = 0.
            for j in range(p):
                u = x[i, j] - tmp_t[i] * portho[nc, j]
                tv += u * cov_xy[j]
                ty += u * coefs[i, j]
                ssx += u * u
                x[i, j] = u
            pred_scores[nc, i] = tv
            pred_y[nc, i] = ty
        # SSX of corrected matrix
        ssx_corr[nc] = ssx

        # SSX of orthogonal matrix
        jk = 0
        ssx = 0.
        for i in range(n):
            for j in range(p):
                u = tmp_tp[jk] + corr_scores[nc, i] * portho[nc, j]
                ssx += u * u
                tmp_tp[jk] = u
                jk += 1
        ssx_ortho[nc] = ssx

    free(tmp_t)
    free(tmp_tp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int scale_xy(double[:, ::1] tt_x, double[::1] tt_y, int ntrains, int tag):

    cdef:
        Py_ssize_t i, j, a
        Py_ssize_t n = tt_x.shape[0]
        Py_ssize_t p = tt_x.shape[1]
        Py_ssize_t pj = 0
        int * sel_ix = <int *> malloc(p * sizeof(int))
        double * minus_arr = <double *> malloc(p * sizeof(double))
        double * normalizer_arr = <double *> malloc(p * sizeof(double))
        double dr = <double> ntrains
        double ssx = 0.
        double ssy = 0.
        double tv, tm, ts, minus_y, norm_y

    if tag == 4:
        # min-max scaling
        for j in range(p):
            ts = tt_x[0, j]
            tm = tt_x[0, j]
            for i in range(ntrains):
                tv = tt_x[i, j]
                if tv > ts:
                    ts = tv
                if tv < tm:
                    tm = tv
            if ts - tm > 0.:
                sel_ix[pj] = <int> j
                minus_arr[pj] = tm
                normalizer_arr[pj] = ts
                pj += 1

        tm = tt_y[0]
        ts = tt_y[0]
        for i in range(ntrains):
            if tt_y[i] > ts:
                ts = tt_y[i]
            if tt_y[i] < tm:
                tm = tt_y[i]
        minus_y = tm
        norm_y = ts - tm
    else:
        # autoscaling, pareto scaling or centering
        for j in range(p):
            tv = 0.
            tm = 0.
            for i in range(ntrains):
                tv += tt_x[i, j] * tt_x[i, j]
                tm += tt_x[i, j]
            tm /= dr
            ts = sqrt(tv / dr - tm * tm)
            if ts > 0.00000001:
                sel_ix[pj] = <int> j
                minus_arr[pj] = tm
                normalizer_arr[pj] = ts
                pj += 1

        tv = 0.
        tm = 0.
        for i in range(ntrains):
            tm += tt_y[i]
            tv += tt_y[i] * tt_y[i]
        tm /= dr
        ts = sqrt(tv / dr - tm * tm)
        minus_y = tm

        if tag == 2:
            # pareto scaling
            for j in range(pj):
                normalizer_arr[j] = sqrt(normalizer_arr[j])
            norm_y = sqrt(norm_y)

    if tag == 1:
        for i in range(n):
            for j in range(pj):
                a = <ssize_t> sel_ix[j]
                tt_x[i, j] = tt_x[i, a] - minus_arr[j]
            tt_y[i] -= minus_y
    else:
        for i in range(n):
            for j in range(pj):
                a = <ssize_t> sel_ix[j]
                tt_x[i, j] = (tt_x[i, a] - minus_arr[j]) / normalizer_arr[j]
            tt_y[i] = (tt_y[i] - minus_y) / norm_y

    free(sel_ix)
    free(minus_arr)
    free(normalizer_arr)

    return <int> pj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kfold_cv_opls(double[:, ::1] x, double[::1] y, int k, int pre_tag,
                  double tol, int max_iter):
    """
    K-fold cross validation.

    Parameters
    ----------
    x: np.ndarray
        x
    y: np.ndarray
        y
    k: int
        Number of folds for cross validation.
    pre_tag: int
        Tag for pretreatment of the data matrix.
            1 for mean centering.
            2 for pareto scaling.
            3 for autoscaling.
            4 for min-max scaling.
    alg_tag: int
        Tag for the algorithm.
    tol: double
        Tolerance for NIPALS algorithm for PLS and OPLS.
    max_iter: int
        Maximum iteration of the algorithm.

    Returns
    -------

    """
    cdef:
        Py_ssize_t i, j, a, j_cv
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t dm = min(n, p)
        int val_npc = <int> dm
        int kr, nsel, n_pc, n_opt
        int[::1] test_ix = np.zeros(n, dtype=DTYPE)
        int[::1] no_mis_class = np.zeros(dm, dtype=DTYPE)
        double[:, ::1] tt_x = np.zeros((n, p), dtype=DTYPE_F)
        double[:, ::1] tr_t_othro = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] tr_p_othro = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] tr_w_othro = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] tr_t_p = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] tr_p_p = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] tr_w = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] te_t_p = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] press_y = np.zeros((n, dm), dtype=DTYPE_F)
        double[:, ::1] t_ortho = np.zeros((n, dm), dtype=DTYPE_F)
        double[:, ::1] tmp_x = np.zeros((n, p), dtype=DTYPE_F)
        double[:, ::1] tmp_corr_tp = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] tmp_pred_tp = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] tmp_pred_y = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] test_coefs = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] cv_t_ortho = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] cv_t_p = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] cv_pred_y = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] ssx_corr = np.zeros((k, dm), dtype=DTYPE_F)
        double[:, ::1] ssx_ortho = np.zeros((k, dm), dtype=DTYPE_F)
        double[::1] tr_w_y = np.zeros(dm, dtype=DTYPE_F)
        double[::1] tr_var_xy = np.zeros(p, dtype=DTYPE_F)
        double[::1] tt_y = np.zeros(n, dtype=DTYPE_F)
        double[::1] q2 = np.zeros(dm, dtype=DTYPE_F)
        double[::1] r2xcorr = np.zeros(dm, dtype=DTYPE_F)
        double[::1] r2xyo = np.zeros(dm, dtype=DTYPE_F)
        double[::1] tmp_y = np.zeros(n, dtype=DTYPE_F)
        double[::1] tmp_ssx_corr = np.zeros(dm, dtype=DTYPE_F)
        double[::1] cv_group_tags = np.zeros(n, dtype=DTYPE_F)
        double[::1] pressy = np.zeros(dm, dtype=DTYPE_F)
        double ssx = 0.
        double ssy = 0.
        double ik = <double> k
        double c, tv, tc, d

    for i in range(n):
        cv_group_tags[i] = fmod(<double> i, ik)

    for j_cv in range(k):
        # extract training and testing data
        c = <double> i
        kr = get_train_tests(x, y, cv_group_tags, c, tt_x, tt_y, test_ix)

        # scaling
        nsel = scale_xy(tt_x, tt_y, kr, pre_tag)
        # SSX and SSY of testing data
        for i in range(kr, n):
            for j in range(nsel):
                ssx += tt_x[i, j] * tt_x[i, j]
            ssy += tt_y[i] * tt_y[i]

        # copy of the training and testing data matrix
        for i in range(n):
            for j in range(nsel):
                tmp_x[i, j] = tt_x[i, j]
            tmp_y[i] = tt_y[i]

        # fitting the model
        n_pc = min(kr, <int> p)
        if n_pc < val_npc:
            val_npc = n_pc
        correct_fit_(tmp_x[:kr][:, :nsel], tmp_y[:kr], n_pc, tol, max_iter,
                     tr_t_othro, tr_p_othro, tr_w_othro, tr_t_p, tr_w_y,
                     tr_p_p, tr_w, tr_var_xy)

        # coefficients of various components
        for i in range(n_pc):
            for j in range(nsel):
                test_coefs[i, j] = tr_w_y[i] * tr_var_xy[j]

        # correction, prediction and scores
        correct_predict(tmp_x[kr:][:, :nsel], test_coefs, tr_w_othro,
                        tr_p_othro, tr_var_xy, ssx_corr[j_cv], ssx_ortho[j_cv],
                        tmp_corr_tp, tmp_pred_y, tmp_pred_tp)

        # save the parameters for model quality assessments
        # Orthogonal and predictive scores
        for i in range(n - kr):
            j = <ssize_t> test_ix[i]
            for a in range(n_pc):
                cv_t_ortho[a, j] = tmp_corr_tp[a, i]
                cv_t_p[a, j] = tmp_pred_tp[a, i]
                cv_pred_y[a, j] = tmp_pred_y[a, i]
                d = tmp_pred_y[a, i] - tt_y[kr + i]
                pressy[a] += d * d

    # the optimal number of PCs
    n_opt = get_opt_pcs(cv_pred_y, y, val_npc, no_mis_class)

    # summary the cross validation results
    for a in range(n_pc):
        tv = 0.
        tc = 0.
        for j_cv in range(k):
            tv += ssx_ortho[j_cv, a]
            tc += ssx_corr[j_cv, a]
        r2xyo[a] = tv / ssx
        r2xcorr[a] = tc / ssx
        q2[a] = 1. - pressy[a] / ssy

    return (np.asarray(q2[:n_pc]), np.asarray(r2xyo[:n_pc]),
            np.asarray(r2xcorr[:n_pc]), np.asarray(no_mis_class[:n_pc]),
            np.asarray(cv_t_ortho[:n_pc]), np.asarray(cv_t_p[:n_pc]),
            n_opt, val_npc)