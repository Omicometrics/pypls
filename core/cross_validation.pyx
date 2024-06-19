cimport cython

from libc.stdlib cimport malloc, calloc, free
from libc.math cimport fmod, sqrt

import numpy as np
cimport numpy as np

from .opls cimport correct_fit_
from .pls cimport pls_
from .scale_xy cimport scale_x_class, scale_x_reg

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
            if (pred_y[a, i] >= 0. and y[i] > 0.) or (pred_y[a, i] < 0. and y[i] < 0.):
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
                          double[::1] ssx_ortho, double[::1] corr_scores,
                          double[:, ::1] pred_y, double[:, ::1] pred_scores):

    cdef:
        Py_ssize_t nc, i, j, jk
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t n_comps = portho.shape[0]
        double * tmp_t = <double *> malloc(n * sizeof(double))
        double * tmp_tp = <double *> calloc(n * p, sizeof(double))
        double tv, ty, u, v, ssx_p, ssx_o

    for nc in range(n_comps):
        for i in range(n):
            tv = 0.
            for j in range(p):
                tv += x[i, j] * wortho[nc, j]
            tmp_t[i] = tv

        # the first orthogonal scores
        if nc == 0:
            for i in range(n):
                corr_scores[i] = tmp_t[i]

        # update data matrix by removing orthogonal components and
        # calculate SSX
        jk = 0
        ssx_p = 0.
        ssx_o = 0.
        for i in range(n):
            tv = 0.
            ty = 0.
            for j in range(p):
                v = tmp_t[i] * portho[nc, j]
                u = x[i, j] - v
                x[i, j] = u
                tmp_tp[jk] += v
                tv += u * cov_xy[j]
                ty += u * coefs[nc, j]
                ssx_p += u * u
                ssx_o += tmp_tp[jk] * tmp_tp[jk]
                jk += 1
            pred_scores[nc, i] = tv
            pred_y[nc, i] = ty
        # SSX of corrected matrix
        ssx_corr[nc] = ssx_p
        # cumulative SSX of orthogonal matrix
        ssx_ortho[nc] = ssx_o

    free(tmp_t)
    free(tmp_tp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double cal_ssy(double[::1] y, int ntrains):
    cdef:
        Py_ssize_t n = y.shape[0]
        double tm = 0.
        double t2 = 0.
        double d

    for i in range(ntrains):
        tm += y[i]
    tm /= <double> ntrains

    for i in range(ntrains, n):
        d = y[i] - tm
        t2 += d * d

    return t2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kfold_cv_opls(double[:, ::1] x, double[::1] y, int k, int pre_tag,
                  double tol, int max_iter):
    """
    K-fold cross validation, for OPLS.

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
        Py_ssize_t i, j, a, j_cv, jv
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t dm = min(n, p)
        int val_npc = <int> dm
        int kr, nsel, n_pc, n_opt
        int[::1] test_ix = np.zeros(n, dtype=DTYPE)
        int[::1] no_mis_class = np.zeros(dm, dtype=DTYPE)
        int[::1] sel_var_ix = np.zeros(p, dtype=DTYPE)
        double[:, ::1] tt_x = np.zeros((n, p), dtype=DTYPE_F)
        double[:, ::1] tr_t_ortho = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] tr_p_ortho = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] tr_w_ortho = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] tr_t_p = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] tr_p_p = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] tr_w = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] tmp_x = np.zeros((n, p), dtype=DTYPE_F)
        double[:, ::1] tmp_pred_tp = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] tmp_pred_y = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] tr_coefs = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] cv_t_ortho = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] cv_t_p = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] cv_p_p = np.zeros((dm * k, p), dtype=DTYPE_F)
        double[:, ::1] cv_p_o = np.zeros((dm * k, p), dtype=DTYPE_F)
        double[:, ::1] cv_pred_y = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] ssx_corr = np.zeros((k, dm), dtype=DTYPE_F)
        double[:, ::1] ssx_ortho = np.zeros((k, dm), dtype=DTYPE_F)
        double[::1] tmp_corr_tp = np.zeros(n, dtype=DTYPE_F)
        double[::1] tr_w_y = np.zeros(dm, dtype=DTYPE_F)
        double[::1] tr_var_xy = np.zeros(p, dtype=DTYPE_F)
        double[::1] tt_y = np.zeros(n, dtype=DTYPE_F)
        double[::1] q2 = np.zeros(dm, dtype=DTYPE_F)
        double[::1] r2xcorr = np.zeros(dm, dtype=DTYPE_F)
        double[::1] r2xyo = np.zeros(dm, dtype=DTYPE_F)
        double[::1] tmp_y = np.zeros(n, dtype=DTYPE_F)
        double[::1] cv_gix = np.zeros(n, dtype=DTYPE_F)
        double[::1] pressy = np.zeros(dm, dtype=DTYPE_F)
        double ssx = 0.
        double ssy = 0.
        double ik = <double> k
        double tv, tc, d, ssy_k

    for i in range(n):
        cv_gix[i] = fmod(<double> i, ik)

    for j_cv in range(k):
        # extract training and testing data
        kr = get_train_tests(x, y, cv_gix, <double> j_cv, tt_x, tt_y, test_ix)

        # scaling
        nsel = scale_x_class(tt_x, tt_y, kr, pre_tag, sel_var_ix)
        # SSX and SSY of testing data
        for i in range(kr, n):
            for j in range(nsel):
                ssx += tt_x[i, j] * tt_x[i, j]
        ssy += cal_ssy(tt_y, kr)

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
                     tr_t_ortho, tr_p_ortho, tr_w_ortho, tr_t_p, tr_w_y,
                     tr_p_p, tr_w, tr_var_xy)

        # coefficients of various components
        for a in range(n_pc):
            i = a * k + j_cv
            for j in range(nsel):
                tr_coefs[a, j] = tr_w_y[a] * tr_var_xy[j]
                jv = <ssize_t> sel_var_ix[j]
                cv_p_p[i, jv] = tr_p_p[a, j]
                cv_p_o[i, jv] = tr_p_ortho[a, j]

        # correction, prediction and scores
        correct_predict(tmp_x[kr:][:, :nsel], tr_coefs, tr_w_ortho, tr_p_ortho,
                        tr_var_xy, ssx_corr[j_cv], ssx_ortho[j_cv],
                        tmp_corr_tp, tmp_pred_y, tmp_pred_tp)

        # save the parameters for model quality assessments
        # Orthogonal and predictive scores
        for i in range(n - kr):
            j = <ssize_t> test_ix[i]
            for a in range(n_pc):
                # the first y-orthogonal scores
                cv_t_ortho[a, j] = tmp_corr_tp[i]
                # the first predictive scores
                cv_t_p[a, j] = tmp_pred_tp[a, i]
                cv_pred_y[a, j] = tmp_pred_y[a, i]
                d = tmp_pred_y[a, i] - tt_y[kr + i]
                pressy[a] += d * d

        # reset the values
        tr_var_xy[:] = 0.

    # the optimal number of PCs
    n_opt = get_opt_pcs(cv_pred_y, y, val_npc, no_mis_class)

    # summary the cross validation results
    for a in range(val_npc):
        tv = 0.
        tc = 0.
        for j_cv in range(k):
            tv += ssx_ortho[j_cv, a]
            tc += ssx_corr[j_cv, a]
        r2xyo[a] = tv / ssx
        r2xcorr[a] = tc / ssx
        q2[a] = 1. - pressy[a] / ssy

    a = n_opt * k

    return (np.asarray(cv_pred_y), np.asarray(q2[:val_npc]),
            np.asarray(r2xyo[:val_npc]), np.asarray(r2xcorr[:val_npc]),
            np.asarray(no_mis_class[:val_npc]), np.asarray(cv_t_ortho[:val_npc]),
            np.asarray(cv_t_p[:val_npc]), np.asarray(cv_p_p[a: a + k]),
            np.asarray(cv_p_o[a: a + k]), n_opt, val_npc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kfold_cv_pls(double[:, ::1] x, double[::1] y, int k, int pre_tag,
                 double tol, int max_iter):
    """
    K-fold cross validation, for PLS.

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
    tol: double
        Tolerance for NIPALS algorithm for PLS and OPLS.
    max_iter: int
        Maximum iteration of the algorithm.

    Returns
    -------

    """
    cdef:
        Py_ssize_t i, j, a, j_cv, b, jk, jb
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t dm = min(n, p)
        int val_npc = <int> dm
        int kr, nsel, n_pc, n_opt
        int[::1] test_ix = np.zeros(n, dtype=DTYPE)
        int[::1] no_mis_class = np.zeros(dm, dtype=DTYPE)
        int[::1] sel_var_ix = np.zeros(p, dtype=DTYPE)
        double * tmp_pc = <double *> malloc(dm * sizeof(double))
        double * tmp_coefs = <double *> malloc(p * sizeof(double))
        double * tmp_press = <double *> calloc(dm, sizeof(double))
        double[:, ::1] tt_x = np.zeros((n, p), dtype=DTYPE_F)
        double[:, ::1] tr_t_p = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] tr_p_p = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] tr_w = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] cv_pred_y = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] cv_t = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] cv_p = np.zeros((dm * k, p), dtype=DTYPE_F)
        double[::1] tr_w_y = np.zeros(dm, dtype=DTYPE_F)
        double[::1] tt_y = np.zeros(n, dtype=DTYPE_F)
        double[::1] q2 = np.zeros(dm, dtype=DTYPE_F)
        double[::1] cv_gix = np.zeros(n, dtype=DTYPE_F)
        double[:, ::1] inv_pw
        double ssy = 0.
        double ik = <double> k
        double tv, tc, d

    for i in range(n):
        cv_gix[i] = fmod(<double> i, ik)

    for j_cv in range(k):
        # extract training and testing data
        kr = get_train_tests(x, y, cv_gix, <double> j_cv, tt_x, tt_y, test_ix)
        # scaling
        nsel = scale_x_class(tt_x, tt_y, kr, pre_tag, sel_var_ix)

        # SSY of testing data
        ssy += cal_ssy(tt_y, kr)

        # fitting the model
        n_pc = min(kr, <int> p)
        if n_pc < val_npc:
            val_npc = n_pc
        pls_(tt_x[:kr][:, :nsel], tt_y[:kr], n_pc, tol, max_iter, tr_t_p,
             tr_p_p, tr_w, tr_w_y)

        for a in range(n_pc):
            b = a + 1
            inv_pw = np.linalg.inv(np.dot(tr_p_p[:b][:, :nsel], tr_w[:b][:, :nsel].T))
            for i in range(b):
                tv = 0.
                for j in range(i, b):
                    tv += inv_pw[i, j] * tr_w_y[j]
                tmp_pc[i] = tv

            jk = a * k + j_cv
            for j in range(nsel):
                tc = 0.
                for i in range(b):
                    tc += tr_w[i, j] * tmp_pc[i]
                tmp_coefs[j] = tc
                jb = <ssize_t> sel_var_ix[j]
                cv_p[jk, jb] = tr_p_p[a, j]

            # prediction using the coefficients
            for i in range(n - kr):
                tv = 0.
                tc = 0.
                b = <ssize_t> kr + i
                for j in range(nsel):
                    tv += tmp_coefs[j] * tt_x[b, j]
                    tc += tr_w[a, j] * tt_x[b, j]
                d = tt_y[b] - tv
                b = <ssize_t> test_ix[i]
                cv_pred_y[a, b] = tv
                cv_t[a, b] = tc
                # PRESS
                tmp_press[a] += d * d

    for a in range(n_pc):
        q2[a] = 1. - tmp_press[a] / ssy

    # the optimal number of PCs
    n_opt = get_opt_pcs(cv_pred_y, y, val_npc, no_mis_class)

    free(tmp_press)
    free(tmp_pc)
    free(tmp_coefs)

    a = n_opt * k

    return (np.asarray(cv_pred_y), np.asarray(q2[:val_npc]),
            np.asarray(no_mis_class[:val_npc]), np.asarray(cv_t[:val_npc]),
            np.asarray(cv_p[a: a + k]), n_opt, val_npc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def kfold_cv_pls_reg(double[:, ::1] x, double[::1] y, int k, int pre_tag,
                     double tol, int max_iter):
    """
    K-fold cross validation, for PLS regression.

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
    tol: double
        Tolerance for NIPALS algorithm for PLS and OPLS.
    max_iter: int
        Maximum iteration of the algorithm.

    Returns
    -------

    """
    cdef:
        Py_ssize_t i, j, a, j_cv, b, jk, jb
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t dm = min(n, p)
        int val_npc = <int> dm
        int kr, nsel, n_pc, n_opt
        int[::1] test_ix = np.zeros(n, dtype=DTYPE)
        int[::1] sel_var_ix = np.zeros(p, dtype=DTYPE)
        double * tmp_pc = <double *> malloc(dm * sizeof(double))
        double * tmp_coefs = <double *> malloc(p * sizeof(double))
        double[:, ::1] tt_x = np.zeros((n, p), dtype=DTYPE_F)
        double[:, ::1] tr_t_p = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] tr_p_p = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] tr_w = np.zeros((dm, p), dtype=DTYPE_F)
        double[:, ::1] pred_y = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] cv_t = np.zeros((dm, n), dtype=DTYPE_F)
        double[:, ::1] cv_p = np.zeros((dm * k, p), dtype=DTYPE_F)
        double[::1] cv_q2 = np.zeros(dm, dtype=DTYPE_F)
        double[::1] cv_pred_y = np.zeros(n, dtype=DTYPE_F)
        double[::1] cv_rmse = np.zeros(dm, dtype=DTYPE_F)
        double[::1] tr_w_y = np.zeros(dm, dtype=DTYPE_F)
        double[::1] tt_y = np.zeros(n, dtype=DTYPE_F)
        double[::1] cv_gix = np.zeros(n, dtype=DTYPE_F)
        double[:, ::1] inv_pw
        double ik = <double> k
        double nk = <double> n
        double ssy = 0.
        double tv, tc, d, ym, sse

    for i in range(n):
        cv_gix[i] = fmod(<double> i, ik)

    for j_cv in range(k):
        # extract training and testing data
        kr = get_train_tests(x, y, cv_gix, <double> j_cv, tt_x, tt_y, test_ix)
        # scaling
        nsel = scale_x_reg(tt_x, kr, pre_tag, sel_var_ix)

        # centering y
        ym = 0.
        for i in range(kr):
            ym += tt_y[i]
        ym /= <double> kr
        for i in range(n):
            tt_y[i] -= ym

        for i in range(kr, n):
            ssy += tt_y[i] * tt_y[i]

        # fitting the model
        n_pc = min(kr, <int> p)
        if n_pc < val_npc:
            val_npc = n_pc
        pls_(tt_x[:kr][:, :nsel], tt_y[:kr], n_pc, tol, max_iter, tr_t_p,
             tr_p_p, tr_w, tr_w_y)

        for a in range(n_pc):
            b = a + 1
            inv_pw = np.linalg.inv(np.dot(tr_p_p[:b][:, :nsel], tr_w[:b][:, :nsel].T))
            for i in range(b):
                tv = 0.
                for j in range(i, b):
                    tv += inv_pw[i, j] * tr_w_y[j]
                tmp_pc[i] = tv

            jk = a * k + j_cv
            for j in range(nsel):
                tc = 0.
                for i in range(b):
                    tc += tr_w[i, j] * tmp_pc[i]
                tmp_coefs[j] = tc
                jb = <ssize_t> sel_var_ix[j]
                cv_p[jk, jb] = tr_p_p[a, j]

            # prediction using the coefficients
            for i in range(n - kr):
                tv = 0.
                tc = 0.
                b = <ssize_t> kr + i
                for j in range(nsel):
                    tv += tmp_coefs[j] * tt_x[b, j]
                    tc += tr_w[a, j] * tt_x[b, j]
                d = tt_y[b] - tv
                b = <ssize_t> test_ix[i]
                pred_y[a, b] = tv + ym
                cv_t[a, b] = tc
                # PRESS
                cv_rmse[a] += d * d

    for a in range(val_npc):
        cv_q2[a] = 1. - cv_rmse[a] / ssy

    n_opt = 0
    sse = cv_rmse[n_opt]
    cv_rmse[0] = sqrt(sse / nk)
    for a in range(1, val_npc):
        if cv_rmse[a] < sse:
            n_opt = <int> a
            sse = cv_rmse[a]
        cv_rmse[a] = sqrt(cv_rmse[a] / nk)

    for i in range(n):
        cv_pred_y[i] = pred_y[n_opt, i]

    free(tmp_pc)
    free(tmp_coefs)

    a = n_opt * k

    return (np.asarray(cv_pred_y), np.asarray(cv_q2[:val_npc]),
            np.asarray(cv_t[:val_npc]), np.asarray(cv_p[a: a + k]),
            np.asarray(cv_rmse[:val_npc]), n_opt, val_npc)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def kfold_prediction(double[:, ::1] x, double[::1] y, int k, int num_pc,
                     int pre_tag, int alg_tag, double tol, int max_iter):
    """
    K-fold cross validated prediction.

    """
    cdef:
        Py_ssize_t i, j, a, j_cv
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t pc_k = <ssize_t> num_pc - 1
        int kr, nsel
        int[::1] test_ix = np.zeros(n, dtype=DTYPE)
        int[::1] sel_var_ix = np.zeros(p, dtype=DTYPE)
        double * coefs = <double *> malloc(p * sizeof(double))
        double * pred_y = <double *> malloc(n * sizeof(double))
        double * tmp_pc = <double *> malloc(num_pc * sizeof(double))
        double[:, ::1] tt_x = np.zeros((n, p), dtype=DTYPE_F)
        double[:, ::1] tr_t_ortho = np.zeros((num_pc, n), dtype=DTYPE_F)
        double[:, ::1] tr_p_ortho = np.zeros((num_pc, p), dtype=DTYPE_F)
        double[:, ::1] tr_w_ortho = np.zeros((num_pc, p), dtype=DTYPE_F)
        double[:, ::1] tr_t_p = np.zeros((num_pc, n), dtype=DTYPE_F)
        double[:, ::1] tr_p_p = np.zeros((num_pc, p), dtype=DTYPE_F)
        double[:, ::1] tr_w = np.zeros((num_pc, p), dtype=DTYPE_F)
        double[::1] tr_w_y = np.zeros(num_pc, dtype=DTYPE_F)
        double[::1] tr_var_xy = np.zeros(p, dtype=DTYPE_F)
        double[::1] tt_y = np.zeros(n, dtype=DTYPE_F)
        double[::1] cv_gix = np.zeros(n, dtype=DTYPE_F)
        double[:, ::1] inv_pw
        double pressy = 0.
        double ssy = 0.
        double ik = <double> k
        double tv, tc, d, q2

    for i in range(n):
        cv_gix[i] = fmod(<double> i, ik)

    for j_cv in range(k):
        # extract training and testing data
        kr = get_train_tests(x, y, cv_gix, <double> j_cv, tt_x, tt_y, test_ix)
        # scaling
        nsel = scale_x_class(tt_x, tt_y, kr, pre_tag, sel_var_ix)
        ssy += cal_ssy(tt_y, kr)

        if alg_tag == 1:
            # OPLS-DA
            # correct and fit
            correct_fit_(tt_x[:kr][:, :nsel], tt_y[:kr], num_pc, tol, max_iter,
                         tr_t_ortho, tr_p_ortho, tr_w_ortho, tr_t_p, tr_w_y,
                         tr_p_p, tr_w, tr_var_xy)

            # correction
            for a in range(num_pc):
                for i in range(kr, n):
                    tv = 0.
                    for j in range(nsel):
                        tv += tt_x[i, j] * tr_w_ortho[a, j]
                    # update data matrix by removing orthogonal components
                    for j in range(nsel):
                        tt_x[i, j] -= tv * tr_p_ortho[a, j]

            # coefficients
            for j in range(nsel):
                coefs[j] = tr_w_y[pc_k] * tr_var_xy[j]

            # reset the values
            tr_var_xy[:] = 0.
        else:
            pls_(tt_x[:kr][:, :nsel], tt_y[:kr], num_pc, tol, max_iter, tr_t_p,
                 tr_p_p, tr_w, tr_w_y)

            inv_pw = np.linalg.inv(np.dot(tr_p_p[:, :nsel], tr_w[:, :nsel].T))
            for i in range(num_pc):
                tv = 0.
                for j in range(i, num_pc):
                    tv += inv_pw[i, j] * tr_w_y[j]
                tmp_pc[i] = tv

            for j in range(nsel):
                tc = 0.
                for i in range(num_pc):
                    tc += tr_w[i, j] * tmp_pc[i]
                coefs[j] = tc

        # prediction using the coefficients
        for i in range(kr, n):
            tv = 0.
            for j in range(nsel):
                tv += coefs[j] * tt_x[i, j]
            a = <ssize_t> test_ix[i - kr]
            pred_y[a] = tv
            d = tt_y[i] - tv
            # PRESS
            pressy += d * d

    # summary the cross validation results
    q2 = 1. - pressy / ssy

    # error rate
    d = 0.
    for i in range(n):
        # y is -1 or 1
        if (pred_y[i] >= 0. and y[i] < 0.) or (pred_y[i] < 0. and y[i] > 0.):
            d += 1.

    free(coefs)
    free(pred_y)
    free(tmp_pc)

    return q2, d / <double> n
