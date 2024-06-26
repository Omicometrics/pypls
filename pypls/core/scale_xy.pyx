cimport cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport calloc, free
from libc.math cimport sqrt

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int scale_x_class_(double[:, ::1] x, double[::1] y, int ntrains, int tag,
                        int[::1] sel_var_index, double[::1] vec_minus,
                        double[::1] vec_div):

    cdef:
        Py_ssize_t i, j, a
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t pj = 0
        double * sum_val_g1 = <double *> calloc(p, sizeof(double))
        double * sum_val_g2 = <double *> calloc(p, sizeof(double))
        double * sum_val2 = <double *> calloc(p, sizeof(double))
        double dr = <double> ntrains
        double n1 = 0.
        double n2 = 0.
        double tv, tm, ts

    if tag == 4:
        # min-max scaling
        for j in range(p):
            ts = x[0, j]
            tm = x[0, j]
            for i in range(ntrains):
                tv = x[i, j]
                if tv > ts:
                    ts = tv
                if tv < tm:
                    tm = tv
            if ts - tm > 0.:
                sel_var_index[pj] = <int> j
                vec_minus[pj] = tm
                vec_div[pj] = ts - tm
                pj += 1
    else:
        # autoscaling, pareto scaling or centering
        for i in range(ntrains):
            if y[i] == -1.:
                for j in range(p):
                    tv = x[i, j]
                    sum_val_g1[j] += tv
                    sum_val2[j] += tv * tv
                n1 += 1.
            else:
                for j in range(p):
                    tv = x[i, j]
                    sum_val_g2[j] += tv
                    sum_val2[j] += tv * tv
                n2 += 1.

        for j in range(p):
            tm = (sum_val_g1[j] + sum_val_g2[j]) / dr
            ts = sqrt(sum_val2[j] / dr - tm * tm)
            if ts > 0.000001:
                sel_var_index[pj] = <int> j
                # weighted mean
                vec_minus[pj] = (sum_val_g1[j] / n1 + sum_val_g2[j] / n2) / 2.
                vec_div[pj] = ts
                pj += 1

        if tag == 2:
            # pareto
            for j in range(pj):
                vec_div[j] = sqrt(vec_div[j])

    if tag == 1:
        for i in range(n):
            for j in range(pj):
                a = <ssize_t> sel_var_index[j]
                x[i, j] = x[i, a] - vec_minus[j]
    else:
        for i in range(n):
            for j in range(pj):
                a = <ssize_t> sel_var_index[j]
                x[i, j] = (x[i, a] - vec_minus[j]) / vec_div[j]

    free(sum_val_g1)
    free(sum_val_g2)
    free(sum_val2)

    return <int> pj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int scale_x_reg_(double[:, ::1] x, int ntrains, int tag,
                      int[::1] sel_var_index, double[::1] vec_minus,
                      double[::1] vec_div):

    cdef:
        Py_ssize_t i, j, a
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t pj = 0
        double * sum_val_g = <double *> calloc(p, sizeof(double))
        double * sum_val2 = <double *> calloc(p, sizeof(double))
        double dr = <double> ntrains
        double tv, tm, ts

    if tag == 4:
        # min-max scaling
        for j in range(p):
            ts = x[0, j]
            tm = x[0, j]
            for i in range(ntrains):
                tv = x[i, j]
                if tv > ts:
                    ts = tv
                if tv < tm:
                    tm = tv
            if ts - tm > 0.:
                sel_var_index[pj] = <int> j
                vec_minus[pj] = tm
                vec_div[pj] = ts - tm
                pj += 1
    else:
        # autoscaling, pareto scaling or centering
        for i in range(ntrains):
            for j in range(p):
                tv = x[i, j]
                sum_val_g[j] += tv
                sum_val2[j] += tv * tv

        for j in range(p):
            tm = sum_val_g[j] / dr
            ts = sqrt(sum_val2[j] / dr - tm * tm)
            if ts > 0.000001:
                sel_var_index[pj] = <int> j
                # weighted mean
                vec_minus[pj] = tm
                vec_div[pj] = ts
                pj += 1

    if tag == 1:
        for i in range(n):
            for j in range(pj):
                a = <ssize_t> sel_var_index[j]
                x[i, j] = x[i, a] - vec_minus[j]
    else:
        for i in range(n):
            for j in range(pj):
                a = <ssize_t> sel_var_index[j]
                x[i, j] = (x[i, a] - vec_minus[j]) / vec_div[j]

    free(sum_val_g)
    free(sum_val2)

    return <int> pj


@cython.boundscheck(False)
@cython.wraparound(False)
def scale_x_class(double[:, ::1] x, double[::1] y, int pre_tag):
    """
    Scales x for classification.

    Parameters
    ----------
    x: np.ndarray
        x
    y: np.ndarray
        y
    pre_tag: int
        A tag for pretreatment

    Returns
    -------

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        int[::1] val_ix = np.zeros(p, dtype=np.int32)
        int nval
        double[::1] center_x = np.zeros(p, dtype=np.float64)
        double[::1] norm_x = np.zeros(p, dtype=np.float64)

    nval = scale_x_class_(x, y, <int> n, pre_tag, val_ix, center_x, norm_x)

    return (np.asarray(x[:, :nval]), np.asarray(val_ix[:nval]),
            np.asarray(center_x[:nval]), np.asarray(norm_x[:nval]))


@cython.boundscheck(False)
@cython.wraparound(False)
def scale_x_reg(double[:, ::1] x, int pre_tag):
    """
    Scales x for classification.

    Parameters
    ----------
    x: np.ndarray
        x
    pre_tag: int
        A tag for pretreatment

    Returns
    -------

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        int[::1] val_ix = np.zeros(p, dtype=np.int32)
        int nval
        double[::1] center_x = np.zeros(p, dtype=np.float64)
        double[::1] norm_x = np.zeros(p, dtype=np.float64)

    nval = scale_x_reg_(x, <int> n, pre_tag, val_ix, center_x, norm_x)

    return (np.asarray(x[:, :nval]), np.asarray(val_ix[:nval]),
            np.asarray(center_x[:nval]), np.asarray(norm_x[:nval]))
