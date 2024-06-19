cimport cython

from libc.stdlib cimport malloc, calloc, free
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int scale_x_class(double[:, ::1] tt_x, double[::1] tt_y, int ntrains,
                       int tag, int[::1] sel_var_index):

    cdef:
        Py_ssize_t i, j, a
        Py_ssize_t n = tt_x.shape[0]
        Py_ssize_t p = tt_x.shape[1]
        Py_ssize_t pj = 0
        double * sum_val_g1 = <double *> calloc(p, sizeof(double))
        double * sum_val_g2 = <double *> calloc(p, sizeof(double))
        double * sum_val2 = <double *> calloc(p, sizeof(double))
        double * minus_arr = <double *> malloc(p * sizeof(double))
        double * normalizer_arr = <double *> malloc(p * sizeof(double))
        double dr = <double> ntrains
        double n1 = 0.
        double n2 = 0.
        double tv, tm, ts

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
                sel_var_index[pj] = <int> j
                minus_arr[pj] = tm
                normalizer_arr[pj] = ts
                pj += 1
    else:
        # autoscaling, pareto scaling or centering
        for i in range(ntrains):
            if tt_y[i] == -1.:
                for j in range(p):
                    tv = tt_x[i, j]
                    sum_val_g1[j] += tv
                    sum_val2[j] += tv * tv
                n1 += 1.
            else:
                for j in range(p):
                    tv = tt_x[i, j]
                    sum_val_g2[j] += tv
                    sum_val2[j] += tv * tv
                n2 += 1.

        for j in range(p):
            tm = (sum_val_g1[j] + sum_val_g2[j]) / dr
            ts = sqrt(sum_val2[j] / dr - tm * tm)
            if ts > 0.000001:
                sel_var_index[pj] = <int> j
                # weighted mean
                minus_arr[pj] = (sum_val_g1[j] / n1 + sum_val_g2[j] / n2) / 2.
                normalizer_arr[pj] = ts
                pj += 1

    if tag == 1:
        for i in range(n):
            for j in range(pj):
                a = <ssize_t> sel_var_index[j]
                tt_x[i, j] = tt_x[i, a] - minus_arr[j]
    else:
        for i in range(n):
            for j in range(pj):
                a = <ssize_t> sel_var_index[j]
                tt_x[i, j] = (tt_x[i, a] - minus_arr[j]) / normalizer_arr[j]

    free(minus_arr)
    free(normalizer_arr)
    free(sum_val_g1)
    free(sum_val_g2)
    free(sum_val2)

    return <int> pj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int scale_x_reg(double[:, ::1] tt_x, int ntrains, int tag, int[::1] sel_var_index):

    cdef:
        Py_ssize_t i, j, a
        Py_ssize_t n = tt_x.shape[0]
        Py_ssize_t p = tt_x.shape[1]
        Py_ssize_t pj = 0
        double * sum_val_g = <double *> calloc(p, sizeof(double))
        double * sum_val2 = <double *> calloc(p, sizeof(double))
        double * minus_arr = <double *> malloc(p * sizeof(double))
        double * normalizer_arr = <double *> malloc(p * sizeof(double))
        double dr = <double> ntrains
        double tv, tm, ts

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
                sel_var_index[pj] = <int> j
                minus_arr[pj] = tm
                normalizer_arr[pj] = ts
                pj += 1
    else:
        # autoscaling, pareto scaling or centering
        for i in range(ntrains):
            for j in range(p):
                tv = tt_x[i, j]
                sum_val_g[j] += tv
                sum_val2[j] += tv * tv

        for j in range(p):
            tm = sum_val_g[j] / dr
            ts = sqrt(sum_val2[j] / dr - tm * tm)
            if ts > 0.000001:
                sel_var_index[pj] = <int> j
                # weighted mean
                minus_arr[pj] = tm
                normalizer_arr[pj] = ts
                pj += 1

    if tag == 1:
        for i in range(n):
            for j in range(pj):
                a = <ssize_t> sel_var_index[j]
                tt_x[i, j] = tt_x[i, a] - minus_arr[j]
    else:
        for i in range(n):
            for j in range(pj):
                a = <ssize_t> sel_var_index[j]
                tt_x[i, j] = (tt_x[i, a] - minus_arr[j]) / normalizer_arr[j]

    free(minus_arr)
    free(normalizer_arr)
    free(sum_val_g)
    free(sum_val2)

    return <int> pj
