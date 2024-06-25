cdef:
    double nipals_c(double[:, ::1] x, double[::1] y, double tol, int max_iter,
                    double[::1] w, double[::1] t, double[::1] u)
