cdef:
    void pls_(double[:, ::1] x, double[::1] y, int num_comp, double tol,
              int max_iter, double[:, ::1] scores, double[:, ::1] loadings,
              double[:, ::1] weights, double[::1] y_weights)
