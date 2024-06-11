cdef:
    void correct_fit_(double[:, ::1] x, double[::1] y, int num_comp,
                      double tol, int max_iter, double[:, ::1] tortho,
                      double[:, ::1] portho, double[:, ::1] wortho,
                      double[:, ::1] scores, double[::1] y_weight,
                      double[:, ::1] loadings, double[:, ::1] weights,
                      double[::1] score_weights)
    void correct_1d_(double[::1] x, double[:, ::1] wortho,
                     double[:, ::1] portho, int num_comp, double[::1] scores)
    void correct_2d_(double[:, ::1] x, double[:, ::1] wortho,
                     double[:, ::1] portho, double[:, ::1] scores)
