cdef:
    int scale_x_class_(double[:, ::1] tt_x, double[::1] tt_y, int ntrains,
                       int tag, int[::1] sel_var_index, double[::1] vec_minus,
                       double[::1] vec_div)
    int scale_x_reg_(double[:, ::1] tt_x, int ntrains, int tag,
                     int[::1] sel_var_index, double[::1] vec_minus,
                     double[::1] vec_div)
