cdef:
    int scale_x_class(double[:, ::1] tt_x, double[::1] tt_y, int ntrains,
                      int tag, int[::1] sel_var_index)
    int scale_x_reg(double[:, ::1] tt_x, int ntrains, int tag, int[::1] sel_var_index)
