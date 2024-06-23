import unittest
import os

from pypls.core.cross_validation import (kfold_cv_opls, kfold_cv_pls,
                                         kfold_prediction, kfold_cv_pls_reg)
from pypls.core.pls import pls_c, pls_vip

import numpy as np
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))


class TestOpls(unittest.TestCase):
    def setUp(self):
        data_txt = np.genfromtxt(os.path.join(path, r"ST002016_AN003285.txt"),
                                 delimiter="\t", dtype=str)
        labels = data_txt[1][1:]
        ix_1 = ((labels == "group:COVID_before_ventilator")
                | (labels == "group:COVID_non-acute")
                | (labels == "group:COVID_ICU-no"))
        ix_2 = labels == "group:Healthy"
        data_txt[data_txt == ""] = "0."
        tmp_x = data_txt[2:][:, 1:].astype(np.float64).T
        self.x = np.ascontiguousarray(np.r_[tmp_x[ix_1], tmp_x[ix_2]])
        self.y = np.ones(self.x.shape[0], dtype=np.float64)
        self.y[:np.count_nonzero(ix_1)] = -1.

        # from S Wold, M Sjöström, L. Eriksson. PLS-regression: a basic tool
        # of chemometrics. Chem Intel Lab Sys. 2001, 58(2), 109-130.
        self.x2 = np.ascontiguousarray([
            [0.23, 0.31, -0.55, 254.2, 2.126, -0.02, 82.2],
            [-0.48, -0.60, 0.51, 303.6, 2.994, -1.24, 112.3],
            [-0.61, -0.77, 1.20, 287.9, 2.994, -1.08, 103.7],
            [0.45, 1.54, -1.40, 282.9, 2.933, -0.11, 99.1],
            [-0.11, -0.22, 0.29, 335.0, 3.458, -1.19, 127.5],
            [-0.51, -0.64, 0.76, 311.6, 3.243, -1.43, 120.5],
            [0.00, 0.00, 0.00, 224.9, 1.662, 0.03, 65.0],
            [0.15, 0.13, -0.25, 337.2, 3.856, -1.06, 140.6],
            [1.20, 1.80, -2.10, 322.6, 3.350, 0.04, 131.7],
            [1.28, 1.70, -2.00, 324.0, 3.518, 0.12, 131.5],
            [-0.77, -0.99, 0.78, 336.6, 2.933, -2.26, 144.3],
            [0.90, 1.23, -1.60, 336.3, 3.860, -0.33, 132.3],
            [1.56, 1.79, -2.60, 366.1, 4.638, -0.05, 155.8],
            [0.38, 0.49, -1.50, 288.5, 2.876, -0.31, 106.7],
            [0.00, -0.04, 0.09, 266.7, 2.279, -0.40, 88.5],
            [0.17, 0.26, -0.58, 283.9, 2.743, -0.53, 105.3],
            [1.85, 2.25, -2.70, 401.8, 5.755, -0.31, 185.9],
            [0.89, 0.96, -1.70, 377.8, 4.791, -0.84, 162.7],
            [0.71, 1.22, -1.60, 295.1, 3.054, -0.13, 115.6]
        ], dtype=np.float64)
        self.y2 = np.fromiter(
            [8.5, 8.2, 8.5, 11.0, 6.3, 8.8, 7.1, 10.1, 16.8, 15.0, 7.9,
             13.3, 11.2, 8.2, 7.4, 8.8, 9.9, 8.8, 12.0], np.float64)

    def test_kfold_cv_opls(self):
        (pred_y, q2, r2xyo, r2xcorr, no_mcs, t_o, t_p, p_p, p_o, n_opt,
         n0) = kfold_cv_opls(self.x, self.y, 5, 3, 1e-10, 1000)

        # print("==" * 20)
        # with np.printoptions(suppress=True, precision=4):
        #     print(np.asarray(q2[:10]))
        # #     print(p_p)
        # #     print(r2xyo)
        #     print(pred_y[n_opt])
        #     print(no_mcs[:n_opt + 3])
        #
        # print(n_opt, n0)

        # jx = self.y == 1.
        # fig, ax = plt.subplots()
        # ax.plot(t_p[n_opt][jx], t_o[n_opt][jx], "r.")
        # ax.plot(t_p[n_opt][~jx], t_o[n_opt][~jx], "b+")
        # plt.show()
        #
        # fig, ax = plt.subplots()
        # ax.bar(np.arange(p_p.shape[1]), p_p.mean(axis=0))
        # plt.show()

    def test_kfold_cv_pls(self):
        pred_y, q2, no_mcs, cv_t, cv_p, n_opt, n0 = kfold_cv_pls(
            self.x, self.y, 5, 3, 1e-10, 1000)

        # with np.printoptions(suppress=True, precision=4):
        #     print(q2[:n_opt + 3])
        #     print(no_mcs[:n_opt + 3])
        #     print(pred_y[n_opt])
        # print(n_opt, n0)

        # jx = self.y == 1.
        # fig, ax = plt.subplots()
        # ax.plot(cv_t[0][jx], cv_t[1][jx], "r.")
        # ax.plot(cv_t[0][~jx], cv_t[1][~jx], "b+")
        # plt.show()

    def test_kfold_prediction(self):
        print("\ntest kfold_prediction:")
        q2_o, r2_o, err_o = kfold_prediction(self.x, self.y, 5, 3, 3, 1,
                                             1e-10, 1000)
        print(q2_o, r2_o, err_o)
        q2_p, r2_o, err_p = kfold_prediction(self.x, self.y, 5, 4, 3, 2,
                                             1e-10, 1000)
        print(q2_p, r2_o, err_p)

    def test_kfold_cv_pls2(self):
        # xs = (self.x2 - self.x2.mean(axis=0)) / self.x2.std(axis=0)
        # xs[:, 5] = self.x2[:, 5]
        # xs *= 1.5
        ty = np.ones(self.x2.shape[0], dtype=bool)
        ty[[12, 16, 17]] = False
        x2 = self.x2[ty]
        y2 = self.y2[ty]
        pred_y, q2, cv_t, cv_p, rmse, n_opt, n0 = kfold_cv_pls_reg(
            x2, y2, 16, 3, 1e-10, 1000)
        print("test PLS, v2:")
        print(self.x2.shape)
        with np.printoptions(suppress=True, precision=4):
            print(q2)

        fig, ax = plt.subplots()
        ax.plot(y2, pred_y, "r.")
        plt.show()

    def test_kfold_cv_pls_vip(self):
        p: int = self.x2.shape[1]
        ty = np.ones(self.x2.shape[0], dtype=bool)
        ty[[12, 16, 17]] = False
        x2 = np.zeros((np.count_nonzero(ty), p + 4), dtype=np.float64)
        x2[:, :p] = self.x2[ty]
        x2[:, p] = x2[:, 0] ** 2
        x2[:, p + 1] = x2[:, 1] ** 2
        x2[:, p + 2] = x2[:, 2] ** 2
        x2[:, p + 3] = x2[:, 5] ** 2
        y2 = self.y2[ty]
        pred_y, q2, cv_t, cv_p, rmse, n_opt, n0 = kfold_cv_pls_reg(
            x2, y2, 16, 3, 1e-10, 1000)
        print("test PLS, vip:")
        print(self.x2.shape)
        with np.printoptions(suppress=True, precision=4):
            print(q2)

        xs = (x2 - x2.mean(axis=0)) / x2.std(axis=0)
        t, w, ld, c, coefs = pls_c(xs.copy(), y2.copy(), 3)
        vips = pls_vip(w, t, c)

        fig, ax = plt.subplots()
        ax.plot(y2, pred_y, "r.")
        plt.show()

        fig, ax = plt.subplots()
        ax.bar(np.arange(p + 4), vips[2])
        plt.show()


if __name__ == '__main__':
    unittest.main()
