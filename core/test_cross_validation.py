import unittest
import os

from .cross_validation import kfold_cv_opls, kfold_cv_pls

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
        self.y = np.zeros(self.x.shape[0], dtype=np.float64)
        self.y[np.count_nonzero(ix_1):] = 1.

    def test_kfold_cv_opls(self):
        (q2, r2xyo, r2xcorr, no_mcs, t_o, t_p, p_p, p_o, n_opt,
         n0) = kfold_cv_opls(self.x, self.y, 5, 3, 1e-10, 1000)

        # print("==" * 20)
        # print(self.x.shape)
        # with np.printoptions(suppress=True, precision=4):
        #     print(p_p)
        #     print(r2xyo)
        #     print(r2xcorr)
        #     print(no_mcs)

        print(n_opt, n0)

        jx = self.y == 1.
        fig, ax = plt.subplots()
        ax.plot(t_p[n_opt][jx], t_o[n_opt][jx], "r.")
        ax.plot(t_p[n_opt][~jx], t_o[n_opt][~jx], "b+")
        plt.show()

        fig, ax = plt.subplots()
        ax.bar(np.arange(p_p.shape[1]), p_p.mean(axis=0))
        plt.show()

    def test_kfold_cv_pls(self):
        q2, no_mcs, cv_t, cv_p, n_opt, n0 = kfold_cv_pls(self.x, self.y, 5, 3,
                                                         1e-10, 1000)

        with np.printoptions(suppress=True, precision=4):
            print(q2[:n_opt + 3])
            print(no_mcs[:n_opt + 3])
        print(n_opt, n0)

        jx = self.y == 1.
        fig, ax = plt.subplots()
        ax.plot(cv_t[0][jx], cv_t[1][jx], "r.")
        ax.plot(cv_t[0][~jx], cv_t[1][~jx], "b+")
        plt.show()


if __name__ == '__main__':
    unittest.main()
