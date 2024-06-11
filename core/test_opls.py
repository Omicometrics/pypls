import unittest
import os

import numpy as np
import matplotlib.pyplot as plt
from unittest import TestCase

from .opls import correct_fit, correct_x_1d, correct_x_2d

path = os.path.dirname(os.path.abspath(__file__))


def get_data():
    data_texts = np.genfromtxt(os.path.join(path, r"ST002016_AN003285.txt"),
                               delimiter="\t", dtype=str)
    return data_texts


class TestOpls(TestCase):
    def setUp(self):
        data_txt = get_data()
        labels = data_txt[1][1:]
        ix_1 = ((labels == "group:COVID_before_ventilator")
                | (labels == "group:COVID_non-acute")
                | (labels == "group:COVID_ICU-no"))
        ix_2 = ((labels == "group:Healthy"))
        print(ix_1.sum(), ix_2.sum())
        data_txt[data_txt==""] = "0."
        tmp_x = data_txt[2:][:, 1:].astype(np.float64).T
        x = np.ascontiguousarray(np.r_[tmp_x[ix_1], tmp_x[ix_2]])
        self.y = np.zeros(x.shape[0], dtype=np.float64)
        self.y[np.count_nonzero(ix_1):] = 1.

        x_m = x.mean(axis=0)
        x_sd = x.std(axis=0)

        self.x = (x - x_m) / x_sd

    def test_correct_fit(self):
        t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
            self.x.copy(), self.y, 2, 1e-6, 1000)

        jx = self.y == 1.

        fig, ax = plt.subplots()
        ax.plot(t_p[1][jx], t_o[1][jx], "r.")
        ax.plot(t_p[1][~jx], t_o[1][~jx], "b+")
        plt.show()

    def test_correct_x_1d(self):
        t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
            self.x.copy(), self.y, 2, 1e-6, 1000)
        x_corr, tp = correct_x_1d(self.x[0].copy(), w_o, p_o)
        with np.printoptions(precision=3, suppress=True):
            print(x_corr)
            print(tp)

    def test_correct_x_2d(self):
        t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
            self.x.copy(), self.y, 2, 1e-6, 1000)
        x_corr, tp = correct_x_2d(self.x[:, :3].copy(), w_o, p_o)
        with np.printoptions(precision=3, suppress=True):
            print(x_corr)
            print(tp)


if __name__ == '__main__':
    unittest.main()
