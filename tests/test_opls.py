import unittest
import os

import numpy as np
import matplotlib.pyplot as plt

from pypls.core.opls import correct_fit, correct_x_1d, correct_x_2d, opls_vip

path = os.path.dirname(os.path.abspath(__file__))


class TestOpls(unittest.TestCase):
    def setUp(self):
        self.x = np.ascontiguousarray([[-2.18, -2.18],
                                       [1.84, -0.16],
                                       [-0.48, 1.52],
                                       [0.83, 0.83]], dtype=np.float64)
        self.y = np.fromiter([2., 2., 0., -4.], dtype=np.float64)

        # metabolomics data
        data_txt = np.genfromtxt(os.path.join(path, r"ST000415_AN000657.txt"),
                                 delimiter="\t", dtype=str)
        tmp_x = data_txt[1:][:, 1:].astype(np.float64).T
        self.x_metab = np.ascontiguousarray(tmp_x)
        self.y_metab = np.ones(self.x_metab.shape[0], dtype=np.float64)
        self.y_metab[:8] = -1.

    def test_correct_fit(self):
        yp, t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
            self.x.copy(), self.y.copy(), 1, 1e-6, 1000)
        w_o_t = np.fromiter([-0.89, 0.45], dtype=np.float64)
        p_o_t = np.fromiter([-1.16, -0.09], dtype=np.float64)
        t_o_t = np.fromiter([0.97, -1.71, 1.11, -0.37], dtype=np.float64)
        self.assertTrue(np.allclose(w_o[0], w_o_t, atol=0.01))
        self.assertTrue(np.allclose(p_o[0], p_o_t, atol=0.01))
        self.assertTrue(np.allclose(t_o[0], t_o_t, atol=0.01))

        w_p_t = np.fromiter([-0.45, -0.89], dtype=np.float64)
        p_p_t = np.fromiter([-0.45, -0.89], dtype=np.float64)
        self.assertTrue(np.allclose(w_p[0], w_p_t, atol=0.01))
        self.assertTrue(np.allclose(p_p[0], p_p_t, atol=0.01))

        coefs_t = np.fromiter([-0.41, -0.82], dtype=np.float64)
        self.assertTrue(np.allclose(coefs, coefs_t, atol=0.01))

    def test_correct_x_1d(self):
        yp, t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
            self.x.copy(), self.y, 1, 1e-6, 1000)
        x_corr, tp = correct_x_1d(self.x[0].copy(), w_o, p_o)
        with np.printoptions(precision=3, suppress=True):
            print(x_corr)
            print(tp)

    def test_correct_x_2d(self):
        yp, t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
            self.x.copy(), self.y, 1, 1e-6, 1000)
        x_corr, tp = correct_x_2d(self.x[:, :3].copy(), w_o, p_o)
        with np.printoptions(precision=3, suppress=True):
            print(x_corr)
            print(tp)

    def test_vip(self):
        xs = ((self.x_metab - self.x_metab.mean(axis=0))
              / self.x_metab.std(axis=0))
        print("**" * 40)
        print(xs.shape)
        npc: int = 2
        yp, t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
            xs.copy(), self.y_metab.copy(), npc, 1e-6, 1000)
        vip_o, vip_p, vip_t = opls_vip(t_o, p_o, t_p[npc - 1], p_p[npc - 1])

        fig, ax = plt.subplots()
        ax.plot(t_p[npc - 1][:8], t_o[npc - 1][:8],
                "o", mfc="none", mec="firebrick", ms=6)
        ax.plot(t_p[npc - 1][8:], t_o[npc - 1][8:],
                "s", mfc="none", mec="royalblue", ms=6)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.plot([0, 0], [y0, y1], "--k")
        ax.plot([x0, x1], [0, 0], "--k")
        ax.set_xlim(left=x0, right=x1)
        ax.set_ylim(bottom=y0, top=y1)
        plt.show()

        fig, ax = plt.subplots()
        ax.bar(np.arange(vip_t.shape[0]), vip_t)
        plt.show()


if __name__ == '__main__':
    unittest.main()
