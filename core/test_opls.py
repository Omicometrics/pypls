import unittest
import os

import numpy as np

from .opls import correct_fit, correct_x_1d, correct_x_2d

path = os.path.dirname(os.path.abspath(__file__))


class TestOpls(unittest.TestCase):
    def setUp(self):
        self.x = np.ascontiguousarray([[-2.18, -2.18],
                                       [1.84, -0.16],
                                       [-0.48, 1.52],
                                       [0.83, 0.83]], dtype=np.float64)
        self.y = np.fromiter([2., 2., 0., -4.], dtype=np.float64)

    def test_correct_fit(self):
        t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
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
        t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
            self.x.copy(), self.y, 1, 1e-6, 1000)
        x_corr, tp = correct_x_1d(self.x[0].copy(), w_o, p_o)
        with np.printoptions(precision=3, suppress=True):
            print(x_corr)
            print(tp)

    def test_correct_x_2d(self):
        t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
            self.x.copy(), self.y, 1, 1e-6, 1000)
        x_corr, tp = correct_x_2d(self.x[:, :3].copy(), w_o, p_o)
        with np.printoptions(precision=3, suppress=True):
            print(x_corr)
            print(tp)


if __name__ == '__main__':
    unittest.main()
