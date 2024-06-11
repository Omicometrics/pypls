import numpy as np
from unittest import TestCase

from .nipals import nipals
from .pls import pls_c, pls_vip

import matplotlib.pyplot as plt


class TestPLS(TestCase):
    def setUp(self):
        self.x = np.ascontiguousarray([[18.7, 26.8, 42.1, 56.6, 70.0, 83.2],
                                       [31.3, 33.4, 45.7, 49.3, 53.8, 55.3],
                                       [30.0, 35.1, 48.3, 53.5, 59.2, 57.7],
                                       [20.0, 25.7, 39.3, 46.6, 56.5, 57.8],
                                       [31.5, 34.8, 46.5, 46.7, 48.5, 51.1],
                                       [22.0, 28.0, 38.5, 46.7, 54.1, 53.6],
                                       [25.7, 31.4, 41.1, 50.6, 53.5, 49.3],
                                       [18.7, 26.8, 37.8, 50.6, 65.0, 72.3],
                                       [27.3, 34.6, 47.8, 55.9, 67.9, 75.2],
                                       [18.3, 22.8, 32.8, 43.4, 49.6, 51.1]],
                                      dtype=np.float64)
        self.y = np.fromiter(
            [0.89, 0.46, 0.45, 0.56, 0.41, 0.44, 0.34, 0.74, 0.75, 0.48],
            dtype=np.float64
        )

    def test_nipals(self):
        w, t, c = nipals(self.x, self.y)
        self.assertTrue((np.linalg.norm(w) - 1.) <= 1e-6)

    def test_pls(self):
        # standardize the matrix
        xs = (self.x - self.x.mean(axis=0)) / self.x.std(axis=0)
        ys = self.y - self.y.mean(axis=0)
        num_comp: int = 4
        t, w, p, c, coefs = pls_c(xs.copy(), ys.copy(), num_comp)
        yp = np.dot(xs, coefs.T)
        r2 = (1. - ((ys - yp[:, num_comp-1]) ** 2).sum()/(ys ** 2).sum())
        self.assertTrue(abs(r2 - 0.996270) <= 0.0001)

        # fig, ax = plt.subplots()
        # ax.plot(ys, yp[:, num_comp - 1], ".")
        # plt.show()

    def test_pls_vip(self):
        xs = (self.x - self.x.mean(axis=0)) / self.x.std(axis=0)
        ys = self.y - self.y.mean(axis=0)
        num_comp: int = 4
        t, w, p, c, coefs = pls_c(xs.copy(), ys.copy(), num_comp)
        vip = pls_vip(w, t, c)

        with np.printoptions(precision=4, suppress=True):
            print(vip[num_comp - 1])

        fig, ax = plt.subplots()
        ax.bar(np.arange(xs.shape[1]), vip[num_comp - 1])
        plt.show()


if __name__ == '__main__':
    unittest.main()
