import numpy as np
import unittest

from pypls import pretreatment


class TestPretreatment(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(30, 10).astype(np.float64)
        self.y = np.ones(30, dtype=np.float64)
        self.y[:10] = -1.
        self.ix = self.y == 1.

    def test_autoscaling(self):
        """
        Test autoscaling
        :return: None
        """
        scaler = pretreatment.Scaler(scaler="uv")
        xs = scaler.fit(self.x.copy(), self.y)

        m1 = self.x[self.ix].mean(axis=0)
        m2 = self.x[~self.ix].mean(axis=0)
        m = (m1 + m2)/2.
        xs2 = (self.x - m) / self.x.std(axis=0)

        self.assertTrue(np.allclose(xs, xs2))

    def test_paretoscaling(self):
        """ Test pareto scaling """
        scaler = pretreatment.Scaler(scaler="pareto")
        xs = scaler.fit(self.x.copy(), self.y)

        m1 = self.x[self.ix].mean(axis=0)
        m2 = self.x[~self.ix].mean(axis=0)
        m = (m1 + m2) / 2.
        xs2 = (self.x - m) / np.sqrt(self.x.std(axis=0))

        self.assertTrue(np.allclose(xs, xs2))

    def test_minmaxscaling(self):
        """ Test minmax scaling """
        scaler = pretreatment.Scaler(scaler="minmax")
        xs = scaler.fit(self.x.copy(), self.y)

        m1 = np.max(self.x, axis=0)
        m2 = np.min(self.x, axis=0)
        xs2 = (self.x - m2) / (m1 - m2)

        self.assertTrue(np.allclose(xs, xs2))

    def test_centering(self):
        """ Test centering """
        scaler = pretreatment.Scaler(scaler="mean")
        xs = scaler.fit(self.x.copy(), self.y)

        m1 = self.x[self.ix].mean(axis=0)
        m2 = self.x[~self.ix].mean(axis=0)
        m = (m1 + m2) / 2.
        xs2 = self.x - m

        self.assertTrue(np.allclose(xs, xs2))

    def test_autoscaling_balanced(self):
        """ Test autoscaling balanced """
        scaler = pretreatment.Scaler(scaler="uv", fit_type="balanced")
        xs = scaler.fit(self.x.copy(), self.y)

        m = self.x.mean(axis=0)
        xs2 = (self.x - m) / self.x.std(axis=0)

        self.assertTrue(np.allclose(xs, xs2))


if __name__ == '__main__':
    unittest.main()
