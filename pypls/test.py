import numpy as np
import pretreatment
from unittest import TestCase


class Test(TestCase):
    def test_autoscaling(self):
        """
        Test autoscaling
        :return: None
        """
        x = np.random.randn(30)
        x = pretreatment.autoscaling(x)
        print(x.mean(), x.std())

        x = np.random.randn(30, 3)
        x = pretreatment.autoscaling(x)
        print(x.mean(axis=0), x.std(axis=0))

    def test_paretoscaling(self):
        """ Test pareto scaling """
        x = np.random.randn(30)
        x = pretreatment.paretoscaling(x)
        print(x.mean(), x.std())

        x = np.random.randn(30, 3)
        x = pretreatment.paretoscaling(x)
        print(x.mean(axis=0), x.std(axis=0))

    def test_minmaxscaling(self):
        """ Test minmax scaling """
        x = np.random.randn(30)
        x = pretreatment.minmaxscaling(x)
        print(x.max(), x.min())

        x = np.random.randn(30, 3)
        x = pretreatment.minmaxscaling(x)
        print(x.max(axis=0), x.min(axis=0))

